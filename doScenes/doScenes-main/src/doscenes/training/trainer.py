from __future__ import annotations

import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from doscenes.config import AppConfig
from doscenes.data.dataset import build_nuscenes_dataset, collate_batch, split_indices_by_scene_hash
from doscenes.evaluation.evaluator import evaluate_model
from doscenes.evaluation.metrics import ade_fde, trajectory_loss_l2
from doscenes.models.trajectory_text_model import TrajectoryTextModel
from doscenes.utils import seed_everything, select_device, write_json


def _query_gpu_utilization() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return float(out.strip().splitlines()[0].strip())
    except Exception:
        return None


def _throttle_gpu(max_util_ratio: float, poll_sec: float) -> tuple[bool, float | None]:
    max_util_pct = max_util_ratio * 100.0
    util = _query_gpu_utilization()
    throttled = False
    while util is not None and util > max_util_pct:
        throttled = True
        time.sleep(max(0.05, poll_sec))
        util = _query_gpu_utilization()
    return throttled, util


def build_loaders(cfg: AppConfig) -> tuple[DataLoader, DataLoader]:
    dataset = build_nuscenes_dataset(
        path_file=cfg.data.path_file,
        hist_sec=cfg.data.hist_sec,
        fut_sec=cfg.data.fut_sec,
        sample_freq=cfg.data.sample_freq,
        relative_coords=cfg.data.relative_coords,
        align_heading=cfg.data.align_heading,
        coord_scale=cfg.data.coord_scale,
        window_stride=cfg.data.window_stride,
        keep_empty_instruction=cfg.data.keep_empty_instruction,
    )
    scene_names = [record.scene_name for record in dataset.records]
    train_idx, val_idx = split_indices_by_scene_hash(dataset.window_index, scene_names, cfg.data.val_ratio, cfg.data.split_seed)

    train_subset = Subset(dataset, train_idx)
    train_sampler = None
    train_shuffle = True
    if cfg.data.rebalance_non_empty_instruction:
        weights: list[float] = []
        base_w = 1.0
        pos_w = float(cfg.data.non_empty_instruction_weight)
        for global_idx in train_idx:
            record_idx, _ = dataset.window_index[global_idx]
            has_text = bool(dataset.records[record_idx].instruction)
            weights.append(pos_w if has_text else base_w)
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(train_subset, batch_size=cfg.training.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=cfg.data.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_batch)
    return train_loader, val_loader


def build_model(cfg: AppConfig, device: torch.device) -> TrajectoryTextModel:
    return TrajectoryTextModel(
        text_model_name=cfg.model.text_model_name,
        hidden_dim=cfg.model.hidden_dim,
        pred_steps=cfg.model.pred_steps,
        freeze_text_encoder=cfg.model.freeze_text_encoder,
        local_files_only=cfg.model.local_files_only,
        attention_heads=cfg.model.attention_heads,
        text_unfreeze_last_n_layers=cfg.model.text_unfreeze_last_n_layers,
    ).to(device)


def _assess_overfit(history: list[dict[str, float]]) -> tuple[str, dict[str, float]]:
    if not history:
        return "unknown", {"final_gap": 0.0, "max_gap": 0.0, "worsen_streak": 0.0, "gap_std": 0.0}

    gaps = [float(x["gap_ade"]) for x in history]
    final_gap = gaps[-1]
    max_gap = max(gaps)

    streak = 0
    best_streak = 0
    for i in range(1, len(gaps)):
        if gaps[i] > gaps[i - 1] + 1e-6:
            streak += 1
        else:
            streak = 0
        best_streak = max(best_streak, streak)

    mean_gap = sum(gaps) / len(gaps)
    var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    gap_std = var_gap ** 0.5

    if final_gap > 0.08 or (best_streak >= 2 and final_gap > 0.05) or (max_gap > 0.12 and gap_std > 0.08):
        level = "high"
    elif final_gap > 0.04 or best_streak >= 2 or max_gap > 0.10:
        level = "caution"
    else:
        level = "low"

    return level, {"final_gap": final_gap, "max_gap": max_gap, "worsen_streak": float(best_streak), "gap_std": gap_std}


def train(cfg: AppConfig, show_progress: bool = True, log_interval: int = 20, gpu_max_util: float | None = None, gpu_poll_sec: float | None = None) -> Path:
    seed_everything(cfg.runtime.seed)
    device = select_device(cfg.runtime.device)
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)

    max_util = cfg.runtime.gpu_max_util if gpu_max_util is None else gpu_max_util
    poll_sec = cfg.runtime.gpu_poll_sec if gpu_poll_sec is None else gpu_poll_sec

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    best_ade = float("inf")
    best_epoch = 0
    stale_epochs = 0
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg.training.checkpoint_name

    report_dir = Path(cfg.training.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "train_report.json"

    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(
        f"Rebalance non-empty instruction: {cfg.data.rebalance_non_empty_instruction} "
        f"(weight={cfg.data.non_empty_instruction_weight})"
    )
    print(
        f"Text encoder freeze={cfg.model.freeze_text_encoder}, "
        f"unfreeze_last_n_layers={cfg.model.text_unfreeze_last_n_layers}"
    )
    print(
        f"Dual-head loss weight: baseline_head_loss_weight={cfg.training.baseline_head_loss_weight}, "
        f"rank_loss_weight={cfg.training.rank_loss_weight}, rank_margin={cfg.training.rank_margin}"
    )
    if device.type == "cuda":
        print(f"GPU throttle enabled: max_util={max_util * 100:.1f}% poll={poll_sec:.2f}s")

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_ade_sum = 0.0
        epoch_fde_sum = 0.0
        epoch_rank_sum = 0.0
        seen_batches = 0

        iterator = train_loader
        if show_progress and tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Train {epoch:03d}/{cfg.training.epochs:03d}", leave=False)

        for step, batch in enumerate(iterator, start=1):
            throttled = False
            gpu_util = None
            if device.type == "cuda" and max_util > 0:
                throttled, gpu_util = _throttle_gpu(max_util, poll_sec)

            history_xy = batch["history_xy"].to(device)
            future_xy_gt = batch["future_xy_gt"].to(device)
            instructions = batch["instruction"]
            has_instruction = batch["has_instruction"].to(device)

            pred_inst = model(history_xy, instructions, head="instruction")
            pred_base_full = model(history_xy, ["" for _ in instructions], head="baseline")
            loss_main_inst = trajectory_loss_l2(pred_inst, future_xy_gt)
            loss_main_base = trajectory_loss_l2(pred_base_full, future_xy_gt)
            loss_main = loss_main_inst + cfg.training.baseline_head_loss_weight * loss_main_base
            rank_loss = torch.tensor(0.0, device=device)
            if cfg.training.rank_loss_weight > 0:
                mask = has_instruction > 0
                if bool(mask.any()):
                    pred_inst_masked = pred_inst[mask]
                    gt_inst = future_xy_gt[mask]
                    pred_base = pred_base_full[mask]
                    loss_inst = torch.norm(pred_inst_masked - gt_inst, dim=-1).mean(dim=-1)
                    loss_base = torch.norm(pred_base - gt_inst, dim=-1).mean(dim=-1)
                    rank_loss = F.relu(loss_inst - loss_base + cfg.training.rank_margin).mean()

            loss = loss_main + cfg.training.rank_loss_weight * rank_loss
            ade, fde = ade_fde(pred_inst.detach(), future_xy_gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_ade_sum += float(ade.item())
            epoch_fde_sum += float(fde.item())
            epoch_rank_sum += float(rank_loss.item())
            seen_batches += 1

            if show_progress and tqdm is not None and hasattr(iterator, "set_postfix"):
                postfix = {"loss": f"{loss.item():.4f}", "ade": f"{ade.item():.4f}", "fde": f"{fde.item():.4f}"}
                postfix["rank"] = f"{rank_loss.item():.4f}"
                if gpu_util is not None:
                    postfix["gpu%"] = f"{gpu_util:.0f}"
                iterator.set_postfix(**postfix)

            if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                util_text = f" gpu={gpu_util:.0f}%" if gpu_util is not None else ""
                throttle_text = " throttled=1" if throttled else ""
                print(f"[Epoch {epoch:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f} ade={ade.item():.4f} fde={fde.item():.4f} rank={rank_loss.item():.4f}{util_text}{throttle_text}")

        train_loss = epoch_loss_sum / max(1, seen_batches)
        train_ade = epoch_ade_sum / max(1, seen_batches)
        train_fde = epoch_fde_sum / max(1, seen_batches)
        train_rank = epoch_rank_sum / max(1, seen_batches)

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            ignore_text=False,
            head="instruction",
            show_progress=show_progress,
            log_prefix=f"Val {epoch:03d}",
        )
        gap_ade = float(val_metrics["ade"] - train_ade)

        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_ade": train_ade,
            "train_fde": train_fde,
            "train_rank_loss": train_rank,
            "val_loss": float(val_metrics["loss"]),
            "val_ade": float(val_metrics["ade"]),
            "val_fde": float(val_metrics["fde"]),
            "gap_ade": gap_ade,
        })

        print(
            f"Epoch {epoch:03d} summary | train_ADE={train_ade:.4f} "
            f"val_ADE={val_metrics['ade']:.4f} gap_ADE={gap_ade:+.4f} rank_loss={train_rank:.4f}"
        )

        if val_metrics["ade"] < (best_ade - cfg.training.early_stop_min_delta):
            best_ade = float(val_metrics["ade"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_ade": best_ade, "config": cfg.to_dict()}, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")
        else:
            stale_epochs += 1

        if cfg.training.early_stop_patience > 0 and stale_epochs >= cfg.training.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch:03d} (patience={cfg.training.early_stop_patience})")
            break

    risk_level, risk_stats = _assess_overfit(history)

    report = {
        "best_checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_ade": best_ade,
        "epochs_ran": len(history),
        "early_stop_patience": cfg.training.early_stop_patience,
        "final_gap_ade": risk_stats["final_gap"],
        "max_gap_ade": risk_stats["max_gap"],
        "gap_std": risk_stats["gap_std"],
        "worsen_streak": risk_stats["worsen_streak"],
        "overfit_risk": risk_level,
        "gpu_max_util": max_util,
        "gpu_poll_sec": poll_sec,
        "history": history,
        "baseline_head_loss_weight": cfg.training.baseline_head_loss_weight,
        "rank_loss_weight": cfg.training.rank_loss_weight,
        "rank_margin": cfg.training.rank_margin,
    }
    write_json(report_path, report)
    print(f"Saved train report: {report_path}")

    return ckpt_path


def load_checkpoint_for_eval(cfg: AppConfig, checkpoint: str) -> tuple[torch.nn.Module, DataLoader, torch.device]:
    device = select_device(cfg.runtime.device)
    _, val_loader = build_loaders(cfg)
    model = build_model(cfg, device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return model, val_loader, device

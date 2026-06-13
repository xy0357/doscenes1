from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from doscenes.config import load_config
from doscenes.data.dataset import load_paths
from doscenes.evaluation.diagnostics import language_effect_report, precheck_submission_csv
from doscenes.evaluation.evaluator import evaluate_model
from doscenes.evaluation.metrics import metric_delta
from doscenes.evaluation.submission import export_submission_csv
from doscenes.training.trainer import load_checkpoint_for_eval, train
from doscenes.utils import write_json


def _load_token_list(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token list file not found: {p}")
    if p.suffix.lower() == ".json":
        payload = __import__("json").loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            vals = payload.get("scene_tokens") or payload.get("sample_tokens") or []
            return [str(x).strip() for x in vals if str(x).strip()]
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        if "," in first_line:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            token_col = "scene_token" if "scene_token" in fieldnames else "sample_token"
            out: list[str] = []
            for row in reader:
                v = str(row.get(token_col, "")).strip()
                if v:
                    out.append(v)
            return out
        return [x.strip() for x in f.readlines() if x.strip() and not x.strip().startswith("#")]


def _aggregate_one_row_per_scene(
    all_tokens: list[str],
    all_instructions: list[str],
    all_preds: torch.Tensor,
    ordered_scene_tokens: list[str],
    prefer_non_empty_instruction: bool,
) -> tuple[list[str], list[str], torch.Tensor, int]:
    by_token: dict[str, list[int]] = {}
    for i, tok in enumerate(all_tokens):
        by_token.setdefault(tok, []).append(i)

    out_tokens: list[str] = []
    out_insts: list[str] = []
    out_preds: list[torch.Tensor] = []
    missing = 0

    for tok in ordered_scene_tokens:
        idxs = by_token.get(tok, [])
        if not idxs:
            missing += 1
            continue
        picked = idxs
        if prefer_non_empty_instruction:
            idxs_non_empty = [i for i in idxs if str(all_instructions[i]).strip()]
            if idxs_non_empty:
                picked = idxs_non_empty
        pred_mean = all_preds[picked].mean(dim=0)
        inst = ""
        if prefer_non_empty_instruction:
            for i in picked:
                text = str(all_instructions[i]).strip()
                if text:
                    inst = text
                    break
        out_tokens.append(tok)
        out_insts.append(inst)
        out_preds.append(pred_mean)

    if out_preds:
        pred_tensor = torch.stack(out_preds, dim=0)
    else:
        pred_tensor = torch.zeros((0, all_preds.size(1), all_preds.size(2)), dtype=all_preds.dtype)
    return out_tokens, out_insts, pred_tensor, missing


def _yaw_from_wxyz(rotation_wxyz: list[float]) -> float:
    try:
        from pyquaternion import Quaternion
    except Exception:
        return 0.0
    q = Quaternion(rotation_wxyz)
    w, x, y, z = q.elements
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _build_history_xy_from_scene_token(
    nusc,
    scene_token: str,
    hist_frames: int,
    relative_coords: bool,
    align_heading: bool,
    coord_scale: float,
) -> torch.Tensor:
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]
    xy: list[list[float]] = []
    yaws: list[float] = []

    while token and len(xy) < hist_frames:
        sample = nusc.get("sample", token)
        sd_token = sample["data"].get("LIDAR_TOP") or sample["data"].get("CAM_FRONT")
        sd = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        xy.append([float(ego_pose["translation"][0]), float(ego_pose["translation"][1])])
        yaws.append(_yaw_from_wxyz(ego_pose["rotation"]))
        token = sample["next"]

    if len(xy) < hist_frames:
        raise ValueError(f"Scene {scene_token} does not have enough samples for history_len={hist_frames}")

    history_xy = torch.tensor(xy, dtype=torch.float32)
    heading = float(yaws[-1])
    origin = history_xy[-1].clone()

    if relative_coords:
        history_xy -= origin
        if align_heading:
            cos_y = math.cos(-heading)
            sin_y = math.sin(-heading)
            rot = torch.tensor([[cos_y, -sin_y], [sin_y, cos_y]], dtype=torch.float32)
            history_xy = history_xy @ rot.T

    history_xy /= float(coord_scale)
    return history_xy


def _export_submission_official_scene_map(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model, _, device = load_checkpoint_for_eval(cfg, args.checkpoint)
    model.eval()

    nusc_root, _ = load_paths(cfg.data.path_file)
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.nuscenes_version, dataroot=nusc_root, verbose=False)

    map_path = Path(args.official_scene_map_csv)
    if not map_path.exists():
        raise FileNotFoundError(f"Official scene map csv not found: {map_path}")

    rows: list[dict[str, str]] = []
    with map_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tok = str(row.get("scene_token", "")).strip()
            inst = str(row.get("instruction", "")).strip()
            if tok:
                rows.append({"scene_token": tok, "instruction": inst})

    if not rows:
        raise ValueError(f"No valid rows found in official scene map: {map_path}")
    if args.expected_scenes is not None and len(rows) != args.expected_scenes:
        raise ValueError(f"Official scene map row count {len(rows)} != expected {args.expected_scenes}")

    sample_tokens: list[str] = []
    instructions_all: list[str] = []
    preds_inst: list[torch.Tensor] = []
    preds_base: list[torch.Tensor] = []
    hist_frames = int(cfg.data.hist_sec * cfg.data.sample_freq) + 1

    iterator = rows
    if args.progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(rows, desc="Export official", leave=False)
        except Exception:
            iterator = rows

    with torch.no_grad():
        for row in iterator:
            tok = row["scene_token"]
            inst = row["instruction"]
            history_xy = _build_history_xy_from_scene_token(
                nusc=nusc,
                scene_token=tok,
                hist_frames=hist_frames,
                relative_coords=cfg.data.relative_coords,
                align_heading=cfg.data.align_heading,
                coord_scale=cfg.data.coord_scale,
            ).unsqueeze(0).to(device)

            pred_i = model(history_xy, [inst if not args.ignore_text else ""], head="instruction").cpu() * float(cfg.data.coord_scale)
            pred_b = model(history_xy, [""], head="baseline").cpu() * float(cfg.data.coord_scale)

            sample_tokens.append(tok)
            instructions_all.append(inst if not args.ignore_text else "")
            preds_inst.append(pred_i.squeeze(0))
            preds_base.append(pred_b.squeeze(0))

    pred_inst_tensor = torch.stack(preds_inst, dim=0)
    output, raw_rows, dedup_rows = export_submission_csv(sample_tokens, instructions_all, pred_inst_tensor, args.output)
    print(f"Saved submission csv: {output}")
    print(f"Submission rows (raw -> dedup): {raw_rows} -> {dedup_rows}")
    print(f"Official scene map used: {map_path}")

    if args.baseline_output:
        pred_base_tensor = torch.stack(preds_base, dim=0)
        baseline_insts = ["" for _ in sample_tokens]
        out_b, raw_b, dedup_b = export_submission_csv(sample_tokens, baseline_insts, pred_base_tensor, args.baseline_output)
        print(f"Saved baseline csv: {out_b}")
        print(f"Baseline rows (raw -> dedup): {raw_b} -> {dedup_b}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ckpt = train(cfg, show_progress=args.progress, log_interval=args.log_interval, gpu_max_util=args.gpu_max_util, gpu_poll_sec=args.gpu_poll_sec)
    print(f"Training finished. Best checkpoint: {ckpt}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model, val_loader, device = load_checkpoint_for_eval(cfg, args.checkpoint)
    eval_head = "baseline" if args.ignore_text else "instruction"
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        ignore_text=args.ignore_text,
        head=eval_head,
        show_progress=args.progress,
        log_prefix="Eval",
    )
    metrics["checkpoint"] = args.checkpoint
    metrics["mode"] = "baseline" if args.ignore_text else "instruction"
    write_json(args.output, metrics)
    print(f"Saved evaluation report: {args.output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base = __import__("json").loads(Path(args.baseline).read_text(encoding="utf-8-sig"))
    inst = __import__("json").loads(Path(args.instruction).read_text(encoding="utf-8-sig"))
    payload = {"baseline": base, "instruction": inst, "delta_ade": metric_delta(base["ade"], inst["ade"]), "delta_fde": float(base["fde"] - inst["fde"])}
    if "ade_m" in base and "ade_m" in inst:
        payload["delta_ade_m"] = metric_delta(base["ade_m"], inst["ade_m"])
    if "fde_m" in base and "fde_m" in inst:
        payload["delta_fde_m"] = float(base["fde_m"] - inst["fde_m"])
    write_json(args.output, payload)
    print(f"Saved delta report: {args.output}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results: list[dict[str, float]] = []

    for seed in seed_values:
        cfg = load_config(args.config)
        cfg = cfg.__class__(data=cfg.data, model=cfg.model, training=cfg.training, runtime=cfg.runtime.__class__(seed=seed, device=cfg.runtime.device, gpu_max_util=cfg.runtime.gpu_max_util, gpu_poll_sec=cfg.runtime.gpu_poll_sec))

        print(f"=== Seed {seed} ===")
        ckpt = train(cfg, show_progress=args.progress, log_interval=args.log_interval, gpu_max_util=args.gpu_max_util, gpu_poll_sec=args.gpu_poll_sec)
        model, val_loader, device = load_checkpoint_for_eval(cfg, str(ckpt))

        instruction = evaluate_model(
            model,
            val_loader,
            device,
            ignore_text=False,
            head="instruction",
            show_progress=args.progress,
            log_prefix=f"Eval instruction s{seed}",
        )
        baseline = evaluate_model(
            model,
            val_loader,
            device,
            ignore_text=True,
            head="baseline",
            show_progress=args.progress,
            log_prefix=f"Eval baseline s{seed}",
        )
        delta_ade = metric_delta(float(baseline["ade"]), float(instruction["ade"]))
        delta_fde = float(baseline["fde"] - instruction["fde"])
        delta_ade_m = metric_delta(float(baseline["ade_m"]), float(instruction["ade_m"]))
        delta_fde_m = float(baseline["fde_m"] - instruction["fde_m"])

        results.append({
            "seed": float(seed),
            "ade_instruction": float(instruction["ade"]),
            "fde_instruction": float(instruction["fde"]),
            "ade_baseline": float(baseline["ade"]),
            "fde_baseline": float(baseline["fde"]),
            "delta_ade": delta_ade,
            "delta_fde": delta_fde,
            "ade_instruction_m": float(instruction["ade_m"]),
            "fde_instruction_m": float(instruction["fde_m"]),
            "ade_baseline_m": float(baseline["ade_m"]),
            "fde_baseline_m": float(baseline["fde_m"]),
            "delta_ade_m": delta_ade_m,
            "delta_fde_m": delta_fde_m,
        })

    def mean_std(key: str) -> dict[str, float]:
        vals = [x[key] for x in results]
        return {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0}

    summary = {
        "runs": results,
        "summary": {
            "ade_instruction": mean_std("ade_instruction"),
            "fde_instruction": mean_std("fde_instruction"),
            "ade_baseline": mean_std("ade_baseline"),
            "fde_baseline": mean_std("fde_baseline"),
            "delta_ade": mean_std("delta_ade"),
            "delta_fde": mean_std("delta_fde"),
            "ade_instruction_m": mean_std("ade_instruction_m"),
            "fde_instruction_m": mean_std("fde_instruction_m"),
            "ade_baseline_m": mean_std("ade_baseline_m"),
            "fde_baseline_m": mean_std("fde_baseline_m"),
            "delta_ade_m": mean_std("delta_ade_m"),
            "delta_fde_m": mean_std("delta_fde_m"),
        },
    }
    write_json(args.output, summary)
    print(f"Saved benchmark report: {args.output}")
    return 0


def cmd_language_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model, val_loader, device = load_checkpoint_for_eval(cfg, args.checkpoint)
    report = language_effect_report(
        model=model,
        loader=val_loader,
        device=device,
        out_json=args.output_json,
        out_csv=args.output_csv,
    )
    print(f"Saved language effect report: {args.output_json}")
    print(f"Saved language effect samples: {args.output_csv}")
    print(f"Overall delta_ade={report['overall']['delta_ade']:.6f}")
    return 0


def cmd_precheck_submission(args: argparse.Namespace) -> int:
    result = precheck_submission_csv(
        submission_csv=args.submission,
        expected_steps=args.expected_steps,
        expected_rows=args.expected_rows,
        expected_tokens_file=args.expected_tokens_file,
        out_json=args.output,
    )
    print(f"Saved submission precheck report: {args.output}")
    print(f"Precheck ok={result['ok']}")
    if result["issues"]:
        print("Issues:")
        for x in result["issues"]:
            print(f"- {x}")
    return 0


def cmd_export_submission(args: argparse.Namespace) -> int:
    if args.official_scene_map_csv:
        return _export_submission_official_scene_map(args)

    cfg = load_config(args.config)
    model, val_loader, device = load_checkpoint_for_eval(cfg, args.checkpoint)
    model.eval()

    sample_tokens: list[str] = []
    instructions_all: list[str] = []
    preds_all: list[torch.Tensor] = []
    preds_all_baseline: list[torch.Tensor] = []

    iterator = val_loader
    if args.progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(val_loader, desc="Export", leave=False)
        except Exception:
            iterator = val_loader

    with torch.no_grad():
        for batch in iterator:
            history_xy = batch["history_xy"].to(device)
            inst_batch = batch["instruction"]
            if args.ignore_text:
                pred_inst = model(history_xy, ["" for _ in inst_batch], head="instruction").cpu()
            else:
                pred_inst = model(history_xy, inst_batch, head="instruction").cpu()
            pred_inst = pred_inst * float(batch.get("coord_scale", 1.0))
            batch_tokens = batch.get("scene_token", batch["sample_id"])
            sample_tokens.extend(batch_tokens)
            instructions_all.extend(inst_batch)
            preds_all.append(pred_inst)
            if args.baseline_output:
                pred_base = model(history_xy, ["" for _ in inst_batch], head="baseline").cpu()
                pred_base = pred_base * float(batch.get("coord_scale", 1.0))
                preds_all_baseline.append(pred_base)

    preds_tensor = torch.cat(preds_all, dim=0)
    export_tokens = sample_tokens
    export_insts = instructions_all
    export_preds = preds_tensor

    missing = 0
    if args.scene_token_list:
        ordered_tokens = _load_token_list(args.scene_token_list)
        export_tokens, export_insts, export_preds, missing = _aggregate_one_row_per_scene(
            all_tokens=sample_tokens,
            all_instructions=instructions_all,
            all_preds=preds_tensor,
            ordered_scene_tokens=ordered_tokens,
            prefer_non_empty_instruction=not args.ignore_text,
        )
        if args.expected_scenes is not None and len(export_tokens) != args.expected_scenes:
            raise ValueError(
                f"Official export rows {len(export_tokens)} != expected {args.expected_scenes}; missing={missing}. "
                "Please ensure scene_token_list matches the dataloader official test scene list."
            )

    output, raw_rows, dedup_rows = export_submission_csv(export_tokens, export_insts, export_preds, args.output)
    print(f"Saved submission csv: {output}")
    print(f"Submission rows (raw -> dedup): {raw_rows} -> {dedup_rows}")
    if args.scene_token_list:
        print(f"Official scene list used: {args.scene_token_list}")
        print(f"Missing scene_token in model export: {missing}")

    if args.baseline_output:
        if not preds_all_baseline:
            raise RuntimeError("baseline_output requested but no baseline predictions were collected")
        preds_tensor_base = torch.cat(preds_all_baseline, dim=0)
        b_tokens = sample_tokens
        b_insts = ["" for _ in instructions_all]
        b_preds = preds_tensor_base
        b_missing = 0
        if args.scene_token_list:
            ordered_tokens = _load_token_list(args.scene_token_list)
            b_tokens, b_insts, b_preds, b_missing = _aggregate_one_row_per_scene(
                all_tokens=sample_tokens,
                all_instructions=b_insts,
                all_preds=preds_tensor_base,
                ordered_scene_tokens=ordered_tokens,
                prefer_non_empty_instruction=False,
            )
            if args.expected_scenes is not None and len(b_tokens) != args.expected_scenes:
                raise ValueError(
                    f"Baseline export rows {len(b_tokens)} != expected {args.expected_scenes}; missing={b_missing}. "
                    "Please ensure scene_token_list matches the dataloader official test scene list."
                )
        out_b, raw_b, dedup_b = export_submission_csv(b_tokens, b_insts, b_preds, args.baseline_output)
        print(f"Saved baseline csv: {out_b}")
        print(f"Baseline rows (raw -> dedup): {raw_b} -> {dedup_b}")
        if args.scene_token_list:
            print(f"Baseline missing scene_token in model export: {b_missing}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="doScenes formal CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default="configs/default.json")
    p_train.add_argument("--progress", action="store_true", default=True)
    p_train.add_argument("--log-interval", type=int, default=0)
    p_train.add_argument("--gpu-max-util", type=float, default=None, help="max GPU utilization ratio, e.g. 0.8")
    p_train.add_argument("--gpu-poll-sec", type=float, default=None, help="GPU utilization polling interval seconds")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--config", default="configs/default.json")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--ignore-text", action="store_true")
    p_eval.add_argument("--output", default="artifacts/experiments/eval_report.json")
    p_eval.add_argument("--progress", action="store_true", default=True)
    p_eval.set_defaults(func=cmd_eval)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--baseline", required=True)
    p_compare.add_argument("--instruction", required=True)
    p_compare.add_argument("--output", default="artifacts/experiments/delta_report.json")
    p_compare.set_defaults(func=cmd_compare)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--config", default="configs/default.json")
    p_bench.add_argument("--seeds", default="42,43,44")
    p_bench.add_argument("--output", default="artifacts/experiments/benchmark_report.json")
    p_bench.add_argument("--progress", action="store_true", default=True)
    p_bench.add_argument("--log-interval", type=int, default=0)
    p_bench.add_argument("--gpu-max-util", type=float, default=None)
    p_bench.add_argument("--gpu-poll-sec", type=float, default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    p_lang = sub.add_parser("language-report")
    p_lang.add_argument("--config", default="configs/default.json")
    p_lang.add_argument("--checkpoint", required=True)
    p_lang.add_argument("--output-json", default="artifacts/experiments/language_effect_report.json")
    p_lang.add_argument("--output-csv", default="artifacts/experiments/language_effect_samples.csv")
    p_lang.set_defaults(func=cmd_language_report)

    p_pre = sub.add_parser("precheck-submission")
    p_pre.add_argument("--submission", default="artifacts/submissions/submission.csv")
    p_pre.add_argument("--expected-steps", type=int, default=12)
    p_pre.add_argument("--expected-rows", type=int, default=None)
    p_pre.add_argument("--expected-tokens-file", type=str, default=None)
    p_pre.add_argument("--output", default="artifacts/experiments/submission_precheck_report.json")
    p_pre.set_defaults(func=cmd_precheck_submission)

    p_sub = sub.add_parser("export-submission")
    p_sub.add_argument("--config", default="configs/default.json")
    p_sub.add_argument("--checkpoint", required=True)
    p_sub.add_argument("--ignore-text", action="store_true")
    p_sub.add_argument("--output", default="artifacts/submissions/submission.csv")
    p_sub.add_argument("--baseline-output", type=str, default=None, help="optional matched baseline csv output path")
    p_sub.add_argument("--scene-token-list", type=str, default=None, help="official scene_token list file (txt/csv/json)")
    p_sub.add_argument("--official-scene-map-csv", type=str, default=None, help="official scene map csv with columns: scene_token,instruction")
    p_sub.add_argument("--nuscenes-version", type=str, default="v1.0-test", help="nuscenes version for official scene map export")
    p_sub.add_argument("--expected-scenes", type=int, default=None, help="expected number of exported rows in official mode")
    p_sub.add_argument("--progress", action="store_true", default=True)
    p_sub.set_defaults(func=cmd_export_submission)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

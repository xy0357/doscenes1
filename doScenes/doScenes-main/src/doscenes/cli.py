from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from doscenes.config import load_config
from doscenes.evaluation.diagnostics import language_effect_report, precheck_submission_csv
from doscenes.evaluation.evaluator import evaluate_model
from doscenes.evaluation.metrics import metric_delta
from doscenes.evaluation.submission import export_submission_csv
from doscenes.training.trainer import load_checkpoint_for_eval, train
from doscenes.utils import write_json


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
    cfg = load_config(args.config)
    model, val_loader, device = load_checkpoint_for_eval(cfg, args.checkpoint)
    model.eval()

    sample_tokens: list[str] = []
    instructions_all: list[str] = []
    preds_all: list[torch.Tensor] = []

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
            instructions = ["" for _ in batch["instruction"]] if args.ignore_text else batch["instruction"]
            pred = model(history_xy, instructions, head="instruction").cpu()
            pred = pred * float(batch.get("coord_scale", 1.0))
            batch_tokens = batch.get("scene_token", batch["sample_id"])
            sample_tokens.extend(batch_tokens)
            instructions_all.extend(batch["instruction"])
            preds_all.append(pred)

    preds_tensor = torch.cat(preds_all, dim=0)
    output, raw_rows, dedup_rows = export_submission_csv(sample_tokens, instructions_all, preds_tensor, args.output)
    print(f"Saved submission csv: {output}")
    print(f"Submission rows (raw -> dedup): {raw_rows} -> {dedup_rows}")
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
    p_pre.add_argument("--output", default="artifacts/experiments/submission_precheck_report.json")
    p_pre.set_defaults(func=cmd_precheck_submission)

    p_sub = sub.add_parser("export-submission")
    p_sub.add_argument("--config", default="configs/default.json")
    p_sub.add_argument("--checkpoint", required=True)
    p_sub.add_argument("--ignore-text", action="store_true")
    p_sub.add_argument("--output", default="artifacts/submissions/submission.csv")
    p_sub.add_argument("--progress", action="store_true", default=True)
    p_sub.set_defaults(func=cmd_export_submission)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

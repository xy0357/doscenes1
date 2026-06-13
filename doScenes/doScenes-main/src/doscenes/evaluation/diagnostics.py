from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from doscenes.evaluation.metrics import ade_fde, metric_delta


@torch.no_grad()
def language_effect_report(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_json: str,
    out_csv: str,
) -> dict[str, Any]:
    model.eval()

    rows: list[dict[str, Any]] = []
    grouped = {
        "all": {"count": 0, "ade_inst": 0.0, "fde_inst": 0.0, "ade_base": 0.0, "fde_base": 0.0, "ade_inst_m": 0.0, "fde_inst_m": 0.0, "ade_base_m": 0.0, "fde_base_m": 0.0},
        "has_instruction": {"count": 0, "ade_inst": 0.0, "fde_inst": 0.0, "ade_base": 0.0, "fde_base": 0.0, "ade_inst_m": 0.0, "fde_inst_m": 0.0, "ade_base_m": 0.0, "fde_base_m": 0.0},
        "empty_instruction": {"count": 0, "ade_inst": 0.0, "fde_inst": 0.0, "ade_base": 0.0, "fde_base": 0.0, "ade_inst_m": 0.0, "fde_inst_m": 0.0, "ade_base_m": 0.0, "fde_base_m": 0.0},
    }

    for batch in loader:
        history_xy = batch["history_xy"].to(device)
        future_xy_gt = batch["future_xy_gt"].to(device)
        instructions = batch["instruction"]
        has_instruction = batch.get("has_instruction")
        coord_scale = float(batch.get("coord_scale", 1.0))
        if has_instruction is None:
            has_instruction = torch.tensor([1 if str(x).strip() else 0 for x in instructions], dtype=torch.long)

        pred_inst = model(history_xy, instructions, head="instruction")
        pred_base = model(history_xy, ["" for _ in instructions], head="baseline")

        disp_inst = torch.norm(pred_inst - future_xy_gt, dim=-1)
        disp_base = torch.norm(pred_base - future_xy_gt, dim=-1)

        ade_inst_per = disp_inst.mean(dim=1)
        fde_inst_per = disp_inst[:, -1]
        ade_base_per = disp_base.mean(dim=1)
        fde_base_per = disp_base[:, -1]

        for i in range(history_xy.size(0)):
            sid = batch["sample_id"][i]
            flag = int(has_instruction[i].item())
            ai = float(ade_inst_per[i].item())
            fi = float(fde_inst_per[i].item())
            ab = float(ade_base_per[i].item())
            fb = float(fde_base_per[i].item())
            da = ab - ai
            df = fb - fi
            ai_m = ai * coord_scale
            fi_m = fi * coord_scale
            ab_m = ab * coord_scale
            fb_m = fb * coord_scale

            rows.append(
                {
                    "sample_id": sid,
                    "has_instruction": flag,
                    "ade_instruction_m": ai_m,
                    "fde_instruction_m": fi_m,
                    "ade_baseline_m": ab_m,
                    "fde_baseline_m": fb_m,
                    "delta_ade_m": ab_m - ai_m,
                    "delta_fde_m": fb_m - fi_m,
                    "ade_instruction_norm": ai,
                    "fde_instruction_norm": fi,
                    "ade_baseline_norm": ab,
                    "fde_baseline_norm": fb,
                    "delta_ade_norm": da,
                    "delta_fde_norm": df,
                }
            )

            for key in ["all", "has_instruction" if flag == 1 else "empty_instruction"]:
                grouped[key]["count"] += 1
                grouped[key]["ade_inst"] += ai
                grouped[key]["fde_inst"] += fi
                grouped[key]["ade_base"] += ab
                grouped[key]["fde_base"] += fb
                grouped[key]["ade_inst_m"] += ai_m
                grouped[key]["fde_inst_m"] += fi_m
                grouped[key]["ade_base_m"] += ab_m
                grouped[key]["fde_base_m"] += fb_m

    def finalize(g: dict[str, float]) -> dict[str, float]:
        n = max(1, int(g["count"]))
        ade_i = g["ade_inst"] / n
        fde_i = g["fde_inst"] / n
        ade_b = g["ade_base"] / n
        fde_b = g["fde_base"] / n
        ade_i_m = g["ade_inst_m"] / n
        fde_i_m = g["fde_inst_m"] / n
        ade_b_m = g["ade_base_m"] / n
        fde_b_m = g["fde_base_m"] / n
        return {
            "count": int(g["count"]),
            "ade_instruction": ade_i_m,
            "fde_instruction": fde_i_m,
            "ade_baseline": ade_b_m,
            "fde_baseline": fde_b_m,
            "delta_ade": metric_delta(ade_b_m, ade_i_m),
            "delta_fde": float(fde_b_m - fde_i_m),
            "ade_instruction_m": ade_i_m,
            "fde_instruction_m": fde_i_m,
            "ade_baseline_m": ade_b_m,
            "fde_baseline_m": fde_b_m,
            "delta_ade_m": metric_delta(ade_b_m, ade_i_m),
            "delta_fde_m": float(fde_b_m - fde_i_m),
            "ade_instruction_norm": ade_i,
            "fde_instruction_norm": fde_i,
            "ade_baseline_norm": ade_b,
            "fde_baseline_norm": fde_b,
            "delta_ade_norm": metric_delta(ade_b, ade_i),
            "delta_fde_norm": float(fde_b - fde_i),
        }

    report = {
        "overall": finalize(grouped["all"]),
        "has_instruction": finalize(grouped["has_instruction"]),
        "empty_instruction": finalize(grouped["empty_instruction"]),
        "sample_count": len(rows),
    }

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return report


def precheck_submission_csv(
    submission_csv: str,
    expected_steps: int = 12,
    expected_rows: int | None = None,
    expected_tokens_file: str | None = None,
    out_json: str | None = None,
) -> dict[str, Any]:
    path = Path(submission_csv)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Submission file is empty")

    header = rows[0]
    data = rows[1:]

    expected_cols = 2 + expected_steps * 2
    issues: list[str] = []

    token_col = header[0] if len(header) > 0 else ""
    instruction_col = header[1] if len(header) > 1 else ""
    if token_col != "sample_token":
        issues.append(f"Unexpected first column: {token_col} (expected sample_token)")
    if instruction_col != "instruction":
        issues.append(f"Unexpected second column: {instruction_col} (expected instruction)")

    if len(header) != expected_cols:
        issues.append(f"Header column count {len(header)} != expected {expected_cols}")

    seen = set()
    dup = 0
    nan_cells = 0
    bad_rows = 0

    for r in data:
        if len(r) != len(header):
            bad_rows += 1
            continue
        key = (r[0], r[1] if len(r) > 1 else "")
        if key in seen:
            dup += 1
        seen.add(key)

        for cell in r[2:]:
            c = cell.strip()
            if c == "" or c.lower() in {"nan", "none", "null"}:
                nan_cells += 1
                continue
            try:
                float(c)
            except Exception:
                nan_cells += 1

    if expected_rows is not None and len(data) != expected_rows:
        issues.append(f"Row count {len(data)} != expected {expected_rows}")

    expected_tokens: list[str] | None = None
    if expected_tokens_file is not None:
        expected_path = Path(expected_tokens_file)
        if not expected_path.exists():
            issues.append(f"Expected tokens file not found: {expected_path}")
        else:
            if expected_path.suffix.lower() == ".json":
                payload = json.loads(expected_path.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict):
                    vals = payload.get("scene_tokens") or payload.get("sample_tokens") or []
                    expected_tokens = [str(x).strip() for x in vals if str(x).strip()]
                elif isinstance(payload, list):
                    expected_tokens = [str(x).strip() for x in payload if str(x).strip()]
                else:
                    expected_tokens = []
            else:
                with expected_path.open("r", encoding="utf-8-sig", newline="") as f:
                    first_line = f.readline()
                    f.seek(0)
                    if "," in first_line:
                        reader = csv.DictReader(f)
                        fieldnames = reader.fieldnames or []
                        token_col = "scene_token" if "scene_token" in fieldnames else "sample_token"
                        expected_tokens = []
                        for row in reader:
                            v = str(row.get(token_col, "")).strip()
                            if v:
                                expected_tokens.append(v)
                    else:
                        expected_tokens = [x.strip() for x in f.readlines() if x.strip() and not x.strip().startswith("#")]
            if expected_tokens is not None:
                expected_set = set(expected_tokens)
                actual_set = {r[0] for r in data if len(r) > 0}
                missing = expected_set - actual_set
                extra = actual_set - expected_set
                if missing:
                    issues.append(f"Missing sample_token(s): {len(missing)}")
                if extra:
                    issues.append(f"Unexpected sample_token(s): {len(extra)}")

    if dup > 0:
        issues.append(f"Duplicate token rows: {dup}")
    if bad_rows > 0:
        issues.append(f"Rows with wrong column count: {bad_rows}")
    if nan_cells > 0:
        issues.append(f"Invalid numeric cells: {nan_cells}")

    result = {
        "file": str(path),
        "rows": len(data),
        "columns": len(header),
        "token_column": token_col,
        "expected_steps": expected_steps,
        "coordinate_unit": "meters",
        "coordinate_frame": "ego_vehicle_at_prediction_time",
        "expected_tokens_file": expected_tokens_file,
        "expected_token_count": len(expected_tokens) if expected_tokens is not None else None,
        "duplicate_rows": dup,
        "bad_rows": bad_rows,
        "invalid_numeric_cells": nan_cells,
        "ok": len(issues) == 0,
        "issues": issues,
    }

    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result

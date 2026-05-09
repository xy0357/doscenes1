from __future__ import annotations

import csv
from pathlib import Path

import torch


def export_submission_csv(
    sample_tokens: list[str],
    instructions: list[str],
    preds: torch.Tensor,
    output_csv: str,
) -> tuple[Path, int, int]:
    if preds.dim() != 3 or preds.size(-1) != 2:
        raise ValueError("preds must be [B, T, 2]")
    if len(sample_tokens) != preds.size(0):
        raise ValueError("sample_tokens length must match preds batch dimension")
    if len(instructions) != preds.size(0):
        raise ValueError("instructions length must match preds batch dimension")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    horizon = preds.size(1)
    coord_headers: list[str] = []
    for t in range(1, horizon + 1):
        coord_headers.extend([f"x{t}", f"y{t}"])

    # Deduplicate by (sample_token, instruction): repeated annotator records may
    # map to the same query. We aggregate by mean prediction.
    grouped: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
    for i, token in enumerate(sample_tokens):
        inst = instructions[i]
        key = (token, inst)
        pred_i = preds[i]
        old = grouped.get(key)
        if old is None:
            grouped[key] = (pred_i.clone(), 1)
        else:
            grouped[key] = (old[0] + pred_i, old[1] + 1)

    ordered_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for i, token in enumerate(sample_tokens):
        key = (token, instructions[i])
        if key not in seen_keys:
            ordered_keys.append(key)
            seen_keys.add(key)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_token", "instruction", *coord_headers])
        for sample_token, instruction in ordered_keys:
            summed_pred, cnt = grouped[(sample_token, instruction)]
            pred_mean = summed_pred / float(cnt)
            row = [sample_token, instruction]
            for t in range(horizon):
                row.append(float(pred_mean[t, 0].item()))
                row.append(float(pred_mean[t, 1].item()))
            writer.writerow(row)

    return out_path, len(sample_tokens), len(ordered_keys)

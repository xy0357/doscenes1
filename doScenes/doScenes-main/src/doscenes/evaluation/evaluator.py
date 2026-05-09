from __future__ import annotations

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from doscenes.evaluation.metrics import ade_fde, trajectory_loss_l2


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_text: bool = False,
    head: str = "instruction",
    show_progress: bool = True,
    log_prefix: str = "Eval",
) -> dict[str, float]:
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_loss = 0.0
    count = 0

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=log_prefix, leave=False)

    for batch in iterator:
        history_xy = batch["history_xy"].to(device)
        future_xy_gt = batch["future_xy_gt"].to(device)
        instructions = ["" for _ in batch["instruction"]] if ignore_text else batch["instruction"]

        pred = model(history_xy, instructions, head=head)
        loss = trajectory_loss_l2(pred, future_xy_gt)
        ade, fde = ade_fde(pred, future_xy_gt)

        bs = history_xy.size(0)
        total_loss += float(loss.item()) * bs
        total_ade += float(ade.item()) * bs
        total_fde += float(fde.item()) * bs
        count += bs

        if show_progress and tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}", ade=f"{ade.item():.4f}", fde=f"{fde.item():.4f}")

    if count == 0:
        raise RuntimeError("No samples evaluated")

    return {
        "samples": count,
        "loss": total_loss / count,
        "ade": total_ade / count,
        "fde": total_fde / count,
    }

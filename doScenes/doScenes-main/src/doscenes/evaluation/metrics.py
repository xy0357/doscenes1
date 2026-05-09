from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_loss_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    return F.mse_loss(pred, target)


def ade_fde(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    displacement = torch.norm(pred - target, dim=-1)
    return displacement.mean(), displacement[:, -1].mean()


def metric_delta(baseline_ade: float, instruction_ade: float) -> float:
    return float(baseline_ade - instruction_ade)

from __future__ import annotations

import torch

from doscenes.evaluation.metrics import ade_fde


def test_ade_fde_shapes() -> None:
    pred = torch.zeros(2, 12, 2)
    target = torch.ones(2, 12, 2)
    ade, fde = ade_fde(pred, target)
    assert ade.item() > 0
    assert fde.item() > 0

import pytest

from doscenes.evaluation.metrics import metric_delta


def test_metric_delta_positive_when_instruction_better() -> None:
    assert metric_delta(0.42, 0.35) == 0.07


def test_metric_delta_negative_when_instruction_worse() -> None:
    assert metric_delta(0.31, 0.40) == pytest.approx(-0.09)

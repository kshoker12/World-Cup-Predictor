"""Tests for calibration backtest metrics."""

from __future__ import annotations

import pytest

from worldcup_predictor.simulation.calibration_backtest import compute_calibration_metrics


def test_compute_calibration_metrics():
    probs = {"France": 0.25, "Brazil": 0.20, "Germany": 0.15, "Spain": 0.40}
    metrics = compute_calibration_metrics(probs, "France")
    assert metrics.p_actual_champion == 0.25
    assert metrics.actual_champion_rank == 2
    assert metrics.multiclass_brier == pytest.approx(
        0.75**2 + 0.20**2 + 0.15**2 + 0.40**2
    )
    assert metrics.log_score == pytest.approx(__import__("math").log(0.25))


def test_missing_actual_champion_raises():
    with pytest.raises(ValueError, match="missing"):
        compute_calibration_metrics({"France": 1.0}, "Argentina")

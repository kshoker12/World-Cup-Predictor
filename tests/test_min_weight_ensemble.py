"""Tests for minimum-weight ensemble fitting."""

from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.calibration.ensemble import fit_ensemble_weights


def test_fit_three_way_enforces_min_weight():
    rng = np.random.default_rng(0)
    n = 200
    y_h = rng.integers(0, 4, n)
    y_a = rng.integers(0, 4, n)
    lh_gbm = rng.uniform(0.5, 2.5, n)
    la_gbm = rng.uniform(0.5, 2.5, n)
    lh_nn = rng.uniform(0.5, 2.5, n)
    la_nn = rng.uniform(0.5, 2.5, n)
    lh_bay = rng.uniform(0.5, 2.5, n)
    la_bay = rng.uniform(0.5, 2.5, n)

    weights = fit_ensemble_weights(
        y_h,
        y_a,
        lh_gbm,
        la_gbm,
        lh_nn,
        la_nn,
        lh_bay,
        la_bay,
        min_weight=0.10,
    )
    total = weights.w_gbm + weights.w_nn + weights.w_bayesian
    assert total == pytest.approx(1.0)
    assert weights.w_gbm >= 0.10 - 1e-6
    assert weights.w_nn >= 0.10 - 1e-6
    assert weights.w_bayesian >= 0.10 - 1e-6

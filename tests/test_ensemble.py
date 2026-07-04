"""Tests for three-way ensemble weighting."""

from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.calibration.ensemble import (
    EnsembleParams,
    combine_lambda,
    fit_ensemble_weights,
)


def test_combine_lambda_three_way():
    weights = EnsembleParams(w_gbm=0.5, w_nn=0.3, w_bayesian=0.2)
    lh, la = combine_lambda(
        np.array([2.0]),
        np.array([1.0]),
        weights,
        np.array([3.0]),
        np.array([1.5]),
        np.array([1.0]),
        np.array([2.0]),
    )
    assert lh[0] == pytest.approx(0.5 * 2.0 + 0.3 * 3.0 + 0.2 * 1.0)
    assert la[0] == pytest.approx(0.5 * 1.0 + 0.3 * 1.5 + 0.2 * 2.0)


def test_ensemble_params_backward_compat():
    params = EnsembleParams.from_dict({"w_gbm": 0.6, "w_nn": 0.4})
    assert params.w_bayesian == 0.0
    assert params.w_gbm + params.w_nn == pytest.approx(1.0)


def test_fit_three_way_weights_sum_to_one():
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
        y_h, y_a, lh_gbm, la_gbm, lh_nn, la_nn, lh_bay, la_bay
    )
    total = weights.w_gbm + weights.w_nn + weights.w_bayesian
    assert total == pytest.approx(1.0)
    assert weights.w_gbm >= 0
    assert weights.w_nn >= 0
    assert weights.w_bayesian >= 0

"""Tests for Bayesian Dixon-Coles model (no PyMC required for core math)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.bayesian.model import (
    dixon_coles_logp_numpy,
    match_log_prob_numpy,
    prepare_match_data,
)

PYMC_AVAILABLE = importlib.util.find_spec("pymc") is not None
PYMC_TEST = os.environ.get("PYMC_TEST", "") == "1"


def _synthetic_matches(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(8)]
    rows = []
    for i in range(n):
        home = teams[rng.integers(0, len(teams))]
        away = teams[rng.integers(0, len(teams))]
        while away == home:
            away = teams[rng.integers(0, len(teams))]
        rows.append(
            {
                "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": int(rng.integers(0, 4)),
                "away_score": int(rng.integers(0, 4)),
                "split": "val" if i % 2 else "train",
                **{col: float(rng.normal()) for col in FEATURE_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def test_match_log_prob_increases_with_likely_score():
    lp_low = match_log_prob_numpy(0, 0, 0.5, 0.5, rho=-0.05)
    lp_high = match_log_prob_numpy(5, 5, 0.5, 0.5, rho=-0.05)
    assert lp_low > lp_high


def test_dixon_coles_logp_vectorized_matches_scalar():
    y_h = np.array([0, 1, 2])
    y_a = np.array([0, 1, 0])
    lh = np.array([1.2, 1.2, 1.2])
    la = np.array([0.8, 0.8, 0.8])
    rho = -0.04
    vec = dixon_coles_logp_numpy(y_h, y_a, lh, la, rho)
    for i in range(3):
        assert vec[i] == pytest.approx(
            match_log_prob_numpy(int(y_h[i]), int(y_a[i]), lh[i], la[i], rho)
        )


def test_prepare_match_data_shapes():
    df = _synthetic_matches(50)
    data = prepare_match_data(df)
    assert data.n_matches == 50
    assert data.n_teams == 8
    assert data.x.shape == (50, len(FEATURE_COLUMNS))
    assert data.home_idx.shape == (50,)


@pytest.mark.skipif(not PYMC_AVAILABLE or not PYMC_TEST, reason="Set PYMC_TEST=1 with pymc installed")
def test_pymc_synthetic_fit(default_config):
    from worldcup_predictor.models.bayesian.trainer import fit_bayesian

    df = _synthetic_matches(120)
    artifacts = fit_bayesian(df, default_config, show_progress=False)
    assert default_config.calibration.rho_min <= artifacts.rho_mean <= default_config.calibration.rho_max
    assert artifacts.n_matches == 120

"""Tests for BayesianPredictor fast inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts
from worldcup_predictor.models.bayesian.predictor import BayesianPredictor


def _make_artifact(**kwargs) -> BayesianArtifacts:
    defaults = dict(
        rho_mean=-0.04,
        rho_std=0.01,
        intercept_mean=0.2,
        n_matches=100,
        n_teams=2,
        team_index={"Alpha": 0, "Beta": 1},
        feature_means={col: 0.0 for col in FEATURE_COLUMNS},
        feature_stds={col: 1.0 for col in FEATURE_COLUMNS},
        beta_mean={col: 0.0 for col in FEATURE_COLUMNS},
        att_mean={"Alpha": 0.3, "Beta": -0.3},
        def_mean={"Alpha": -0.1, "Beta": 0.1},
        chains=2,
        draws=50,
        tune=50,
    )
    defaults.update(kwargs)
    return BayesianArtifacts(**defaults)


def _make_row(home: str, away: str) -> dict:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update({"home_team": home, "away_team": away})
    return row


def test_predict_lambda_positive():
    artifact = _make_artifact()
    predictor = BayesianPredictor(artifact)
    df = pd.DataFrame([_make_row("Alpha", "Beta")])
    pred = predictor.predict_lambda(df)
    assert pred["lambda_home"].iloc[0] > 0
    assert pred["lambda_away"].iloc[0] > 0


def test_unknown_team_uses_zero_effects():
    artifact = _make_artifact()
    predictor = BayesianPredictor(artifact)
    df = pd.DataFrame([_make_row("Unknown", "Beta")])
    pred = predictor.predict_lambda(df)
    expected_home = np.exp(artifact.intercept_mean + 0.0 - artifact.def_mean["Beta"])
    assert pred["lambda_home"].iloc[0] == pytest.approx(expected_home)


def test_requires_team_columns():
    artifact = _make_artifact()
    predictor = BayesianPredictor(artifact)
    df = pd.DataFrame({col: [0.0] for col in FEATURE_COLUMNS})
    with pytest.raises(ValueError, match="home_team"):
        predictor.predict_lambda(df)

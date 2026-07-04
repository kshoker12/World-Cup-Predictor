"""Calibration integration with Bayesian rho artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.calibration.artifacts import fit_calibration
from worldcup_predictor.calibration.dixon_coles import DixonColesParams
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts
from worldcup_predictor.models.gbm import GBMPredictor


def _make_bayesian_artifact(rho: float = -0.04) -> BayesianArtifacts:
    return BayesianArtifacts(
        rho_mean=rho,
        rho_std=0.01,
        intercept_mean=0.1,
        n_matches=5000,
        n_teams=200,
        team_index={"A": 0, "B": 1},
        feature_means={col: 0.0 for col in FEATURE_COLUMNS},
        feature_stds={col: 1.0 for col in FEATURE_COLUMNS},
        beta_mean={col: 0.0 for col in FEATURE_COLUMNS},
        att_mean={"A": 0.2, "B": -0.2},
        def_mean={"A": -0.1, "B": 0.1},
        chains=2,
        draws=500,
        tune=500,
    )


def test_fit_calibration_uses_bayesian_rho(default_config):
    rng = np.random.default_rng(2)
    n = 400
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["home_team"] = ["A" if i % 2 else "B" for i in range(n)]
    data["away_team"] = ["B" if i % 2 else "A" for i in range(n)]
    data["home_score"] = rng.poisson(1.3, n)
    data["away_score"] = rng.poisson(1.1, n)
    data["split"] = "train"
    df = pd.DataFrame(data)
    train_df = df.iloc[:300].copy()
    val_df = df.iloc[300:].copy()
    val_df["split"] = "val"

    gbm = GBMPredictor(default_config.gbm)
    gbm.fit(train_df, val_df)

    bayesian = _make_bayesian_artifact(rho=-0.06)
    artifacts = fit_calibration(
        gbm,
        val_df,
        default_config.calibration,
        default_config,
        max_goals=10,
        bayesian=bayesian,
        show_progress=False,
    )
    assert artifacts.dixon_coles == DixonColesParams(rho=-0.06)
    total = (
        artifacts.ensemble.w_gbm
        + artifacts.ensemble.w_nn
        + artifacts.ensemble.w_bayesian
    )
    assert abs(total - 1.0) < 1e-6


def test_bayesian_artifacts_roundtrip(tmp_path: Path):
    artifact = _make_bayesian_artifact()
    path = tmp_path / "bayesian.json"
    artifact.save(path)
    loaded = BayesianArtifacts.load(path)
    assert loaded.rho_mean == artifact.rho_mean
    assert loaded.team_index == artifact.team_index
    assert loaded.att_mean == artifact.att_mean

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["rho_mean"] == -0.04

"""Fast Bayesian lambda inference from stored posterior means."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts


class BayesianPredictor:
    """Predict λ from posterior means of hierarchical Dixon-Coles model."""

    def __init__(self, artifacts: BayesianArtifacts) -> None:
        self.artifacts = artifacts
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self._beta = np.array(
            [artifacts.beta_mean.get(col, 0.0) for col in FEATURE_COLUMNS],
            dtype=float,
        )
        self._feat_means = np.array(
            [artifacts.feature_means.get(col, 0.0) for col in FEATURE_COLUMNS],
            dtype=float,
        )
        self._feat_stds = np.array(
            [
                max(artifacts.feature_stds.get(col, 1.0), 1e-8)
                for col in FEATURE_COLUMNS
            ],
            dtype=float,
        )

    def predict_lambda(self, features: pd.DataFrame) -> pd.DataFrame:
        if "home_team" not in features.columns or "away_team" not in features.columns:
            raise ValueError("features must include home_team and away_team columns")

        x_raw = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        x = (x_raw - self._feat_means) / self._feat_stds
        lin = x @ self._beta
        intercept = self.artifacts.intercept_mean

        homes = features["home_team"].astype(str)
        aways = features["away_team"].astype(str)
        att_h = homes.map(lambda t: self.artifacts.att_mean.get(t, 0.0)).to_numpy()
        def_h = homes.map(lambda t: self.artifacts.def_mean.get(t, 0.0)).to_numpy()
        att_a = aways.map(lambda t: self.artifacts.att_mean.get(t, 0.0)).to_numpy()
        def_a = aways.map(lambda t: self.artifacts.def_mean.get(t, 0.0)).to_numpy()

        eta_home = intercept + att_h - def_a + lin
        eta_away = intercept + att_a - def_h + lin
        return pd.DataFrame(
            {
                "lambda_home": np.exp(eta_home),
                "lambda_away": np.exp(eta_away),
            }
        )

    @classmethod
    def from_artifacts_path(cls, path: Path) -> BayesianPredictor:
        return cls(BayesianArtifacts.load(path))

"""Production predictor with calibration and Dixon-Coles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

from worldcup_predictor.calibration.artifacts import CalibrationArtifacts
from worldcup_predictor.calibration.ensemble import combine_lambda
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.models.nn.embeddings import EMBEDDING_PREFIX, compute_embedding_diff
from worldcup_predictor.simulation.score_grid import build_score_grid, wdl_probabilities

if TYPE_CHECKING:
    from worldcup_predictor.models.bayesian.predictor import BayesianPredictor
    from worldcup_predictor.models.nn.predictor import NNPredictor


class CalibratedPredictor:
    def __init__(
        self,
        gbm: GBMPredictor,
        artifacts: CalibrationArtifacts,
        *,
        nn: "NNPredictor | None" = None,
        bayesian: "BayesianPredictor | None" = None,
        max_goals: int = 10,
    ) -> None:
        self.gbm = gbm
        self.nn = nn
        self.bayesian = bayesian
        self.artifacts = artifacts
        self.max_goals = max_goals

    @property
    def rho(self) -> float:
        return self.artifacts.dixon_coles.rho

    def _gbm_features(
        self,
        features: pd.DataFrame,
        home_seq: np.ndarray | None,
        away_seq: np.ndarray | None,
    ) -> pd.DataFrame:
        needs_emb = any(
            col.startswith(EMBEDDING_PREFIX) for col in self.gbm.feature_columns
        )
        if not needs_emb:
            return features
        if self.nn is None or home_seq is None or away_seq is None:
            raise ValueError("GBM embedding model requires NN sequences at predict time")
        diff = compute_embedding_diff(self.nn, home_seq, away_seq, batch_size=len(features))
        emb_cols = [
            col for col in self.gbm.feature_columns if col.startswith(EMBEDDING_PREFIX)
        ]
        if not emb_cols:
            return features
        indices = [int(col.removeprefix(EMBEDDING_PREFIX)) for col in emb_cols]
        emb_df = pd.DataFrame(
            diff[:, indices],
            columns=emb_cols,
            index=features.index,
        )
        base_cols = [col for col in self.gbm.feature_columns if col in features.columns]
        return pd.concat([features[base_cols], emb_df], axis=1)

    def predict_lambda_raw(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.gbm.predict_lambda(features)

    def predict_lambda(
        self,
        features: pd.DataFrame,
        home_seq: np.ndarray | None = None,
        away_seq: np.ndarray | None = None,
    ) -> pd.DataFrame:
        gbm_features = self._gbm_features(features, home_seq, away_seq)
        raw = self.gbm.predict_lambda(gbm_features)
        lh_gbm, la_gbm = self.artifacts.scaling_gbm.apply(
            raw["lambda_home"].to_numpy(),
            raw["lambda_away"].to_numpy(),
        )

        lh_nn = la_nn = None
        if (
            self.nn is not None
            and self.artifacts.ensemble.w_nn > 0
            and home_seq is not None
            and away_seq is not None
        ):
            nn_raw = self.nn.predict_lambda(features, home_seq, away_seq)
            lh_nn, la_nn = self.artifacts.scaling_nn.apply(
                nn_raw["lambda_home"].to_numpy(),
                nn_raw["lambda_away"].to_numpy(),
            )

        lh_bayes = la_bayes = None
        if self.bayesian is not None and self.artifacts.ensemble.w_bayesian > 0:
            bayes_raw = self.bayesian.predict_lambda(features)
            lh_bayes, la_bayes = self.artifacts.scaling_bayesian.apply(
                bayes_raw["lambda_home"].to_numpy(),
                bayes_raw["lambda_away"].to_numpy(),
            )

        lh, la = combine_lambda(
            lh_gbm,
            la_gbm,
            self.artifacts.ensemble,
            lh_nn,
            la_nn,
            lh_bayes,
            la_bayes,
        )
        return pd.DataFrame({"lambda_home": lh, "lambda_away": la})

    def predict_wdl(
        self,
        features: pd.DataFrame,
        home_seq: np.ndarray | None = None,
        away_seq: np.ndarray | None = None,
    ) -> pd.DataFrame:
        pred = self.predict_lambda(features, home_seq, away_seq)
        p_home_list: list[float] = []
        p_draw_list: list[float] = []
        p_away_list: list[float] = []

        for lh, la in zip(pred["lambda_home"], pred["lambda_away"]):
            grid = build_score_grid(
                lh, la, max_goals=self.max_goals, rho=self.rho
            )
            p_h, p_d, p_a = wdl_probabilities(grid)
            p_home_list.append(p_h)
            p_draw_list.append(p_d)
            p_away_list.append(p_a)

        return pd.DataFrame(
            {
                "p_home_win": p_home_list,
                "p_draw": p_draw_list,
                "p_away_win": p_away_list,
            }
        )

    def save_artifacts(self, path: Path) -> None:
        self.artifacts.save(path)

    @classmethod
    def load(
        cls,
        gbm: GBMPredictor,
        artifacts_path: Path,
        *,
        nn: "NNPredictor | None" = None,
        bayesian: "BayesianPredictor | None" = None,
        max_goals: int = 10,
    ) -> CalibratedPredictor:
        artifacts = CalibrationArtifacts.load(artifacts_path)
        return cls(gbm, artifacts, nn=nn, bayesian=bayesian, max_goals=max_goals)


def load_calibrated_predictor(
    config,
    models_dir: Path,
) -> CalibratedPredictor:
    gbm = GBMPredictor(config.gbm)
    gbm.load(models_dir)
    artifacts_path = models_dir / "calibration.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Missing {artifacts_path}. Run fit_calibration.py first."
        )

    nn = None
    nn_weights = models_dir / "nn_model.pt"
    if nn_weights.exists():
        try:
            from worldcup_predictor.models.nn.predictor import NNPredictor

            nn = NNPredictor(config.nn)
            nn.load(models_dir)
        except ImportError:
            nn = None

    bayesian = None
    bayesian_path = models_dir / "bayesian.json"
    if bayesian_path.exists() and config.bayesian.use_in_ensemble:
        from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts
        from worldcup_predictor.models.bayesian.predictor import BayesianPredictor

        try:
            bayesian_artifacts = BayesianArtifacts.load(bayesian_path)
            if bayesian_artifacts.att_mean:
                bayesian = BayesianPredictor(bayesian_artifacts)
        except (KeyError, ValueError):
            bayesian = None

    return CalibratedPredictor.load(
        gbm,
        artifacts_path,
        nn=nn,
        bayesian=bayesian,
        max_goals=config.simulation.max_goals,
    )

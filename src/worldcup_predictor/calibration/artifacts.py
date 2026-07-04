"""Calibration artifact persistence and fitting orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.calibration.dixon_coles import DixonColesParams, fit_dixon_coles_rho
from worldcup_predictor.calibration.ensemble import EnsembleParams, fit_ensemble_weights
from worldcup_predictor.calibration.scaling import ScalingParams, fit_scaling
from typing import TYPE_CHECKING

from worldcup_predictor.config import AppConfig, CalibrationConfig
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.utils.progress import progress

if TYPE_CHECKING:
    from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts
    from worldcup_predictor.models.nn.predictor import NNPredictor


@dataclass(frozen=True)
class CalibrationArtifacts:
    scaling_gbm: ScalingParams
    scaling_nn: ScalingParams
    scaling_bayesian: ScalingParams
    ensemble: EnsembleParams
    dixon_coles: DixonColesParams
    min_ensemble_weight: float = 0.0

    @property
    def scaling(self) -> ScalingParams:
        """Backward-compatible alias for GBM scaling only."""
        return self.scaling_gbm

    def to_dict(self) -> dict:
        return {
            "scaling_gbm": asdict(self.scaling_gbm),
            "scaling_nn": asdict(self.scaling_nn),
            "scaling_bayesian": asdict(self.scaling_bayesian),
            "ensemble": asdict(self.ensemble),
            "dixon_coles": asdict(self.dixon_coles),
            "min_ensemble_weight": self.min_ensemble_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationArtifacts:
        scaling_gbm_raw = data.get("scaling_gbm", data.get("scaling", {}))
        scaling_nn_raw = data.get("scaling_nn", {"s_home": 1.0, "s_away": 1.0})
        scaling_bayesian_raw = data.get(
            "scaling_bayesian", {"s_home": 1.0, "s_away": 1.0}
        )
        return cls(
            scaling_gbm=ScalingParams(**scaling_gbm_raw),
            scaling_nn=ScalingParams(**scaling_nn_raw),
            scaling_bayesian=ScalingParams(**scaling_bayesian_raw),
            ensemble=EnsembleParams.from_dict(data["ensemble"]),
            dixon_coles=DixonColesParams(**data["dixon_coles"]),
            min_ensemble_weight=float(data.get("min_ensemble_weight", 0.0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifacts:
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def fit_calibration(
    gbm: GBMPredictor,
    val_df: pd.DataFrame,
    cal_config: CalibrationConfig,
    config: AppConfig,
    max_goals: int = 10,
    nn: "NNPredictor | None" = None,
    val_home_seq: np.ndarray | None = None,
    val_away_seq: np.ndarray | None = None,
    *,
    bayesian: "BayesianArtifacts | None" = None,
    show_progress: bool = True,
) -> CalibrationArtifacts:
    y_home = val_df["home_score"].to_numpy()
    y_away = val_df["away_score"].to_numpy()

    gbm_raw = gbm.predict_lambda(val_df)
    lh_gbm_raw = gbm_raw["lambda_home"].to_numpy()
    la_gbm_raw = gbm_raw["lambda_away"].to_numpy()

    use_bayesian = (
        bayesian is not None
        and config.bayesian.use_in_ensemble
        and bool(bayesian.att_mean)
    )

    step_names = ["GBM scaling"]
    if nn is not None and val_home_seq is not None and val_away_seq is not None:
        step_names.append("NN scaling")
    if use_bayesian:
        step_names.append("Bayesian scaling")
    if (nn is not None and val_home_seq is not None) or use_bayesian:
        step_names.append("ensemble")
    step_names.append("Dixon-Coles rho")

    scaling_gbm = ScalingParams(1.0, 1.0)
    scaling_nn = ScalingParams(1.0, 1.0)
    scaling_bayesian = ScalingParams(1.0, 1.0)
    lh_gbm: np.ndarray | None = None
    la_gbm: np.ndarray | None = None
    lh_nn: np.ndarray | None = None
    la_nn: np.ndarray | None = None
    lh_bayes: np.ndarray | None = None
    la_bayes: np.ndarray | None = None
    ensemble = EnsembleParams(w_gbm=1.0, w_nn=0.0, w_bayesian=0.0)
    dixon_coles = DixonColesParams(rho=0.0)

    bayesian_predictor = None
    if use_bayesian:
        from worldcup_predictor.models.bayesian.predictor import BayesianPredictor

        bayesian_predictor = BayesianPredictor(bayesian)

    for step in progress(step_names, desc="Fit calibration", disable=not show_progress):
        if step == "GBM scaling":
            scaling_gbm = fit_scaling(
                y_home, y_away, lh_gbm_raw, la_gbm_raw, bounds=cal_config.scaling_bounds
            )
            lh_gbm, la_gbm = scaling_gbm.apply(lh_gbm_raw, la_gbm_raw)
        elif step == "NN scaling":
            nn_raw = nn.predict_lambda(val_df, val_home_seq, val_away_seq)
            lh_nn_raw = nn_raw["lambda_home"].to_numpy()
            la_nn_raw = nn_raw["lambda_away"].to_numpy()
            scaling_nn = fit_scaling(
                y_home, y_away, lh_nn_raw, la_nn_raw, bounds=cal_config.scaling_bounds
            )
            lh_nn, la_nn = scaling_nn.apply(lh_nn_raw, la_nn_raw)
        elif step == "Bayesian scaling":
            assert bayesian_predictor is not None
            bayes_raw = bayesian_predictor.predict_lambda(val_df)
            lh_bayes_raw = bayes_raw["lambda_home"].to_numpy()
            la_bayes_raw = bayes_raw["lambda_away"].to_numpy()
            scaling_bayesian = fit_scaling(
                y_home,
                y_away,
                lh_bayes_raw,
                la_bayes_raw,
                bounds=cal_config.scaling_bounds,
            )
            lh_bayes, la_bayes = scaling_bayesian.apply(lh_bayes_raw, la_bayes_raw)
        elif step == "ensemble":
            assert lh_gbm is not None and la_gbm is not None
            ensemble = fit_ensemble_weights(
                y_home,
                y_away,
                lh_gbm,
                la_gbm,
                lh_nn,
                la_nn,
                lh_bayes,
                la_bayes,
                min_weight=cal_config.min_ensemble_weight,
            )
        elif step == "Dixon-Coles rho":
            if bayesian is not None:
                dixon_coles = DixonColesParams(rho=bayesian.rho_mean)
            else:
                assert lh_gbm is not None and la_gbm is not None
                lh_final, la_final = combine_from_artifacts(
                    lh_gbm,
                    la_gbm,
                    ensemble,
                    lh_nn,
                    la_nn,
                    lh_bayes,
                    la_bayes,
                )
                dixon_coles = fit_dixon_coles_rho(
                    y_home,
                    y_away,
                    lh_final,
                    la_final,
                    rho_min=cal_config.rho_min,
                    rho_max=cal_config.rho_max,
                    max_goals=max_goals,
                )

    return CalibrationArtifacts(
        scaling_gbm=scaling_gbm,
        scaling_nn=scaling_nn,
        scaling_bayesian=scaling_bayesian,
        ensemble=ensemble,
        dixon_coles=dixon_coles,
        min_ensemble_weight=cal_config.min_ensemble_weight,
    )


def combine_from_artifacts(
    lambda_home_gbm: np.ndarray,
    lambda_away_gbm: np.ndarray,
    ensemble: EnsembleParams,
    lambda_home_nn: np.ndarray | None = None,
    lambda_away_nn: np.ndarray | None = None,
    lambda_home_bayes: np.ndarray | None = None,
    lambda_away_bayes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from worldcup_predictor.calibration.ensemble import combine_lambda

    return combine_lambda(
        lambda_home_gbm,
        lambda_away_gbm,
        ensemble,
        lambda_home_nn,
        lambda_away_nn,
        lambda_home_bayes,
        lambda_away_bayes,
    )

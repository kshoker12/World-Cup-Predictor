#!/usr/bin/env python3
"""Validate calibration artifacts and improvement over raw GBM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.artifacts import CalibrationArtifacts  # noqa: E402
from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.metrics import evaluate_full, poisson_deviance  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
BAYESIAN_PATH = MODELS_DIR / "bayesian.json"


class ValidationError(Exception):
    pass


def main() -> int:
    if not CALIBRATION_PATH.exists():
        print(f"ERROR: Missing {CALIBRATION_PATH}. Run fit_calibration.py first.", file=sys.stderr)
        return 1

    config = load_config()
    df = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
    val_idx = np.where(df["split"].values == "val")[0]
    val_df = df.iloc[val_idx]
    test_df = df[df["split"] == "test"]

    gbm = GBMPredictor(config.gbm)
    gbm.load(MODELS_DIR)
    artifacts = CalibrationArtifacts.load(CALIBRATION_PATH)

    nn = None
    home_seq = away_seq = None
    if (MODELS_DIR / "nn_model.pt").exists():
        from worldcup_predictor.models.nn.predictor import NNPredictor

        nn = NNPredictor(config.nn)
        nn.load(MODELS_DIR)
        if INTL_SEQ_PATH.exists():
            seq = np.load(INTL_SEQ_PATH)
            home_seq, away_seq = seq["home_seq"], seq["away_seq"]

    predictor = load_calibrated_predictor(config, MODELS_DIR)

    try:
        if artifacts.scaling_gbm.s_home <= 0 or artifacts.scaling_gbm.s_away <= 0:
            raise ValidationError("GBM scaling factors must be positive")
        if artifacts.scaling_nn.s_home <= 0 or artifacts.scaling_nn.s_away <= 0:
            raise ValidationError("NN scaling factors must be positive")
        if (
            artifacts.scaling_bayesian.s_home <= 0
            or artifacts.scaling_bayesian.s_away <= 0
        ):
            raise ValidationError("Bayesian scaling factors must be positive")

        weight_sum = (
            artifacts.ensemble.w_gbm
            + artifacts.ensemble.w_nn
            + artifacts.ensemble.w_bayesian
        )
        if not (0.999 <= weight_sum <= 1.001):
            raise ValidationError(f"Ensemble weights sum to {weight_sum:.4f}, not 1")

        if not (config.calibration.rho_min <= artifacts.dixon_coles.rho <= config.calibration.rho_max):
            raise ValidationError(
                f"rho={artifacts.dixon_coles.rho} outside "
                f"[{config.calibration.rho_min}, {config.calibration.rho_max}]"
            )

        rho_source = (
            "bayesian posterior"
            if BAYESIAN_PATH.exists()
            else "standalone MLE"
        )
        print(f"Rho source: {rho_source}")
        print(
            f"Ensemble weights: w_gbm={artifacts.ensemble.w_gbm:.4f}, "
            f"w_nn={artifacts.ensemble.w_nn:.4f}, "
            f"w_bayesian={artifacts.ensemble.w_bayesian:.4f}"
        )

        raw_pred = gbm.predict_lambda(val_df)
        raw_dev = poisson_deviance(
            val_df["home_score"].to_numpy(), raw_pred["lambda_home"].to_numpy()
        ) + poisson_deviance(
            val_df["away_score"].to_numpy(), raw_pred["lambda_away"].to_numpy()
        )

        cal_pred = predictor.predict_lambda(
            val_df,
            home_seq=home_seq[val_idx] if home_seq is not None else None,
            away_seq=away_seq[val_idx] if away_seq is not None else None,
        )
        cal_dev = poisson_deviance(
            val_df["home_score"].to_numpy(), cal_pred["lambda_home"].to_numpy()
        ) + poisson_deviance(
            val_df["away_score"].to_numpy(), cal_pred["lambda_away"].to_numpy()
        )

        if cal_dev > raw_dev + 1e-9:
            raise ValidationError(
                f"Calibrated deviance {cal_dev:.4f} worse than raw {raw_dev:.4f}"
            )

        wdl = predictor.predict_wdl(
            val_df,
            home_seq=home_seq[val_idx] if home_seq is not None else None,
            away_seq=away_seq[val_idx] if away_seq is not None else None,
        )
        prob_sum = (
            wdl["p_home_win"] + wdl["p_draw"] + wdl["p_away_win"]
        )
        if not prob_sum.between(0.999, 1.001).all():
            raise ValidationError("W/D/L probabilities do not sum to 1")

        test_idx = np.where(df["split"].values == "test")[0]
        test_pred = predictor.predict_lambda(
            test_df,
            home_seq=home_seq[test_idx] if home_seq is not None else None,
            away_seq=away_seq[test_idx] if away_seq is not None else None,
        )
        test_metrics = evaluate_full(
            test_df["home_score"].to_numpy(),
            test_df["away_score"].to_numpy(),
            test_pred["lambda_home"].to_numpy(),
            test_pred["lambda_away"].to_numpy(),
            rho=predictor.rho,
            max_goals=config.simulation.max_goals,
        )
        print("Test metrics (report only):")
        for k, v in sorted(test_metrics.items()):
            print(f"  {k}: {v:.4f}")

    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("All calibration validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

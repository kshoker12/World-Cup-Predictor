#!/usr/bin/env python3
"""Compare raw GBM vs calibrated predictions on val and test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.metrics import evaluate_full  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def _row(label: str, split: str, metrics: dict[str, float]) -> str:
    return (
        f"{split:5} | {label:11} | "
        f"{metrics['poisson_deviance_total']:8.4f} | "
        f"{metrics['wdl_log_loss']:8.4f} | "
        f"{metrics['wdl_brier']:8.4f} | "
        f"{metrics['wdl_ece']:8.4f}"
    )


def main() -> int:
    config = load_config()
    df = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)

    gbm = GBMPredictor(config.gbm)
    gbm.load(MODELS_DIR)
    calibrated = load_calibrated_predictor(config, MODELS_DIR)

    home_seq = away_seq = None
    if INTL_SEQ_PATH.exists():
        seq = np.load(INTL_SEQ_PATH)
        home_seq, away_seq = seq["home_seq"], seq["away_seq"]

    nn = None
    if (MODELS_DIR / "nn_model.pt").exists():
        from worldcup_predictor.models.nn.predictor import NNPredictor

        nn = NNPredictor(config.nn)
        nn.load(MODELS_DIR)

    print("Split | Model       | PoissonDev | WDL_LL   | Brier    | ECE")
    print("-" * 62)

    for split in ("val", "test"):
        split_idx = np.where(df["split"].values == split)[0]
        split_df = df.iloc[split_idx]
        y_h = split_df["home_score"].to_numpy()
        y_a = split_df["away_score"].to_numpy()

        raw = gbm.predict_lambda(split_df)
        raw_metrics = evaluate_full(
            y_h,
            y_a,
            raw["lambda_home"].to_numpy(),
            raw["lambda_away"].to_numpy(),
            rho=0.0,
            max_goals=config.simulation.max_goals,
        )
        print(_row("raw GBM", split, raw_metrics))

        if nn is not None and home_seq is not None:
            nn_pred = nn.predict_lambda(
                split_df, home_seq[split_idx], away_seq[split_idx]
            )
            nn_metrics = evaluate_full(
                y_h,
                y_a,
                nn_pred["lambda_home"].to_numpy(),
                nn_pred["lambda_away"].to_numpy(),
                rho=0.0,
                max_goals=config.simulation.max_goals,
            )
            print(_row("raw NN", split, nn_metrics))

        bayesian_path = MODELS_DIR / "bayesian.json"
        if bayesian_path.exists():
            from worldcup_predictor.models.bayesian.predictor import BayesianPredictor

            try:
                bayesian = BayesianPredictor.from_artifacts_path(bayesian_path)
                if bayesian.artifacts.att_mean:
                    bayes_pred = bayesian.predict_lambda(split_df)
                    bayes_metrics = evaluate_full(
                        y_h,
                        y_a,
                        bayes_pred["lambda_home"].to_numpy(),
                        bayes_pred["lambda_away"].to_numpy(),
                        rho=0.0,
                        max_goals=config.simulation.max_goals,
                    )
                    print(_row("raw Bayes", split, bayes_metrics))
            except (KeyError, ValueError):
                pass

        cal = calibrated.predict_lambda(
            split_df,
            home_seq=home_seq[split_idx] if home_seq is not None else None,
            away_seq=away_seq[split_idx] if away_seq is not None else None,
        )
        cal_metrics = evaluate_full(
            y_h,
            y_a,
            cal["lambda_home"].to_numpy(),
            cal["lambda_away"].to_numpy(),
            rho=calibrated.rho,
            max_goals=config.simulation.max_goals,
        )
        print(_row("calibrated", split, cal_metrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

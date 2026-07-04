#!/usr/bin/env python3
"""Fit calibration artifacts on validation set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.calibration.artifacts import fit_calibration  # noqa: E402
from worldcup_predictor.calibration.predictor import load_calibrated_predictor  # noqa: E402
from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.metrics import evaluate_full  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
BAYESIAN_PATH = MODELS_DIR / "bayesian.json"


def _load_bayesian_if_available():
    if not BAYESIAN_PATH.exists():
        return None
    from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts

    return BayesianArtifacts.load(BAYESIAN_PATH)


def _load_nn_if_available(config):
    nn_path = MODELS_DIR / "nn_model.pt"
    if not nn_path.exists():
        return None, None, None
    from worldcup_predictor.models.nn.predictor import NNPredictor

    nn = NNPredictor(config.nn)
    nn.load(MODELS_DIR)
    seq = np.load(INTL_SEQ_PATH)
    return nn, seq["home_seq"], seq["away_seq"]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit calibration artifacts")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}", file=sys.stderr)
        return 1
    if not (MODELS_DIR / "gbm_home.txt").exists():
        print("ERROR: GBM models not found. Run train_gbm.py first.", file=sys.stderr)
        return 1

    config = load_config()
    df = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
    val_df = df[df["split"] == "val"]
    val_idx = np.where(df["split"].values == "val")[0]

    gbm = GBMPredictor(config.gbm)
    gbm.load(MODELS_DIR)
    nn, home_seq, away_seq = _load_nn_if_available(config)
    bayesian = _load_bayesian_if_available()

    raw_pred = gbm.predict_lambda(val_df)
    raw_metrics = evaluate_full(
        val_df["home_score"].to_numpy(),
        val_df["away_score"].to_numpy(),
        raw_pred["lambda_home"].to_numpy(),
        raw_pred["lambda_away"].to_numpy(),
        rho=0.0,
        max_goals=config.simulation.max_goals,
    )

    artifacts = fit_calibration(
        gbm,
        val_df,
        config.calibration,
        config,
        max_goals=config.simulation.max_goals,
        nn=nn,
        val_home_seq=home_seq[val_idx] if home_seq is not None else None,
        val_away_seq=away_seq[val_idx] if away_seq is not None else None,
        bayesian=bayesian,
        show_progress=not args.no_progress,
    )
    artifacts.save(CALIBRATION_PATH)

    calibrated = load_calibrated_predictor(config, MODELS_DIR)
    cal_pred = calibrated.predict_lambda(
        val_df,
        home_seq=home_seq[val_idx] if home_seq is not None else None,
        away_seq=away_seq[val_idx] if away_seq is not None else None,
    )
    cal_metrics = evaluate_full(
        val_df["home_score"].to_numpy(),
        val_df["away_score"].to_numpy(),
        cal_pred["lambda_home"].to_numpy(),
        cal_pred["lambda_away"].to_numpy(),
        rho=calibrated.rho,
        max_goals=config.simulation.max_goals,
    )

    print(f"Saved calibration to {CALIBRATION_PATH}")
    print(
        f"  scaling_gbm: s_home={artifacts.scaling_gbm.s_home:.4f}, "
        f"s_away={artifacts.scaling_gbm.s_away:.4f}"
    )
    print(
        f"  scaling_nn:  s_home={artifacts.scaling_nn.s_home:.4f}, "
        f"s_away={artifacts.scaling_nn.s_away:.4f}"
    )
    print(
        f"  scaling_bayesian: s_home={artifacts.scaling_bayesian.s_home:.4f}, "
        f"s_away={artifacts.scaling_bayesian.s_away:.4f}"
    )
    print(
        f"  ensemble: w_gbm={artifacts.ensemble.w_gbm:.4f}, "
        f"w_nn={artifacts.ensemble.w_nn:.4f}, "
        f"w_bayesian={artifacts.ensemble.w_bayesian:.4f}"
    )
    rho_source = "bayesian posterior" if bayesian is not None else "standalone MLE"
    print(f"  dixon_coles rho={artifacts.dixon_coles.rho:.4f} ({rho_source})")

    print("\nValidation comparison:")
    print(f"  raw poisson deviance:        {raw_metrics['poisson_deviance_total']:.4f}")
    print(f"  calibrated poisson deviance: {cal_metrics['poisson_deviance_total']:.4f}")
    print(f"  raw wdl log loss:            {raw_metrics['wdl_log_loss']:.4f}")
    print(f"  calibrated wdl log loss:     {cal_metrics['wdl_log_loss']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

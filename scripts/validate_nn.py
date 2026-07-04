#!/usr/bin/env python3
"""Validate NN experiment vs GBM baseline."""

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
from worldcup_predictor.models.metrics import poisson_deviance  # noqa: E402
from worldcup_predictor.models.nn.predictor import NNPredictor  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
INTL_SEQ_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def main() -> int:
    if not (MODELS_DIR / "nn_model.pt").exists():
        print("ERROR: Missing nn_model.pt. Run train_nn.py first.", file=sys.stderr)
        return 1

    config = load_config()
    features = pd.read_parquet(FEATURES_PATH).reset_index(drop=True)
    val_df = features[features["split"] == "val"].copy()
    val_idx = np.where(features["split"].values == "val")[0]
    seq = np.load(INTL_SEQ_PATH)

    gbm = GBMPredictor(config.gbm)
    gbm.load(MODELS_DIR)
    gbm_raw = gbm.predict_lambda(val_df)
    gbm_dev = poisson_deviance(
        val_df["home_score"].to_numpy(), gbm_raw["lambda_home"].to_numpy()
    ) + poisson_deviance(
        val_df["away_score"].to_numpy(), gbm_raw["lambda_away"].to_numpy()
    )

    nn = NNPredictor(config.nn)
    nn.load(MODELS_DIR)
    nn_raw = nn.predict_lambda(
        val_df, seq["home_seq"][val_idx], seq["away_seq"][val_idx]
    )
    nn_dev = poisson_deviance(
        val_df["home_score"].to_numpy(), nn_raw["lambda_home"].to_numpy()
    ) + poisson_deviance(
        val_df["away_score"].to_numpy(), nn_raw["lambda_away"].to_numpy()
    )

    bayesian = None
    bayesian_path = MODELS_DIR / "bayesian.json"
    if bayesian_path.exists():
        from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts

        bayesian = BayesianArtifacts.load(bayesian_path)

    artifacts = fit_calibration(
        gbm,
        val_df,
        config.calibration,
        config,
        max_goals=config.simulation.max_goals,
        nn=nn,
        val_home_seq=seq["home_seq"][val_idx],
        val_away_seq=seq["away_seq"][val_idx],
        bayesian=bayesian,
    )
    calibrated = load_calibrated_predictor(config, MODELS_DIR)
    cal_pred = calibrated.predict_lambda(
        val_df,
        home_seq=seq["home_seq"][val_idx],
        away_seq=seq["away_seq"][val_idx],
    )
    ens_dev = poisson_deviance(
        val_df["home_score"].to_numpy(), cal_pred["lambda_home"].to_numpy()
    ) + poisson_deviance(
        val_df["away_score"].to_numpy(), cal_pred["lambda_away"].to_numpy()
    )

    print("Validation Poisson deviance (lower is better):")
    print(f"  GBM raw:           {gbm_dev:.4f}")
    print(f"  NN raw:            {nn_dev:.4f}")
    print(f"  Calibrated ensemble: {ens_dev:.4f}")
    print(
        f"  Ensemble weights: w_gbm={artifacts.ensemble.w_gbm:.3f}, "
        f"w_nn={artifacts.ensemble.w_nn:.3f}, "
        f"w_bayesian={artifacts.ensemble.w_bayesian:.3f}"
    )

    if ens_dev <= gbm_dev:
        print("RESULT: PASS — ensemble improves or matches GBM on validation.")
        return 0
    print("RESULT: NULL — ensemble did not beat GBM-only on validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

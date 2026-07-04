#!/usr/bin/env python3
"""Validate trained GBM models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS  # noqa: E402
from worldcup_predictor.models.gbm import GBMPredictor  # noqa: E402
from worldcup_predictor.models.metrics import poisson_deviance  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


class ValidationError(Exception):
    pass


def main() -> int:
    home_path = MODELS_DIR / "gbm_home.txt"
    away_path = MODELS_DIR / "gbm_away.txt"
    if not home_path.exists() or not away_path.exists():
        print("ERROR: Models not found. Run train_gbm.py first.", file=sys.stderr)
        return 1
    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}", file=sys.stderr)
        return 1

    config = load_config()
    df = pd.read_parquet(FEATURES_PATH)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    predictor = GBMPredictor(config.gbm)
    predictor.load(MODELS_DIR)

    try:
        if predictor.feature_columns != list(FEATURE_COLUMNS):
            raise ValidationError("Feature columns mismatch")

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            pred = predictor.predict_lambda(split_df)
            if (pred["lambda_home"] <= 0).any() or (pred["lambda_away"] <= 0).any():
                raise ValidationError(f"Non-positive lambda on {split_name}")

        mean_home = train_df["home_score"].mean()
        mean_away = train_df["away_score"].mean()
        naive_dev = poisson_deviance(
            val_df["home_score"].to_numpy(), np.full(len(val_df), mean_home)
        ) + poisson_deviance(
            val_df["away_score"].to_numpy(), np.full(len(val_df), mean_away)
        )
        model_metrics = predictor.evaluate(val_df)
        if model_metrics["poisson_deviance_total"] >= naive_dev:
            raise ValidationError(
                f"Model deviance {model_metrics['poisson_deviance_total']:.4f} "
                f"not better than naive {naive_dev:.4f}"
            )

        test_metrics = predictor.evaluate(test_df)
        print("Validation metrics:")
        for k, v in sorted(model_metrics.items()):
            print(f"  {k}: {v:.4f}")
        print("Test metrics (report only):")
        for k, v in sorted(test_metrics.items()):
            print(f"  {k}: {v:.4f}")
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("All model validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

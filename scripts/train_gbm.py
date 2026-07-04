#!/usr/bin/env python3
"""Train LightGBM Poisson models on feature parquet."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.gbm import train_from_features  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM Poisson models")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}", file=sys.stderr)
        return 1

    config = load_config()
    predictor, metrics, test_metrics = train_from_features(
        FEATURES_PATH, config, show_progress=not args.no_progress
    )
    predictor.save(MODELS_DIR)

    print(f"Models saved to {MODELS_DIR}")
    print("Validation metrics:")
    for k, v in sorted(metrics.val.items()):
        print(f"  {k}: {v:.4f}")
    print("Test metrics (report only):")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

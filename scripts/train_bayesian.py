#!/usr/bin/env python3
"""Train Bayesian hierarchical Dixon-Coles model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
BAYESIAN_PATH = MODELS_DIR / "bayesian.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train Bayesian Dixon-Coles model")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}", file=sys.stderr)
        return 1

    try:
        from worldcup_predictor.models.bayesian.trainer import train_from_features
    except ImportError as exc:
        print(
            "ERROR: PyMC not installed. Run: pip install -e \".[dev,bayesian]\"",
            file=sys.stderr,
        )
        return 1

    config = load_config()
    artifacts = train_from_features(
        FEATURES_PATH,
        config,
        show_progress=not args.no_progress,
    )
    artifacts.save(BAYESIAN_PATH)

    print(f"Saved Bayesian artifacts to {BAYESIAN_PATH}")
    print(f"  rho_mean={artifacts.rho_mean:.4f} (sd={artifacts.rho_std:.4f})")
    print(f"  n_matches={artifacts.n_matches}, n_teams={artifacts.n_teams}")
    if artifacts.rhat_rho is not None:
        print(f"  r_hat(rho)={artifacts.rhat_rho:.4f}")
    if artifacts.ess_rho is not None:
        print(f"  ess_bulk(rho)={artifacts.ess_rho:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

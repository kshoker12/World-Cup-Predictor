#!/usr/bin/env python3
"""Build international NN sequences aligned to features.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.sequences import build_international_sequences  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "intl_sequences.npz"


def main() -> int:
    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}", file=sys.stderr)
        return 1

    import pandas as pd

    config = load_config()
    features = pd.read_parquet(FEATURES_PATH)
    match_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]
    missing = [c for c in match_cols if c not in features.columns]
    if missing:
        print(f"ERROR: features.parquet missing columns: {missing}", file=sys.stderr)
        return 1

    matches = features[match_cols].copy()
    home_seq, away_seq = build_international_sequences(matches, config)
    np.savez(OUTPUT_PATH, home_seq=home_seq, away_seq=away_seq)
    print(f"Wrote international sequences to {OUTPUT_PATH} ({len(home_seq):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

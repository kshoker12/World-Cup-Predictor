#!/usr/bin/env python3
"""Build club pretrain sequences."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.models.sequences import (  # noqa: E402
    build_club_sequences,
    compute_club_norm_stats,
)
from worldcup_predictor.utils.progress import progress  # noqa: E402

CLUB_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "club_matches.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "club_sequences.npz"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "club_norm_stats.json"


def main() -> int:
    if not CLUB_MATCHES_PATH.exists():
        print(f"ERROR: Missing {CLUB_MATCHES_PATH}. Run build_club_matches.py first.", file=sys.stderr)
        return 1

    import pandas as pd

    config = load_config()
    matches = pd.read_parquet(CLUB_MATCHES_PATH)
    norm_stats = compute_club_norm_stats(matches)
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NORM_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    _ = list(progress(range(1), desc="Build club sequences", total=1))
    home_seq, away_seq, y_home, y_away = build_club_sequences(
        matches,
        seq_len=config.nn.seq_len,
        elo_config=config.elo,
        norm_stats=norm_stats,
    )
    np.savez(
        OUTPUT_PATH,
        home_seq=home_seq,
        away_seq=away_seq,
        y_home=y_home,
        y_away=y_away,
    )
    print(f"Wrote club sequences to {OUTPUT_PATH} ({len(y_home):,} matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

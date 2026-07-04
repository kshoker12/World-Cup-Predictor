#!/usr/bin/env python3
"""Merge Understat and soccer-dataset club matches."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.data.club_loader import (  # noqa: E402
    load_soccer_dataset,
    load_understat_matches,
)
from worldcup_predictor.data.club_merge import merge_club_sources  # noqa: E402

UNDERSTAT_PATH = PROJECT_ROOT / "data" / "processed" / "understat_matches.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "club_matches.parquet"


def main() -> int:
    config = load_config()
    club_dir = PROJECT_ROOT / config.club.data_dir

    if not UNDERSTAT_PATH.exists():
        print(f"ERROR: Missing {UNDERSTAT_PATH}", file=sys.stderr)
        return 1
    if not (club_dir / "fixtures.csv").exists() or not (club_dir / "match_stats.csv").exists():
        print(f"ERROR: Missing soccer-dataset CSVs in {club_dir}", file=sys.stderr)
        return 1

    understat = load_understat_matches(UNDERSTAT_PATH)
    soccer = load_soccer_dataset(club_dir)
    merged = merge_club_sources(
        understat,
        soccer,
        min_date=config.club.min_date,
        forward_fill_within_league=config.club.forward_fill_within_league,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote {len(merged):,} merged club matches to {OUTPUT_PATH}")
    print(merged["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

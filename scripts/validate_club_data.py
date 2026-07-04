#!/usr/bin/env python3
"""Validate both club data sources are present."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402

UNDERSTAT_PATH = PROJECT_ROOT / "data" / "processed" / "understat_matches.parquet"
CLUB_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "club_matches.parquet"
MIN_ROWS = 10_000


class ValidationError(Exception):
    pass


def main() -> int:
    config = load_config()
    club_dir = PROJECT_ROOT / config.club.data_dir
    fixtures = club_dir / "fixtures.csv"
    stats = club_dir / "match_stats.csv"

    errors: list[str] = []
    if not UNDERSTAT_PATH.exists():
        errors.append(
            f"Missing {UNDERSTAT_PATH}. Run fetch_understat_matches.py first."
        )
    if not fixtures.exists() or not stats.exists():
        errors.append(
            f"Missing soccer-dataset files in {club_dir}: "
            "need fixtures.csv and match_stats.csv"
        )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if CLUB_MATCHES_PATH.exists():
        import pandas as pd

        merged = pd.read_parquet(CLUB_MATCHES_PATH)
        if len(merged) < MIN_ROWS:
            print(
                f"ERROR: Combined club matches {len(merged)} < {MIN_ROWS}",
                file=sys.stderr,
            )
            return 1
        print(f"Combined club matches: {len(merged):,}")
        print("Source counts:")
        print(merged["source"].value_counts().to_string())

    print("Club data validation passed (both sources present).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

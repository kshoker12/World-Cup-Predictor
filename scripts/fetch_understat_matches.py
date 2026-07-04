#!/usr/bin/env python3
"""Fetch Understat team match stats."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.data.understat_fetch import (  # noqa: E402
    bootstrap_understat_from_soccer,
    fetch_understat_matches,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "understat_matches.parquet"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Understat match-level data")
    parser.add_argument(
        "--fallback-from-club",
        action="store_true",
        help="Build Understat proxy from soccer-dataset Big 5 when network fetch fails",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config()
    show_progress = not args.no_progress

    if args.fallback_from_club:
        df = bootstrap_understat_from_soccer(
            config.club, OUTPUT_PATH, show_progress=show_progress
        )
    else:
        try:
            df = fetch_understat_matches(
                config.club, OUTPUT_PATH, show_progress=show_progress
            )
        except OSError as exc:
            print(f"WARN: Understat fetch failed ({exc}); using soccer-dataset proxy", file=sys.stderr)
            df = bootstrap_understat_from_soccer(
                config.club, OUTPUT_PATH, show_progress=show_progress
            )

    print(f"Wrote {len(df):,} Understat matches to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build feature parquet from raw international results."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.data.loader import load_and_clean_matches  # noqa: E402
from worldcup_predictor.features.pipeline import MatchPipeline  # noqa: E402
from worldcup_predictor.utils.progress import progress  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"
FORMER_NAMES_PATH = PROJECT_ROOT / "data" / "raw" / "former_names.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build feature parquet")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        print(f"ERROR: Missing {RAW_PATH}", file=sys.stderr)
        return 1

    config = load_config()
    matches = load_and_clean_matches(
        RAW_PATH, config, former_names_path=FORMER_NAMES_PATH
    )
    features = MatchPipeline(config).run(matches, show_progress=not args.no_progress)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote {len(features):,} rows to {OUTPUT_PATH}")
    print("Split distribution:")
    for split, count in features["split"].value_counts().sort_index().items():
        print(f"  {split}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a flat soccer-data folder mirroring the Kaggle dataset layout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "flat_soccer_data"

FILES = {
    "results.csv": PROJECT_ROOT / "data" / "raw" / "results.csv",
    "wc2026_results.csv": PROJECT_ROOT / "data" / "raw" / "wc2026_results.csv",
    "former_names.csv": PROJECT_ROOT / "data" / "raw" / "former_names.csv",
    "fixtures.csv": PROJECT_ROOT / "data" / "raw" / "club" / "fixtures.csv",
    "match_stats.csv": PROJECT_ROOT / "data" / "raw" / "club" / "match_stats.csv",
    "understat_matches.parquet": PROJECT_ROOT / "data" / "processed" / "understat_matches.parquet",
}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, src in FILES.items():
        if not src.exists():
            print(f"ERROR: missing {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, OUTPUT / name)
        print(f"Copied {name}")
    print(f"Flat dataset ready at {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

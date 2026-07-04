"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.config import AppConfig, merge_former_name_aliases
from worldcup_predictor.data.cleaning import clean_matches

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]


def load_raw_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def load_and_clean_matches(
    results_path: Path,
    config: AppConfig,
    former_names_path: Path | None = None,
    *,
    extra_results_paths: list[Path] | None = None,
) -> pd.DataFrame:
    if former_names_path is not None:
        config = merge_former_name_aliases(config, former_names_path)
    raw = load_raw_results(results_path)
    if extra_results_paths:
        extras = [load_raw_results(path) for path in extra_results_paths if path.exists()]
        if extras:
            raw = pd.concat([raw, *extras], ignore_index=True)
    return clean_matches(raw, config.team_aliases)

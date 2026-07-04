#!/usr/bin/env python3
"""Post-build validation for features.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_predictor.config import load_config  # noqa: E402
from worldcup_predictor.data.loader import load_and_clean_matches  # noqa: E402
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS, MatchPipeline  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"
FORMER_NAMES_PATH = PROJECT_ROOT / "data" / "raw" / "former_names.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

REQUIRED_COLS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "neutral",
    *FEATURE_COLUMNS,
    "home_score",
    "away_score",
    "split",
]


class ValidationError(Exception):
    pass


def check_schema(features: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLS) - set(features.columns)
    if missing:
        raise ValidationError(f"Missing columns: {sorted(missing)}")
    null_counts = features[REQUIRED_COLS].isnull().sum()
    bad = null_counts[null_counts > 0]
    if len(bad) > 0:
        raise ValidationError(f"Null values found: {bad.to_dict()}")


def check_row_count(features: pd.DataFrame, expected: int) -> None:
    if len(features) != expected:
        raise ValidationError(
            f"Row count mismatch: features={len(features)}, cleaned input={expected}"
        )


def check_splits(features: pd.DataFrame, config) -> None:
    dates = pd.to_datetime(features["date"])
    train_mask = features["split"] == "train"
    val_mask = features["split"] == "val"
    test_mask = features["split"] == "test"

    if train_mask.any() and dates[train_mask].max().date() >= config.splits.train_end:
        raise ValidationError("Train split contains dates >= train_end")
    if val_mask.any():
        val_dates = dates[val_mask]
        if val_dates.min().date() < config.splits.train_end:
            raise ValidationError("Val split contains dates < train_end")
        if val_dates.max().date() >= config.splits.val_end:
            raise ValidationError("Val split contains dates >= val_end")
    if test_mask.any() and dates[test_mask].min().date() < config.splits.val_end:
        raise ValidationError("Test split contains dates < val_end")

    print("Split distribution:")
    for split, count in features["split"].value_counts().sort_index().items():
        print(f"  {split}: {count:,}")


def check_feature_ranges(features: pd.DataFrame) -> None:
    for col in ("is_home", "is_neutral"):
        unique = set(features[col].unique())
        if not unique.issubset({0, 1}):
            raise ValidationError(f"{col} has invalid values: {unique}")

    ti = set(features["tournament_importance"].unique())
    if not ti.issubset({1, 2, 3}):
        raise ValidationError(f"tournament_importance has invalid values: {ti}")


def check_neutral_consistency(features: pd.DataFrame) -> None:
    bad = features[(features["is_neutral"] == 1) & (features["is_home"] != 0)]
    if len(bad) > 0:
        raise ValidationError(
            f"Neutral rows with is_home != 0: {len(bad)}"
        )


def check_chronological(features: pd.DataFrame) -> None:
    dates = pd.to_datetime(features["date"])
    if not dates.is_monotonic_increasing:
        raise ValidationError("Output dates are not non-decreasing")


def check_no_duplicates(features: pd.DataFrame) -> None:
    dupes = features.duplicated(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep=False,
    )
    if dupes.any():
        raise ValidationError(f"Duplicate match rows: {dupes.sum()}")


def check_sample_replay(features: pd.DataFrame, config) -> None:
    n = min(1000, len(features))
    if n == 0:
        return

    subset_matches = features.iloc[:n][
        ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    ].copy()
    # Rebuild from raw-like input for replay
    replay_input = subset_matches.rename(columns={})
    replay_input["neutral"] = replay_input["neutral"].astype(bool)

    replayed = MatchPipeline(config).run(replay_input)
    last_full = features.iloc[n - 1][FEATURE_COLUMNS]
    last_replay = replayed.iloc[n - 1][FEATURE_COLUMNS]

    for col in FEATURE_COLUMNS:
        if abs(float(last_full[col]) - float(last_replay[col])) > 1e-9:
            raise ValidationError(
                f"Replay mismatch on row {n - 1}, column {col}: "
                f"{last_full[col]} vs {last_replay[col]}"
            )


def main() -> int:
    if not FEATURES_PATH.exists():
        print(f"ERROR: Missing {FEATURES_PATH}. Run build_features.py first.", file=sys.stderr)
        return 1

    config = load_config()
    features = pd.read_parquet(FEATURES_PATH)

    if RAW_PATH.exists():
        matches = load_and_clean_matches(
            RAW_PATH, config, former_names_path=FORMER_NAMES_PATH
        )
        expected_rows = len(matches)
    else:
        expected_rows = len(features)

    try:
        check_schema(features)
        check_row_count(features, expected_rows)
        check_splits(features, config)
        check_feature_ranges(features)
        check_neutral_consistency(features)
        check_chronological(features)
        check_no_duplicates(features)
        check_sample_replay(features, config)
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

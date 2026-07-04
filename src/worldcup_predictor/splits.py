"""Train/validation/test split labeling."""

from __future__ import annotations

from datetime import date

from worldcup_predictor.config import SplitConfig


def assign_split(match_date: date, splits: SplitConfig) -> str:
    if match_date < splits.train_end:
        return "train"
    if match_date < splits.val_end:
        return "val"
    return "test"

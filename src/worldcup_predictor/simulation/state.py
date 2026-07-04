"""Pipeline state management for tournament simulation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from worldcup_predictor.config import AppConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS, MatchPipeline


def build_initial_pipeline(
    matches: pd.DataFrame,
    config: AppConfig,
    before_date: date,
) -> MatchPipeline:
    """Process all matches strictly before kickoff to seed simulation state."""
    historical = matches[matches["date"] < before_date].copy()
    pipeline = MatchPipeline(config)
    if len(historical) > 0:
        pipeline.run(historical)
    return pipeline


def clone_pipeline(pipeline: MatchPipeline) -> MatchPipeline:
    return pipeline.clone()


def features_for_fixture(
    pipeline: MatchPipeline,
    home: str,
    away: str,
    *,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> pd.DataFrame:
    row = pipeline.build_feature_row(
        home, away, tournament=tournament, neutral=neutral
    )
    return pd.DataFrame([row])


def apply_result(
    pipeline: MatchPipeline,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    neutral: bool = True,
) -> None:
    pipeline.apply_match(home, away, home_goals, away_goals, neutral=neutral)


def sequences_for_fixture(
    pipeline: MatchPipeline,
    home: str,
    away: str,
) -> tuple[np.ndarray, np.ndarray]:
    home_seq, away_seq = pipeline.sequence_snapshot(home, away)
    return home_seq, away_seq


def feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)

"""Merge Understat and soccer-dataset club matches."""

from __future__ import annotations

import pandas as pd

from worldcup_predictor.data.club_cleaning import (
    CLUB_MATCH_COLUMNS,
    clean_club_matches,
    normalize_team_name,
)


def _dedupe_key(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"], utc=True).dt.strftime("%Y-%m-%d")
    return (
        dates
        + "|"
        + df["home_team"].map(normalize_team_name)
        + "|"
        + df["away_team"].map(normalize_team_name)
    )


def merge_club_sources(
    understat: pd.DataFrame,
    soccer: pd.DataFrame,
    *,
    min_date: pd.Timestamp | None = None,
    forward_fill_within_league: bool = True,
) -> pd.DataFrame:
    u = clean_club_matches(
        understat,
        min_date=min_date,
        forward_fill_within_league=forward_fill_within_league,
    )
    s = clean_club_matches(
        soccer,
        min_date=min_date,
        forward_fill_within_league=forward_fill_within_league,
    )

    u["_key"] = _dedupe_key(u)
    s["_key"] = _dedupe_key(s)

    understat_keys = set(u["_key"])
    s_unique = s[~s["_key"].isin(understat_keys)].copy()
    merged = pd.concat([u, s_unique], ignore_index=True)
    merged = merged.drop(columns=["_key"])
    merged = merged.sort_values(["date", "home_team", "away_team"]).reset_index(
        drop=True
    )
    return merged[CLUB_MATCH_COLUMNS]

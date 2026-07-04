"""Club match data cleaning utilities."""

from __future__ import annotations

import pandas as pd

CLUB_MATCH_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_ppda",
    "away_ppda",
    "league",
    "source",
]


def normalize_team_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def clean_club_matches(
    df: pd.DataFrame,
    *,
    min_date: pd.Timestamp | None = None,
    forward_fill_within_league: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    out["home_team"] = out["home_team"].map(normalize_team_name)
    out["away_team"] = out["away_team"].map(normalize_team_name)
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")

    for col in [
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "home_ppda",
        "away_ppda",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    if min_date is not None:
        min_ts = pd.Timestamp(min_date, tz="UTC")
        out = out[out["date"] >= min_ts]

    if forward_fill_within_league:
        out = out.sort_values(["league", "date", "home_team", "away_team"])
        fill_cols = [
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_ppda",
            "away_ppda",
        ]
        for col in fill_cols:
            out[col] = out.groupby("league", sort=False)[col].ffill()

    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out[CLUB_MATCH_COLUMNS]

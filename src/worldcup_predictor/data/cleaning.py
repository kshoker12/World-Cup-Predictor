"""Match data cleaning per technical specification."""

from __future__ import annotations

import pandas as pd


def _normalize_neutral(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.upper()
    return normalized.isin({"TRUE", "T", "1", "YES"})


def _apply_team_aliases(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    if not aliases:
        return df
    out = df.copy()
    out["home_team"] = out["home_team"].replace(aliases)
    out["away_team"] = out["away_team"].replace(aliases)
    return out


def clean_matches(df: pd.DataFrame, team_aliases: dict[str, str]) -> pd.DataFrame:
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])

    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["tournament"] = out["tournament"].astype(str).str.strip()

    out = out[out["home_team"] != out["away_team"]]

    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out = out.dropna(subset=["home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out = out[(out["home_score"] >= 0) & (out["away_score"] >= 0)]

    out["neutral"] = _normalize_neutral(out["neutral"])

    out = _apply_team_aliases(out, team_aliases)

    out = out.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep="first",
    )

    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out

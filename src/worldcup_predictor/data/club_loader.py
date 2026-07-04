"""Load club match data from Understat parquet and soccer-dataset CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.data.club_cleaning import CLUB_MATCH_COLUMNS, normalize_team_name


def load_understat_matches(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "understat"
    for col in CLUB_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CLUB_MATCH_COLUMNS]


def _load_team_names(data_dir: Path) -> pd.DataFrame:
    teams_path = data_dir / "teams.csv"
    if not teams_path.exists():
        return pd.DataFrame(columns=["id", "name"])
    teams = pd.read_csv(teams_path)
    name_col = "name" if "name" in teams.columns else "team_name"
    return teams[["id", name_col]].rename(columns={name_col: "name"})


def _load_league_names(data_dir: Path) -> pd.DataFrame:
    leagues_path = data_dir / "leagues.csv"
    if not leagues_path.exists():
        return pd.DataFrame(columns=["id", "name"])
    leagues = pd.read_csv(leagues_path)
    name_col = "name" if "name" in leagues.columns else "league_name"
    return leagues[["id", name_col]].rename(columns={name_col: "name"})


def load_soccer_dataset(data_dir: Path) -> pd.DataFrame:
    fixtures_path = data_dir / "fixtures.csv"
    stats_path = data_dir / "match_stats.csv"
    if not fixtures_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Missing soccer-dataset files in {data_dir}: "
            "need fixtures.csv and match_stats.csv"
        )

    fixtures = pd.read_csv(fixtures_path, low_memory=False)
    stats = pd.read_csv(stats_path, low_memory=False)
    teams = _load_team_names(data_dir)
    leagues = _load_league_names(data_dir)

    join_key = "fixture_id" if "fixture_id" in fixtures.columns else "id"
    stats_key = "fixture_id" if "fixture_id" in stats.columns else "id"
    merged = fixtures.merge(stats, left_on=join_key, right_on=stats_key, how="inner")

    if "home_team" not in merged.columns and "home_team_id" in merged.columns:
        merged = merged.merge(
            teams.rename(columns={"id": "home_team_id", "name": "home_team"}),
            on="home_team_id",
            how="left",
        )
    if "away_team" not in merged.columns and "away_team_id" in merged.columns:
        merged = merged.merge(
            teams.rename(columns={"id": "away_team_id", "name": "away_team"}),
            on="away_team_id",
            how="left",
        )

    league_col = None
    for candidate in ("league_name", "league"):
        if candidate in merged.columns:
            league_col = candidate
            break
    if league_col is None and "league_id" in merged.columns and len(leagues):
        merged = merged.merge(
            leagues.rename(columns={"id": "league_id", "name": "league_name"}),
            on="league_id",
            how="left",
        )
        league_col = "league_name"

    home_goals = merged["home_goals"] if "home_goals" in merged.columns else merged["goals_home"]
    away_goals = merged["away_goals"] if "away_goals" in merged.columns else merged["goals_away"]
    home_col = "home_team_name" if "home_team_name" in merged.columns else "home_team"
    away_col = "away_team_name" if "away_team_name" in merged.columns else "away_team"

    home_shots = merged.get("home_shots")
    if home_shots is None and "home_shots_total" in merged.columns:
        home_shots = merged["home_shots_total"]
    away_shots = merged.get("away_shots")
    if away_shots is None and "away_shots_total" in merged.columns:
        away_shots = merged["away_shots_total"]

    out = pd.DataFrame(
        {
            "date": merged["date"],
            "home_team": merged[home_col],
            "away_team": merged[away_col],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": merged.get("home_xg"),
            "away_xg": merged.get("away_xg"),
            "home_shots": home_shots,
            "away_shots": away_shots,
            "home_ppda": merged.get("home_ppda"),
            "away_ppda": merged.get("away_ppda"),
            "league": merged[league_col] if league_col else "unknown",
            "source": "soccer_dataset",
        }
    )
    return out

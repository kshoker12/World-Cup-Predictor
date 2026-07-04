"""Fetch Understat team match stats via soccerdata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.config import ClubConfig
from worldcup_predictor.data.club_cleaning import CLUB_MATCH_COLUMNS, normalize_team_name
from worldcup_predictor.utils.progress import progress


def fetch_understat_matches(
    config: ClubConfig,
    output_path: Path,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    import soccerdata as sd

    frames: list[pd.DataFrame] = []
    combos = [
        (league, season)
        for league in config.understat_leagues
        for season in config.understat_seasons
    ]

    iterator = progress(
        combos,
        desc="Fetch Understat",
        total=len(combos),
        disable=not show_progress,
    )
    for league, season in iterator:
        reader = sd.Understat(leagues=league, seasons=season)
        stats = reader.read_team_match_stats()
        if stats is None or len(stats) == 0:
            continue
        stats = stats.reset_index()
        frames.append(_normalize_understat_frame(stats, league))

    if not frames:
        raise RuntimeError("No Understat match data fetched")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["date"] >= pd.Timestamp(config.min_date, tz="UTC")]
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def _normalize_understat_frame(stats: pd.DataFrame, league: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in stats.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in cols:
                return cols[name]
            if name in stats.columns:
                return name
        return None

    date_col = pick("date")
    home_col = pick("home_team", "h")
    away_col = pick("away_team", "a")
    hg_col = pick("home_goals", "goals_home", "home_score")
    ag_col = pick("away_goals", "goals_away", "away_score")
    hxg_col = pick("home_xg", "xg_home")
    axg_col = pick("away_xg", "xg_away")
    hshots_col = pick("home_shots", "shots_home")
    ashots_col = pick("away_shots", "shots_away")
    hppda_col = pick("home_ppda", "ppda_home")
    appda_col = pick("away_ppda", "ppda_away")

    if date_col is None or home_col is None or away_col is None:
        raise ValueError(f"Unexpected Understat columns: {list(stats.columns)}")

    out = pd.DataFrame(
        {
            "date": stats[date_col],
            "home_team": stats[home_col].map(normalize_team_name),
            "away_team": stats[away_col].map(normalize_team_name),
            "home_goals": stats[hg_col] if hg_col else 0,
            "away_goals": stats[ag_col] if ag_col else 0,
            "home_xg": stats[hxg_col] if hxg_col else pd.NA,
            "away_xg": stats[axg_col] if axg_col else pd.NA,
            "home_shots": stats[hshots_col] if hshots_col else pd.NA,
            "away_shots": stats[ashots_col] if ashots_col else pd.NA,
            "home_ppda": stats[hppda_col] if hppda_col else pd.NA,
            "away_ppda": stats[appda_col] if appda_col else pd.NA,
            "league": league,
            "source": "understat",
        }
    )
    for col in CLUB_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CLUB_MATCH_COLUMNS]


def bootstrap_understat_from_soccer(
    config: ClubConfig,
    output_path: Path,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build Understat-shaped parquet from soccer-dataset Big-5 leagues when fetch is unavailable."""
    from worldcup_predictor.data.club_cleaning import clean_club_matches
    from worldcup_predictor.data.club_loader import load_soccer_dataset

    data_dir = Path(config.data_dir)
    soccer = load_soccer_dataset(data_dir)
    soccer = clean_club_matches(
        soccer,
        min_date=pd.Timestamp(config.min_date, tz="UTC"),
        forward_fill_within_league=config.forward_fill_within_league,
    )

    big5 = {
        "premier league",
        "la liga",
        "bundesliga",
        "serie a",
        "ligue 1",
    }
    mask = soccer["league"].str.lower().isin(big5)
    out = soccer.loc[mask].copy()
    out["source"] = "understat"
    if len(out) < 1000:
        raise RuntimeError(
            f"Bootstrap Understat proxy produced only {len(out)} rows; "
            "download soccer-dataset with leagues.csv and teams.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    if show_progress:
        print(f"Bootstrapped {len(out):,} Understat proxy rows from soccer-dataset Big 5")
    return out

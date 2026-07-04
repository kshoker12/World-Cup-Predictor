"""Match sequence features for neural network models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from worldcup_predictor.config import AppConfig, EloConfig
from worldcup_predictor.features.state import tournament_importance
from worldcup_predictor.ratings.elo import expected_score, match_result_points, update_rating

SEQ_FEATURE_DIM = 10
# 0 gf, 1 ga, 2 opp_elo/1500, 3 log1p(days), 4 is_home, 5 importance,
# 6 xg_for, 7 xga, 8 shots_for, 9 feature_mask


def make_timestep_vector(
    *,
    goals_for: float,
    goals_against: float,
    opp_elo: float,
    days_since_prev: float,
    is_home: float,
    importance: float,
    xg_for: float = 0.0,
    xga: float = 0.0,
    shots_for: float = 0.0,
    feature_mask: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            goals_for,
            goals_against,
            opp_elo / 1500.0,
            float(np.log1p(max(days_since_prev, 0.0))),
            is_home,
            importance,
            xg_for,
            xga,
            shots_for,
            feature_mask,
        ],
        dtype=np.float32,
    )


@dataclass
class TeamSequenceState:
    seq_len: int
    _vectors: deque[np.ndarray] = field(default_factory=deque)
    _last_date: date | None = None

    def snapshot(self) -> np.ndarray:
        arr = np.zeros((self.seq_len, SEQ_FEATURE_DIM), dtype=np.float32)
        vecs = list(self._vectors)[-self.seq_len :]
        if vecs:
            arr[-len(vecs) :] = np.stack(vecs)
        return arr

    def update(
        self,
        vector: np.ndarray,
        match_date: date | None = None,
    ) -> None:
        self._vectors.append(vector.astype(np.float32))
        while len(self._vectors) > self.seq_len:
            self._vectors.popleft()
        if match_date is not None:
            self._last_date = match_date

    def days_since_prev(self, match_date: date) -> float:
        if self._last_date is None:
            return 0.0
        return float((match_date - self._last_date).days)

    def clone(self) -> TeamSequenceState:
        cloned = TeamSequenceState(seq_len=self.seq_len)
        cloned._vectors = deque(self._vectors, maxlen=self.seq_len)
        cloned._last_date = self._last_date
        return cloned


class ClubEloTracker:
    def __init__(self, elo_config: EloConfig) -> None:
        self.config = elo_config
        self._elo: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self._elo.get(team, self.config.initial)

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        *,
        neutral: bool = False,
    ) -> None:
        elo_home = self.get(home)
        elo_away = self.get(away)
        home_effective = elo_home + (
            0.0 if neutral else self.config.home_advantage
        )
        exp_home = expected_score(home_effective, elo_away)
        exp_away = 1.0 - exp_home
        actual_home, actual_away = match_result_points(home_score, away_score)
        self._elo[home] = update_rating(
            elo_home, exp_home, actual_home, self.config.k_factor
        )
        self._elo[away] = update_rating(
            elo_away, exp_away, actual_away, self.config.k_factor
        )


def _norm_stats_by_league(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, tuple[float, float]]]:
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for league, group in df.groupby("league"):
        stats[str(league)] = {}
        for col in columns:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values) == 0:
                stats[str(league)][col] = (0.0, 1.0)
            else:
                stats[str(league)][col] = (float(values.mean()), float(values.std() or 1.0))
    return stats


def _apply_norm(value: float, mean: float, std: float) -> float:
    if pd.isna(value):
        return 0.0
    if std <= 1e-9:
        return 0.0
    return float((value - mean) / std)


def build_club_sequences(
    matches: pd.DataFrame,
    *,
    seq_len: int,
    elo_config: EloConfig,
    norm_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return home_seq, away_seq, y_home, y_away arrays."""
    elo = ClubEloTracker(elo_config)
    team_states: dict[str, TeamSequenceState] = {}
    home_seqs: list[np.ndarray] = []
    away_seqs: list[np.ndarray] = []
    y_home: list[int] = []
    y_away: list[int] = []

    if norm_stats is None:
        norm_stats = _norm_stats_by_league(
            matches, ["home_xg", "away_xg", "home_shots", "away_shots"]
        )

    for row in matches.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        match_date = pd.Timestamp(row.date).date()
        league = str(row.league)
        league_stats = norm_stats.get(league, {})

        home_state = team_states.setdefault(home, TeamSequenceState(seq_len))
        away_state = team_states.setdefault(away, TeamSequenceState(seq_len))

        home_seqs.append(home_state.snapshot())
        away_seqs.append(away_state.snapshot())
        y_home.append(int(row.home_goals))
        y_away.append(int(row.away_goals))

        hxg_mean, hxg_std = league_stats.get("home_xg", (0.0, 1.0))
        axg_mean, axg_std = league_stats.get("away_xg", (0.0, 1.0))
        hs_mean, hs_std = league_stats.get("home_shots", (0.0, 1.0))

        home_vec = make_timestep_vector(
            goals_for=float(row.home_goals),
            goals_against=float(row.away_goals),
            opp_elo=elo.get(away),
            days_since_prev=home_state.days_since_prev(match_date),
            is_home=1.0,
            importance=1.0,
            xg_for=_apply_norm(getattr(row, "home_xg", np.nan), hxg_mean, hxg_std),
            xga=_apply_norm(getattr(row, "away_xg", np.nan), axg_mean, axg_std),
            shots_for=_apply_norm(getattr(row, "home_shots", np.nan), hs_mean, hs_std),
            feature_mask=1.0,
        )
        away_vec = make_timestep_vector(
            goals_for=float(row.away_goals),
            goals_against=float(row.home_goals),
            opp_elo=elo.get(home),
            days_since_prev=away_state.days_since_prev(match_date),
            is_home=0.0,
            importance=1.0,
            xg_for=_apply_norm(getattr(row, "away_xg", np.nan), axg_mean, axg_std),
            xga=_apply_norm(getattr(row, "home_xg", np.nan), hxg_mean, hxg_std),
            shots_for=_apply_norm(
                getattr(row, "away_shots", np.nan), hs_mean, hs_std
            ),
            feature_mask=1.0,
        )

        home_state.update(home_vec, match_date)
        away_state.update(away_vec, match_date)
        elo.update(home, away, int(row.home_goals), int(row.away_goals), neutral=False)

    return (
        np.stack(home_seqs),
        np.stack(away_seqs),
        np.array(y_home, dtype=np.float32),
        np.array(y_away, dtype=np.float32),
    )


def build_international_sequences(
    matches: pd.DataFrame,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sequences aligned to MatchPipeline row order."""
    from worldcup_predictor.features.pipeline import MatchPipeline

    pipeline = MatchPipeline(config)
    seq_len = config.nn.seq_len
    team_states: dict[str, TeamSequenceState] = {}
    home_seqs: list[np.ndarray] = []
    away_seqs: list[np.ndarray] = []

    for row in matches.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        neutral = bool(row.neutral)
        match_date = row.date
        home_state = team_states.setdefault(home, TeamSequenceState(seq_len))
        away_state = team_states.setdefault(away, TeamSequenceState(seq_len))

        home_seqs.append(home_state.snapshot())
        away_seqs.append(away_state.snapshot())

        opp_elo_home = pipeline.get_elo(away)
        opp_elo_away = pipeline.get_elo(home)
        importance = float(tournament_importance(row.tournament))

        home_vec = make_timestep_vector(
            goals_for=float(row.home_score),
            goals_against=float(row.away_score),
            opp_elo=opp_elo_home,
            days_since_prev=home_state.days_since_prev(match_date),
            is_home=0.0 if neutral else 1.0,
            importance=importance,
            feature_mask=0.0,
        )
        away_vec = make_timestep_vector(
            goals_for=float(row.away_score),
            goals_against=float(row.home_score),
            opp_elo=opp_elo_away,
            days_since_prev=away_state.days_since_prev(match_date),
            is_home=0.0,
            importance=importance,
            feature_mask=0.0,
        )
        home_state.update(home_vec, match_date)
        away_state.update(away_vec, match_date)
        pipeline.apply_match(
            home, away, int(row.home_score), int(row.away_score), neutral=neutral
        )

    return np.stack(home_seqs), np.stack(away_seqs)


def compute_club_norm_stats(matches: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    return _norm_stats_by_league(
        matches, ["home_xg", "away_xg", "home_shots", "away_shots"]
    )

"""
Central chronological match pipeline.

Same-day matches for the same team are processed in global sort order
(date, home_team, away_team), not per-team sub-sorting.
"""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from worldcup_predictor.config import AppConfig
from worldcup_predictor.features.state import H2HState, TeamRollingState, tournament_importance
from worldcup_predictor.models.sequences import TeamSequenceState, make_timestep_vector
from worldcup_predictor.ratings.elo import expected_score, match_result_points, update_rating
from worldcup_predictor.splits import assign_split

FEATURE_COLUMNS = [
    "elo_diff",
    "gf_last_5_diff",
    "ga_last_5_diff",
    "form_diff",
    "is_home",
    "is_neutral",
    "tournament_importance",
    "h2h_gd_weighted",
]


class MatchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._elo: dict[str, float] = {}
        self._teams: dict[str, TeamRollingState] = {}
        self._h2h = H2HState(h2h_decay=config.features.h2h_decay)
        self._seq_states: dict[str, TeamSequenceState] = {}

    def _get_seq_state(self, team: str) -> TeamSequenceState:
        if team not in self._seq_states:
            self._seq_states[team] = TeamSequenceState(self.config.nn.seq_len)
        return self._seq_states[team]

    def sequence_snapshot(self, home: str, away: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._get_seq_state(home).snapshot(),
            self._get_seq_state(away).snapshot(),
        )

    def _get_elo(self, team: str) -> float:
        return self._elo.get(team, self.config.elo.initial)

    def _get_team_state(self, team: str) -> TeamRollingState:
        if team not in self._teams:
            self._teams[team] = TeamRollingState(
                form_window=self.config.features.form_window,
                goals_window=self.config.features.rolling_goals_window,
                form_decay=self.config.features.form_decay,
            )
        return self._teams[team]

    def build_feature_row(
        self,
        home: str,
        away: str,
        *,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
        match_date: date | None = None,
    ) -> dict[str, Any]:
        elo_home = self._get_elo(home)
        elo_away = self._get_elo(away)
        home_state = self._get_team_state(home)
        away_state = self._get_team_state(away)
        is_neutral = 1 if neutral else 0
        is_home = 0 if neutral else 1

        row: dict[str, Any] = {
            "home_team": home,
            "away_team": away,
            "tournament": tournament,
            "neutral": neutral,
            "elo_diff": elo_home - elo_away,
            "gf_last_5_diff": home_state.snapshot_avg_goals_for()
            - away_state.snapshot_avg_goals_for(),
            "ga_last_5_diff": home_state.snapshot_avg_goals_against()
            - away_state.snapshot_avg_goals_against(),
            "form_diff": home_state.snapshot_form_points() - away_state.snapshot_form_points(),
            "is_home": is_home,
            "is_neutral": is_neutral,
            "tournament_importance": tournament_importance(tournament),
            "h2h_gd_weighted": self._h2h.snapshot(home, away),
        }
        if match_date is not None:
            row["date"] = match_date
            row["split"] = assign_split(match_date, self.config.splits)
        return row

    def apply_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        *,
        neutral: bool = False,
        match_date: date | None = None,
        tournament: str = "FIFA World Cup",
    ) -> None:
        elo_home = self._get_elo(home)
        elo_away = self._get_elo(away)
        home_effective = elo_home + (0.0 if neutral else self.config.elo.home_advantage)
        exp_home = expected_score(home_effective, elo_away)
        exp_away = 1.0 - exp_home
        actual_home, actual_away = match_result_points(home_score, away_score)

        self._elo[home] = update_rating(
            elo_home, exp_home, actual_home, self.config.elo.k_factor
        )
        self._elo[away] = update_rating(
            elo_away, exp_away, actual_away, self.config.elo.k_factor
        )

        self._get_team_state(home).update(home_score, away_score)
        self._get_team_state(away).update(away_score, home_score)
        self._h2h.update(home, away, home_score, away_score)

        home_seq = self._get_seq_state(home)
        away_seq = self._get_seq_state(away)
        importance = float(tournament_importance(tournament))
        home_vec = make_timestep_vector(
            goals_for=float(home_score),
            goals_against=float(away_score),
            opp_elo=elo_away,
            days_since_prev=home_seq.days_since_prev(match_date) if match_date else 0.0,
            is_home=0.0 if neutral else 1.0,
            importance=importance,
            feature_mask=0.0,
        )
        away_vec = make_timestep_vector(
            goals_for=float(away_score),
            goals_against=float(home_score),
            opp_elo=elo_home,
            days_since_prev=away_seq.days_since_prev(match_date) if match_date else 0.0,
            is_home=0.0,
            importance=importance,
            feature_mask=0.0,
        )
        home_seq.update(home_vec, match_date)
        away_seq.update(away_vec, match_date)

    def clone(self) -> MatchPipeline:
        cloned = MatchPipeline(self.config)
        cloned._elo = dict(self._elo)
        cloned._teams = copy.deepcopy(self._teams)
        cloned._h2h = copy.deepcopy(self._h2h)
        cloned._seq_states = {k: v.clone() for k, v in self._seq_states.items()}
        return cloned

    def run(
        self,
        matches: pd.DataFrame,
        *,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        from worldcup_predictor.utils.progress import progress

        rows: list[dict[str, Any]] = []
        iterator = progress(
            matches.itertuples(index=False),
            desc="Build features",
            total=len(matches),
            disable=not show_progress,
        )

        for row in iterator:
            home = row.home_team
            away = row.away_team
            neutral = bool(row.neutral)
            home_score = int(row.home_score)
            away_score = int(row.away_score)

            feature_row = self.build_feature_row(
                home,
                away,
                tournament=row.tournament,
                neutral=neutral,
                match_date=row.date,
            )
            feature_row["home_score"] = home_score
            feature_row["away_score"] = away_score
            rows.append(feature_row)

            self.apply_match(home, away, home_score, away_score, neutral=neutral)

        return pd.DataFrame(rows)

    def elo_ratings(self) -> dict[str, float]:
        """Return current Elo ratings after processing (for validation)."""
        return dict(self._elo)

    def get_elo(self, team: str) -> float:
        return self._get_elo(team)

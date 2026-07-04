"""Monte Carlo World Cup tournament simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from worldcup_predictor.config import AppConfig, TournamentConfig
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.features.pipeline import MatchPipeline
from worldcup_predictor.simulation.bracket import (
    advance_winners,
    build_knockout_round,
    build_round_of_16,
)
from worldcup_predictor.simulation.groups import GroupStandings, group_fixtures
from worldcup_predictor.simulation.match import simulate_match
from worldcup_predictor.simulation.state import (
    apply_result,
    clone_pipeline,
    features_for_fixture,
    sequences_for_fixture,
)
from worldcup_predictor.utils.progress import progress


@dataclass
class TournamentResult:
    champion_probs: dict[str, float]
    advancement_probs: dict[str, dict[str, float]]
    n_sims: int
    group_match_count: int
    knockout_match_count: int
    raw_counts: dict[str, dict[str, int]] = field(default_factory=dict)


class TournamentSimulator:
    def __init__(
        self,
        predictor: CalibratedPredictor,
        config: AppConfig,
        initial_pipeline: MatchPipeline,
        tournament: TournamentConfig,
        n_sims: int | None = None,
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        self.predictor = predictor
        self.config = config
        self.initial_pipeline = initial_pipeline
        self.tournament = tournament
        self.n_sims = n_sims or config.simulation.n_sims
        self.rho = predictor.rho
        self.max_goals = config.simulation.max_goals
        self.seed = seed
        self._show_progress = show_progress

    def _predict_lambda(
        self, pipeline: MatchPipeline, home: str, away: str
    ) -> tuple[float, float, float]:
        feat = features_for_fixture(pipeline, home, away, neutral=True)
        home_seq, away_seq = sequences_for_fixture(pipeline, home, away)
        pred = self.predictor.predict_lambda(
            feat,
            home_seq=home_seq[np.newaxis, ...],
            away_seq=away_seq[np.newaxis, ...],
        )
        elo_diff = float(feat.iloc[0]["elo_diff"])
        return (
            float(pred.iloc[0]["lambda_home"]),
            float(pred.iloc[0]["lambda_away"]),
            elo_diff,
        )

    def _play_match(
        self,
        pipeline: MatchPipeline,
        home: str,
        away: str,
        rng: np.random.Generator,
        *,
        knockout: bool,
    ):
        lh, la, elo_diff = self._predict_lambda(pipeline, home, away)
        outcome = simulate_match(
            home,
            away,
            lh,
            la,
            elo_diff,
            knockout=knockout,
            max_goals=self.max_goals,
            rho=self.rho,
            rng=rng,
        )
        apply_result(
            pipeline,
            home,
            away,
            outcome.home_goals,
            outcome.away_goals,
            neutral=True,
        )
        return outcome

    def _simulate_one(self, rng: np.random.Generator) -> tuple[str, dict[str, str]]:
        if self.tournament.groups is None:
            raise ValueError("Full tournament simulation requires groups in config")
        pipeline = clone_pipeline(self.initial_pipeline)
        standings: dict[str, GroupStandings] = {}

        for gname, teams in self.tournament.groups.items():
            gs = GroupStandings(group_name=gname)
            gs.init_teams(teams)
            standings[gname] = gs

        for gname, teams in self.tournament.groups.items():
            for home, away in group_fixtures(teams):
                outcome = self._play_match(pipeline, home, away, rng, knockout=False)
                standings[gname].record_result(
                    home, away, outcome.home_goals, outcome.away_goals
                )

        ranked = {
            g: [s.team for s in standings[g].ranked()]
            for g in self.tournament.groups
        }

        advancement: dict[str, str] = {}
        all_teams = [t for teams in self.tournament.groups.values() for t in teams]
        for t in all_teams:
            advancement[t] = "group"

        for _g, order in ranked.items():
            for team in order[:2]:
                advancement[team] = "round_of_16"

        r16 = build_round_of_16(ranked)
        r16_results = []
        for home, away in r16:
            outcome = self._play_match(pipeline, home, away, rng, knockout=True)
            r16_results.append((home, away, outcome.winner))
            advancement[outcome.winner] = "quarter_finals"

        qf_winners = advance_winners(r16_results)
        qf = build_knockout_round(qf_winners)
        qf_results = []
        for home, away in qf:
            outcome = self._play_match(pipeline, home, away, rng, knockout=True)
            qf_results.append((home, away, outcome.winner))
            advancement[outcome.winner] = "semi_finals"

        sf_winners = advance_winners(qf_results)
        sf = build_knockout_round(sf_winners)
        sf_results = []
        for home, away in sf:
            outcome = self._play_match(pipeline, home, away, rng, knockout=True)
            sf_results.append((home, away, outcome.winner))
            advancement[outcome.winner] = "final"

        final_winners = advance_winners(sf_results)
        final_home, final_away = build_knockout_round(final_winners)[0]
        final_outcome = self._play_match(
            pipeline, final_home, final_away, rng, knockout=True
        )
        champion = final_outcome.winner
        advancement[champion] = "champion"

        return champion, advancement

    def run(self) -> TournamentResult:
        if self.tournament.groups is None:
            raise ValueError("Full tournament simulation requires groups in config")
        rng = np.random.default_rng(self.seed)
        all_teams = [t for teams in self.tournament.groups.values() for t in teams]
        rounds = ["group", "round_of_16", "quarter_finals", "semi_finals", "final", "champion"]
        counts: dict[str, dict[str, int]] = {t: {r: 0 for r in rounds} for t in all_teams}

        group_matches = sum(
            len(group_fixtures(teams)) for teams in self.tournament.groups.values()
        )
        knockout_matches = 8 + 4 + 2 + 1  # R16 + QF + SF + Final

        for sim_idx in progress(
            range(self.n_sims),
            desc="Simulations",
            total=self.n_sims,
            disable=getattr(self, "_show_progress", True) is False,
        ):
            champion, advancement = self._simulate_one(rng)
            for team, reached in advancement.items():
                _increment_cumulative(counts, team, reached)

        champion_probs = {
            t: counts[t]["champion"] / self.n_sims for t in all_teams
        }
        advancement_probs = {
            t: {r: counts[t][r] / self.n_sims for r in rounds}
            for t in all_teams
        }

        return TournamentResult(
            champion_probs=champion_probs,
            advancement_probs=advancement_probs,
            n_sims=self.n_sims,
            group_match_count=group_matches,
            knockout_match_count=knockout_matches,
            raw_counts=counts,
        )


_ROUND_ORDER = [
    "group",
    "round_of_16",
    "quarter_finals",
    "semi_finals",
    "final",
    "champion",
]


def _increment_cumulative(
    counts: dict[str, dict[str, int]], team: str, reached: str
) -> None:
    for r in _ROUND_ORDER:
        counts[team][r] += 1
        if r == reached:
            break


def _round_rank(round_name: str) -> int:
    return _ROUND_ORDER.index(round_name)

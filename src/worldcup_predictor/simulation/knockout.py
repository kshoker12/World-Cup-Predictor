"""Knockout-only tournament simulation from a fixed Round of 16 bracket."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.config import AppConfig, TournamentConfig
from worldcup_predictor.features.pipeline import MatchPipeline
from worldcup_predictor.simulation.bracket import advance_winners, build_knockout_round
from worldcup_predictor.simulation.match import simulate_match
from worldcup_predictor.simulation.state import (
    apply_result,
    clone_pipeline,
    features_for_fixture,
    sequences_for_fixture,
)
from worldcup_predictor.simulation.tournament import TournamentResult
from worldcup_predictor.utils.progress import progress

_KNOCKOUT_ROUNDS = [
    "round_of_16",
    "quarter_finals",
    "semi_finals",
    "final",
    "champion",
]


def _increment_knockout_cumulative(
    counts: dict[str, dict[str, int]],
    team: str,
    reached: str,
    rounds: tuple[str, ...],
) -> None:
    for r in rounds:
        counts[team][r] += 1
        if r == reached:
            break


def _bracket_signature(bracket_path: dict[str, object], rounds: tuple[str, ...]) -> str:
    parts: list[str] = []
    for rnd in rounds:
        for match in bracket_path[rnd]:  # type: ignore[index]
            parts.append(f"{match['home']}|{match['away']}|{match['winner']}")
    final = bracket_path["final"]  # type: ignore[index]
    parts.append(f"{final['home']}|{final['away']}|{final['winner']}")
    return ";;".join(parts)


def _match_key(round_name: str, home: str, away: str) -> tuple[str, str, str]:
    return (round_name, home, away)


def _aggregate_match_win_probs(
    match_counts: dict[tuple[str, str, str], dict[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    round_order = {"round_of_16": 0, "quarter_finals": 1, "semi_finals": 2, "final": 3}
    for (round_name, home, away), counts in sorted(
        match_counts.items(),
        key=lambda item: (round_order.get(item[0][0], 99), item[0][1], item[0][2]),
    ):
        total = counts["total"]
        if total <= 0:
            continue
        p_home = counts["home_wins"] / total
        rows.append(
            {
                "round": round_name,
                "home": home,
                "away": away,
                "p_home_win": round(p_home, 6),
                "p_away_win": round(1.0 - p_home, 6),
                "n_sims": total,
            }
        )
    return rows


@dataclass
class KnockoutSimulationResult:
    tournament: TournamentResult
    sample_bracket: dict[str, object] = field(default_factory=dict)
    most_likely_bracket: dict[str, object] = field(default_factory=dict)
    most_likely_bracket_count: int = 0
    match_win_probs: list[dict[str, object]] = field(default_factory=list)


class KnockoutSimulator:
    """Monte Carlo simulation from Round of 16 or a later fixed knockout round."""

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
        if not tournament.is_knockout_only:
            raise ValueError("Tournament config must use mode=knockout_only")
        if tournament.starts_at_semi_finals:
            if not tournament.semi_finals:
                raise ValueError(
                    "semi_finals fixtures are required when start_round=semi_finals"
                )
            self._teams = sorted(
                {team for home, away in tournament.semi_finals for team in (home, away)}
            )
        elif tournament.starts_at_quarter_finals:
            if not tournament.quarter_finals:
                raise ValueError("quarter_finals fixtures are required when start_round=quarter_finals")
            self._teams = sorted(
                {team for home, away in tournament.quarter_finals for team in (home, away)}
            )
        else:
            if not tournament.round_of_16:
                raise ValueError("round_of_16 fixtures are required")
            self._teams = sorted(
                {team for home, away in tournament.round_of_16 for team in (home, away)}
            )

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
    ):
        lh, la, elo_diff = self._predict_lambda(pipeline, home, away)
        outcome = simulate_match(
            home,
            away,
            lh,
            la,
            elo_diff,
            knockout=True,
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

    def benchmark(self, n_warmup: int = 50) -> float:
        """Wall-clock seconds per knockout tournament simulation."""
        if n_warmup <= 0:
            raise ValueError("n_warmup must be positive")
        rng = np.random.default_rng(self.seed)
        started = time.monotonic()
        for _ in range(n_warmup):
            self._simulate_one(rng)
        return (time.monotonic() - started) / n_warmup

    def _build_quarterfinals(
        self, r16_winners: list[str]
    ) -> list[tuple[str, str]]:
        pairings = self.tournament.quarterfinal_pairings
        if not pairings:
            return build_knockout_round(r16_winners)
        fixtures: list[tuple[str, str]] = []
        for left_idx, right_idx in pairings:
            fixtures.append((r16_winners[left_idx], r16_winners[right_idx]))
        return fixtures

    def _knockout_match_count(self) -> int:
        if self.tournament.starts_at_semi_finals:
            return 3
        if self.tournament.starts_at_quarter_finals:
            return 7
        return 15

    def _initial_advancement(self) -> dict[str, str]:
        if self.tournament.starts_at_semi_finals:
            return {team: "semi_finals" for team in self._teams}
        if self.tournament.starts_at_quarter_finals:
            return {team: "quarter_finals" for team in self._teams}
        return {team: "round_of_16" for team in self._teams}

    def _empty_bracket_path(self) -> dict[str, object]:
        if self.tournament.starts_at_semi_finals:
            return {"semi_finals": [], "final": []}
        if self.tournament.starts_at_quarter_finals:
            return {"quarter_finals": [], "semi_finals": [], "final": []}
        return {"round_of_16": [], "quarter_finals": [], "semi_finals": [], "final": []}

    def _bracket_rounds_for_signature(self) -> tuple[str, ...]:
        if self.tournament.starts_at_semi_finals:
            return ("semi_finals",)
        if self.tournament.starts_at_quarter_finals:
            return ("quarter_finals", "semi_finals")
        return ("round_of_16", "quarter_finals", "semi_finals")

    def _count_rounds(self) -> tuple[str, ...]:
        if self.tournament.starts_at_semi_finals:
            return ("semi_finals", "final", "champion")
        if self.tournament.starts_at_quarter_finals:
            return ("quarter_finals", "semi_finals", "final", "champion")
        return tuple(_KNOCKOUT_ROUNDS)

    def _match_rounds_to_aggregate(self) -> tuple[str, ...]:
        if self.tournament.starts_at_semi_finals:
            return ("semi_finals",)
        if self.tournament.starts_at_quarter_finals:
            return ("quarter_finals", "semi_finals")
        return ("round_of_16", "quarter_finals", "semi_finals")

    def _simulate_one(
        self, rng: np.random.Generator
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        pipeline = clone_pipeline(self.initial_pipeline)
        advancement = self._initial_advancement()
        bracket_path = self._empty_bracket_path()

        if self.tournament.starts_at_semi_finals:
            sf = list(self.tournament.semi_finals)
        else:
            if self.tournament.starts_at_quarter_finals:
                qf = list(self.tournament.quarter_finals)
            else:
                r16_results = []
                for home, away in self.tournament.round_of_16:
                    outcome = self._play_match(pipeline, home, away, rng)
                    r16_results.append((home, away, outcome.winner))
                    advancement[outcome.winner] = "quarter_finals"
                    bracket_path["round_of_16"].append(
                        {
                            "home": home,
                            "away": away,
                            "winner": outcome.winner,
                            "score": f"{outcome.home_goals}-{outcome.away_goals}",
                        }
                    )

                qf_winners = advance_winners(r16_results)
                qf = self._build_quarterfinals(qf_winners)

            qf_results = []
            for home, away in qf:
                outcome = self._play_match(pipeline, home, away, rng)
                qf_results.append((home, away, outcome.winner))
                advancement[outcome.winner] = "semi_finals"
                bracket_path["quarter_finals"].append(
                    {
                        "home": home,
                        "away": away,
                        "winner": outcome.winner,
                        "score": f"{outcome.home_goals}-{outcome.away_goals}",
                    }
                )

            sf = build_knockout_round(advance_winners(qf_results))

        sf_results = []
        for home, away in sf:
            outcome = self._play_match(pipeline, home, away, rng)
            sf_results.append((home, away, outcome.winner))
            advancement[outcome.winner] = "final"
            bracket_path["semi_finals"].append(
                {
                    "home": home,
                    "away": away,
                    "winner": outcome.winner,
                    "score": f"{outcome.home_goals}-{outcome.away_goals}",
                }
            )

        final_winners = advance_winners(sf_results)
        final_home, final_away = build_knockout_round(final_winners)[0]
        final_outcome = self._play_match(pipeline, final_home, final_away, rng)
        champion = final_outcome.winner
        advancement[champion] = "champion"
        bracket_path["final"] = {
            "home": final_home,
            "away": final_away,
            "winner": champion,
            "score": f"{final_outcome.home_goals}-{final_outcome.away_goals}",
        }
        bracket_path["champion"] = champion

        return champion, advancement, bracket_path

    def run(self) -> KnockoutSimulationResult:
        rng = np.random.default_rng(self.seed)
        rounds = self._count_rounds()
        counts: dict[str, dict[str, int]] = {
            t: {r: 0 for r in rounds} for t in self._teams
        }
        sample_bracket: dict[str, object] = {}
        bracket_counts: dict[str, tuple[int, dict[str, object]]] = {}
        match_counts: dict[tuple[str, str, str], dict[str, int]] = {}
        knockout_matches = self._knockout_match_count()

        for sim_idx in progress(
            range(self.n_sims),
            desc="Knockout simulations",
            total=self.n_sims,
            disable=not self._show_progress,
        ):
            champion, advancement, bracket_path = self._simulate_one(rng)
            if sim_idx == 0:
                sample_bracket = bracket_path
            signature = _bracket_signature(
                bracket_path, self._bracket_rounds_for_signature()
            )
            prev = bracket_counts.get(signature)
            if prev is None:
                bracket_counts[signature] = (1, bracket_path)
            else:
                bracket_counts[signature] = (prev[0] + 1, prev[1])

            for round_name in self._match_rounds_to_aggregate():
                for match in bracket_path[round_name]:  # type: ignore[index]
                    key = _match_key(round_name, match["home"], match["away"])
                    bucket = match_counts.setdefault(
                        key, {"home_wins": 0, "total": 0}
                    )
                    bucket["total"] += 1
                    if match["winner"] == match["home"]:
                        bucket["home_wins"] += 1

            final = bracket_path["final"]  # type: ignore[index]
            final_key = _match_key("final", final["home"], final["away"])
            final_bucket = match_counts.setdefault(
                final_key, {"home_wins": 0, "total": 0}
            )
            final_bucket["total"] += 1
            if final["winner"] == final["home"]:
                final_bucket["home_wins"] += 1

            for team, reached in advancement.items():
                _increment_knockout_cumulative(counts, team, reached, rounds)

        most_likely_bracket: dict[str, object] = sample_bracket
        most_likely_bracket_count = 0
        if bracket_counts:
            signature, (mode_count, mode_bracket) = max(
                bracket_counts.items(), key=lambda item: item[1][0]
            )
            del signature
            most_likely_bracket = mode_bracket
            most_likely_bracket_count = mode_count

        champion_probs = {
            t: counts[t]["champion"] / self.n_sims for t in self._teams
        }
        advancement_probs = {
            t: {r: counts[t][r] / self.n_sims for r in rounds} for t in self._teams
        }

        tournament_result = TournamentResult(
            champion_probs=champion_probs,
            advancement_probs=advancement_probs,
            n_sims=self.n_sims,
            group_match_count=0,
            knockout_match_count=knockout_matches,
            raw_counts=counts,
        )
        return KnockoutSimulationResult(
            tournament=tournament_result,
            sample_bracket=sample_bracket,
            most_likely_bracket=most_likely_bracket,
            most_likely_bracket_count=most_likely_bracket_count,
            match_win_probs=_aggregate_match_win_probs(match_counts),
        )

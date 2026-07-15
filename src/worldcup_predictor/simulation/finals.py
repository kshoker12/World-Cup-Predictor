"""Simulation of fixed World Cup final and third-place fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.config import AppConfig, TournamentConfig
from worldcup_predictor.features.pipeline import MatchPipeline
from worldcup_predictor.simulation.match import MatchOutcome, simulate_match
from worldcup_predictor.simulation.state import (
    apply_result,
    clone_pipeline,
    features_for_fixture,
    sequences_for_fixture,
)
from worldcup_predictor.utils.progress import progress


@dataclass(frozen=True)
class FinalsSimulationResult:
    champion_probs: dict[str, float]
    match_win_probs: list[dict[str, object]]
    most_likely_results: dict[str, object]
    most_likely_results_count: int
    sample_results: dict[str, object]
    n_sims: int


class FinalsSimulator:
    """Simulate fixed third-place and final matches."""

    def __init__(
        self,
        predictor: CalibratedPredictor,
        config: AppConfig,
        initial_pipeline: MatchPipeline,
        tournament: TournamentConfig,
        *,
        n_sims: int,
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        if len(tournament.final) != 2 or len(tournament.third_place) != 2:
            raise ValueError("Final and third-place fixtures must each contain two teams")
        self.predictor = predictor
        self.config = config
        self.initial_pipeline = initial_pipeline
        self.tournament = tournament
        self.n_sims = n_sims
        self.seed = seed
        self.show_progress = show_progress

    def _play_match(
        self,
        pipeline: MatchPipeline,
        home: str,
        away: str,
        rng: np.random.Generator,
    ) -> MatchOutcome:
        features = features_for_fixture(pipeline, home, away, neutral=True)
        home_seq, away_seq = sequences_for_fixture(pipeline, home, away)
        pred = self.predictor.predict_lambda(
            features,
            home_seq=home_seq[np.newaxis, ...],
            away_seq=away_seq[np.newaxis, ...],
        )
        outcome = simulate_match(
            home,
            away,
            float(pred.iloc[0]["lambda_home"]),
            float(pred.iloc[0]["lambda_away"]),
            float(features.iloc[0]["elo_diff"]),
            knockout=True,
            max_goals=self.config.simulation.max_goals,
            rho=self.predictor.rho,
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

    @staticmethod
    def _match_payload(home: str, away: str, outcome: MatchOutcome) -> dict[str, str]:
        return {
            "home": home,
            "away": away,
            "winner": outcome.winner,
            "score": f"{outcome.home_goals}-{outcome.away_goals}",
        }

    def run(self) -> FinalsSimulationResult:
        rng = np.random.default_rng(self.seed)
        final_home, final_away = self.tournament.final
        bronze_home, bronze_away = self.tournament.third_place
        home_wins = {"final": 0, "third_place": 0}
        champions = {final_home: 0, final_away: 0}
        result_counts: dict[str, tuple[int, dict[str, object]]] = {}
        sample_results: dict[str, object] = {}

        for sim_idx in progress(
            range(self.n_sims),
            desc="Finals simulations",
            total=self.n_sims,
            disable=not self.show_progress,
        ):
            pipeline = clone_pipeline(self.initial_pipeline)
            bronze = self._play_match(
                pipeline, bronze_home, bronze_away, rng
            )
            final = self._play_match(pipeline, final_home, final_away, rng)
            if bronze.winner == bronze_home:
                home_wins["third_place"] += 1
            if final.winner == final_home:
                home_wins["final"] += 1
            champions[final.winner] += 1

            results = {
                "third_place": self._match_payload(
                    bronze_home, bronze_away, bronze
                ),
                "final": self._match_payload(final_home, final_away, final),
                "champion": final.winner,
                "bronze_winner": bronze.winner,
            }
            if sim_idx == 0:
                sample_results = results
            signature = f"{bronze.winner}|{final.winner}"
            previous = result_counts.get(signature)
            if previous is None:
                result_counts[signature] = (1, results)
            else:
                result_counts[signature] = (previous[0] + 1, previous[1])

        _, (mode_count, mode_results) = max(
            result_counts.items(), key=lambda item: item[1][0]
        )
        match_win_probs = [
            {
                "round": "final",
                "home": final_home,
                "away": final_away,
                "p_home_win": home_wins["final"] / self.n_sims,
                "p_away_win": 1.0 - home_wins["final"] / self.n_sims,
                "n_sims": self.n_sims,
            },
            {
                "round": "third_place",
                "home": bronze_home,
                "away": bronze_away,
                "p_home_win": home_wins["third_place"] / self.n_sims,
                "p_away_win": 1.0 - home_wins["third_place"] / self.n_sims,
                "n_sims": self.n_sims,
            },
        ]
        return FinalsSimulationResult(
            champion_probs={
                team: count / self.n_sims for team, count in champions.items()
            },
            match_win_probs=match_win_probs,
            most_likely_results=mode_results,
            most_likely_results_count=mode_count,
            sample_results=sample_results,
            n_sims=self.n_sims,
        )

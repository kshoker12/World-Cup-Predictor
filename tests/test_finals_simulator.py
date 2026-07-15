"""Tests for fixed final and third-place simulation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from worldcup_predictor.config import TournamentConfig
from worldcup_predictor.simulation.finals import FinalsSimulator
from worldcup_predictor.simulation.state import build_initial_pipeline


class StubPredictor:
    rho = -0.05

    def predict_lambda(self, features, home_seq, away_seq):
        del features, home_seq, away_seq
        return pd.DataFrame({"lambda_home": [1.4], "lambda_away": [1.2]})


def test_finals_simulator_runs_both_medal_matches(default_config):
    tournament = TournamentConfig(
        year=2026,
        kickoff_date=date(2026, 7, 18),
        mode="knockout_only",
        start_round="final",
        final=("Spain", "Argentina"),
        third_place=("France", "England"),
    )
    pipeline = build_initial_pipeline(
        pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "neutral",
            ]
        ),
        default_config,
        tournament.kickoff_date,
    )
    result = FinalsSimulator(
        StubPredictor(),  # type: ignore[arg-type]
        default_config,
        pipeline,
        tournament,
        n_sims=20,
        seed=0,
        show_progress=False,
    ).run()

    assert result.n_sims == 20
    assert pytest.approx(sum(result.champion_probs.values())) == 1.0
    assert {row["round"] for row in result.match_win_probs} == {
        "final",
        "third_place",
    }
    assert result.most_likely_results["champion"] in ("Spain", "Argentina")
    assert result.most_likely_results["bronze_winner"] in ("France", "England")

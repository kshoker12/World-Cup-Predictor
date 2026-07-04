"""Tests for knockout-only tournament simulation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.config import GBMConfig, TournamentConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.simulation.knockout import KnockoutSimulator
from worldcup_predictor.simulation.state import build_initial_pipeline


@pytest.fixture
def knockout_tournament() -> TournamentConfig:
    return TournamentConfig(
        year=2026,
        kickoff_date=date(2026, 7, 4),
        mode="knockout_only",
        round_of_16=(
            ("A1", "B2"),
            ("C1", "D2"),
            ("E1", "F2"),
            ("G1", "H2"),
            ("B1", "A2"),
            ("D1", "C2"),
            ("F1", "E2"),
            ("H1", "G2"),
        ),
        quarterfinal_pairings=((0, 1), (4, 5), (2, 3), (6, 7)),
    )


def test_knockout_simulator_runs(knockout_tournament, default_config):
    rng = np.random.default_rng(0)
    n = 300
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["home_score"] = rng.poisson(1.1, n)
    data["away_score"] = rng.poisson(1.0, n)
    data["date"] = [date(2000, 1, 1)] * n
    data["home_team"] = "X"
    data["away_team"] = "Y"
    data["tournament"] = "Friendly"
    data["neutral"] = False
    data["split"] = "train"
    df = pd.DataFrame(data)
    train_df = df.iloc[:250]
    val_df = df.iloc[250:]
    gbm = GBMPredictor(GBMConfig(num_boost_round=10, early_stopping_rounds=3))
    gbm.fit(train_df, val_df)

    from worldcup_predictor.calibration.artifacts import fit_calibration
    from worldcup_predictor.calibration.predictor import CalibratedPredictor

    artifacts = fit_calibration(
        gbm,
        val_df,
        default_config.calibration,
        default_config,
        max_goals=10,
    )
    predictor = CalibratedPredictor(gbm, artifacts, max_goals=10)

    historical = pd.DataFrame(
        [
            {
                "date": date(2005, 1, 1),
                "home_team": team,
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            }
            for team in {
                t for pair in knockout_tournament.round_of_16 for t in pair
            }
        ]
    )
    pipeline = build_initial_pipeline(
        historical, default_config, knockout_tournament.kickoff_date
    )

    sim = KnockoutSimulator(
        predictor,
        default_config,
        pipeline,
        knockout_tournament,
        n_sims=20,
        seed=0,
        show_progress=False,
    )
    result = sim.run()
    teams = {team for pair in knockout_tournament.round_of_16 for team in pair}

    assert result.tournament.group_match_count == 0
    assert result.tournament.knockout_match_count == 15
    assert set(result.tournament.champion_probs) == teams
    assert pytest.approx(sum(result.tournament.champion_probs.values())) == 1.0
    assert result.tournament.champion_probs
    assert result.sample_bracket["champion"] in teams
    assert result.most_likely_bracket["champion"] in teams
    assert result.most_likely_bracket_count >= 1
    assert result.match_win_probs
    r16_keys = {
        (row["home"], row["away"])
        for row in result.match_win_probs
        if row["round"] == "round_of_16"
    }
    assert len(r16_keys) == 8
    for row in result.match_win_probs:
        assert pytest.approx(row["p_home_win"] + row["p_away_win"]) == 1.0

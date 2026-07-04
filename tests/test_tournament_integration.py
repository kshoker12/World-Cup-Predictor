from datetime import date

import numpy as np
import pandas as pd

from worldcup_predictor.calibration.artifacts import fit_calibration
from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.config import GBMConfig, TournamentConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.simulation.state import build_initial_pipeline
from worldcup_predictor.simulation.tournament import TournamentSimulator


def _train_tiny_calibrated_predictor(
    train_df: pd.DataFrame, val_df: pd.DataFrame, default_config
) -> CalibratedPredictor:
    gbm = GBMPredictor(GBMConfig(num_boost_round=10, early_stopping_rounds=3))
    gbm.fit(train_df, val_df)
    artifacts = fit_calibration(
        gbm, val_df, default_config.calibration, default_config, max_goals=10
    )
    return CalibratedPredictor(gbm, artifacts, max_goals=10)


def test_mini_tournament_completes(default_config):
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

    predictor = _train_tiny_calibrated_predictor(train_df, val_df, default_config)

    historical = pd.DataFrame(
        [
            {
                "date": date(2005, 1, 1),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            }
        ]
    )
    pipeline = build_initial_pipeline(historical, default_config, date(2010, 1, 1))

    tournament = TournamentConfig(
        year=2010,
        kickoff_date=date(2010, 6, 1),
        actual_champion="A1",
        groups={g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"] for g in "ABCDEFGH"},
    )

    sim = TournamentSimulator(
        predictor,
        default_config,
        pipeline,
        tournament,
        n_sims=20,
        seed=123,
    )
    result = sim.run()

    assert result.group_match_count == 48  # 8 groups x 6
    assert abs(sum(result.champion_probs.values()) - 1.0) < 1e-9
    assert max(result.champion_probs.values()) > 0

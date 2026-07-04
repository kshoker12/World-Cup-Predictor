from datetime import date

import numpy as np
import pandas as pd

from worldcup_predictor.calibration.artifacts import fit_calibration
from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.config import TournamentConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.gbm import GBMPredictor
from worldcup_predictor.models.metrics import poisson_deviance
from worldcup_predictor.simulation.state import build_initial_pipeline
from worldcup_predictor.simulation.tournament import TournamentSimulator


def test_calibration_improves_val_deviance(default_config):
    rng = np.random.default_rng(1)
    n = 400
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["home_score"] = rng.poisson(1.3, n)
    data["away_score"] = rng.poisson(1.1, n)
    data["split"] = "train"
    df = pd.DataFrame(data)
    train_df = df.iloc[:300].copy()
    val_df = df.iloc[300:].copy()
    val_df["split"] = "val"

    gbm = GBMPredictor(default_config.gbm)
    gbm.fit(train_df, val_df)

    raw = gbm.predict_lambda(val_df)
    raw_dev = poisson_deviance(
        val_df["home_score"].to_numpy(), raw["lambda_home"].to_numpy()
    ) + poisson_deviance(
        val_df["away_score"].to_numpy(), raw["lambda_away"].to_numpy()
    )

    artifacts = fit_calibration(
        gbm, val_df, default_config.calibration, default_config, max_goals=10
    )
    predictor = CalibratedPredictor(gbm, artifacts, max_goals=10)
    cal = predictor.predict_lambda(val_df)
    cal_dev = poisson_deviance(
        val_df["home_score"].to_numpy(), cal["lambda_home"].to_numpy()
    ) + poisson_deviance(
        val_df["away_score"].to_numpy(), cal["lambda_away"].to_numpy()
    )

    assert cal_dev <= raw_dev + 1e-9


def test_calibrated_tournament_runs(default_config):
    rng = np.random.default_rng(0)
    n = 300
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["home_score"] = rng.poisson(1.1, n)
    data["away_score"] = rng.poisson(1.0, n)
    data["split"] = "train"
    df = pd.DataFrame(data)
    train_df = df.iloc[:250]
    val_df = df.iloc[250:]

    gbm = GBMPredictor(default_config.gbm)
    gbm.fit(train_df, val_df)
    artifacts = fit_calibration(
        gbm, val_df, default_config.calibration, default_config, max_goals=10
    )
    predictor = CalibratedPredictor(gbm, artifacts, max_goals=10)

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
    result = TournamentSimulator(
        predictor, default_config, pipeline, tournament, n_sims=5, seed=1
    ).run()
    assert abs(sum(result.champion_probs.values()) - 1.0) < 1e-9

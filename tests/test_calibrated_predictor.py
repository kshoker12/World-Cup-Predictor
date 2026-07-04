from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.calibration.artifacts import CalibrationArtifacts
from worldcup_predictor.calibration.dixon_coles import DixonColesParams
from worldcup_predictor.calibration.ensemble import EnsembleParams
from worldcup_predictor.calibration.predictor import CalibratedPredictor
from worldcup_predictor.calibration.scaling import ScalingParams
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.gbm import GBMPredictor


def test_wdl_sums_to_one(default_config):
    rng = np.random.default_rng(0)
    n = 50
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["home_score"] = rng.poisson(1.0, n)
    data["away_score"] = rng.poisson(1.0, n)
    df = pd.DataFrame(data)

    gbm = GBMPredictor(default_config.gbm)
    train_df = df.iloc[:40]
    val_df = df.iloc[40:]
    gbm.fit(train_df, val_df)

    artifacts = CalibrationArtifacts(
        scaling_gbm=ScalingParams(1.0, 1.0),
        scaling_nn=ScalingParams(1.0, 1.0),
        scaling_bayesian=ScalingParams(1.0, 1.0),
        ensemble=EnsembleParams(1.0, 0.0, 0.0),
        dixon_coles=DixonColesParams(rho=-0.05),
    )
    predictor = CalibratedPredictor(gbm, artifacts, max_goals=10)
    wdl = predictor.predict_wdl(val_df)
    totals = wdl["p_home_win"] + wdl["p_draw"] + wdl["p_away_win"]
    assert totals.between(0.999, 1.001).all()


def test_artifacts_save_load(tmp_path):
    artifacts = CalibrationArtifacts(
        scaling_gbm=ScalingParams(1.1, 0.9),
        scaling_nn=ScalingParams(1.0, 1.0),
        scaling_bayesian=ScalingParams(1.0, 1.0),
        ensemble=EnsembleParams(1.0, 0.0, 0.0),
        dixon_coles=DixonColesParams(rho=-0.03),
    )
    path = tmp_path / "calibration.json"
    artifacts.save(path)
    loaded = CalibrationArtifacts.load(path)
    assert loaded.scaling_gbm.s_home == 1.1
    assert loaded.dixon_coles.rho == -0.03

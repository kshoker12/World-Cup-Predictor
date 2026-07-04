from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.config import GBMConfig, TournamentConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.gbm import GBMPredictor


def test_gbm_train_predict_roundtrip(default_config, tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    rows = {
        col: rng.normal(size=n) for col in FEATURE_COLUMNS
    }
    rows["home_score"] = rng.poisson(1.2, n)
    rows["away_score"] = rng.poisson(1.0, n)
    rows["split"] = "train"
    df = pd.DataFrame(rows)
    val_df = df.iloc[:50].copy()
    train_df = df.iloc[50:].copy()

    gbm_cfg = GBMConfig(num_boost_round=20, early_stopping_rounds=5)
    predictor = GBMPredictor(gbm_cfg)
    predictor.fit(train_df, val_df)
    pred = predictor.predict_lambda(val_df)
    assert (pred["lambda_home"] > 0).all()
    assert (pred["lambda_away"] > 0).all()

    save_dir = tmp_path / "models"
    predictor.save(save_dir)
    loaded = GBMPredictor(gbm_cfg)
    loaded.load(save_dir)
    pred2 = loaded.predict_lambda(val_df)
    np.testing.assert_allclose(
        pred["lambda_home"].to_numpy(),
        pred2["lambda_home"].to_numpy(),
        rtol=1e-5,
    )

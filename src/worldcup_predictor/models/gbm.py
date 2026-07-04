"""LightGBM Poisson models for expected goal rates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import lightgbm as lgb
import numpy as np
import pandas as pd

from worldcup_predictor.config import AppConfig, GBMConfig
from worldcup_predictor.features.pipeline import FEATURE_COLUMNS
from worldcup_predictor.models.metrics import evaluate_goals
from worldcup_predictor.utils.progress import progress


@dataclass
class GBMMetrics:
    train: dict[str, float]
    val: dict[str, float]


class GBMPredictor:
    def __init__(
        self,
        gbm_config: GBMConfig | None = None,
        *,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.gbm_config = gbm_config or GBMConfig()
        self.model_home: lgb.Booster | None = None
        self.model_away: lgb.Booster | None = None
        self.feature_columns: list[str] = list(feature_columns or FEATURE_COLUMNS)

    def _train_one(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        *,
        desc: str = "Train GBM",
        show_progress: bool = True,
    ) -> lgb.Booster:
        params = {
            "objective": "poisson",
            "learning_rate": self.gbm_config.learning_rate,
            "num_leaves": self.gbm_config.num_leaves,
            "verbose": -1,
        }
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        rounds = self.gbm_config.num_boost_round
        pbar = progress(
            range(rounds),
            desc=desc,
            total=rounds,
            disable=not show_progress,
        )

        def _callback(env: lgb.callback.CallbackEnv) -> None:
            pbar.n = env.iteration + 1
            pbar.refresh()

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(
                    self.gbm_config.early_stopping_rounds, verbose=False
                ),
                _callback,
            ],
        )
        pbar.close()
        return booster

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        show_progress: bool = True,
    ) -> GBMMetrics:
        X_train = train_df[self.feature_columns]
        X_val = val_df[self.feature_columns]
        y_home_train = train_df["home_score"].to_numpy()
        y_away_train = train_df["away_score"].to_numpy()
        y_home_val = val_df["home_score"].to_numpy()
        y_away_val = val_df["away_score"].to_numpy()

        self.model_home = self._train_one(
            X_train,
            y_home_train,
            X_val,
            y_home_val,
            desc="Train GBM home",
            show_progress=show_progress,
        )
        self.model_away = self._train_one(
            X_train,
            y_away_train,
            X_val,
            y_away_val,
            desc="Train GBM away",
            show_progress=show_progress,
        )

        train_pred = self.predict_lambda(train_df)
        val_pred = self.predict_lambda(val_df)
        return GBMMetrics(
            train=evaluate_goals(
                y_home_train,
                y_away_train,
                train_pred["lambda_home"].to_numpy(),
                train_pred["lambda_away"].to_numpy(),
            ),
            val=evaluate_goals(
                y_home_val,
                y_away_val,
                val_pred["lambda_home"].to_numpy(),
                val_pred["lambda_away"].to_numpy(),
            ),
        )

    def predict_lambda(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Models not trained or loaded")
        X = features[self.feature_columns]
        lambda_home = self.model_home.predict(X)
        lambda_away = self.model_away.predict(X)
        return pd.DataFrame(
            {
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
            }
        )

    def evaluate(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        pred = self.predict_lambda(features)
        return evaluate_goals(
            features["home_score"].to_numpy(),
            features["away_score"].to_numpy(),
            pred["lambda_home"].to_numpy(),
            pred["lambda_away"].to_numpy(),
        )

    def save(self, directory: Path) -> None:
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Models not trained")
        directory.mkdir(parents=True, exist_ok=True)
        self.model_home.save_model(str(directory / "gbm_home.txt"))
        self.model_away.save_model(str(directory / "gbm_away.txt"))
        meta_path = directory / "gbm_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"feature_columns": self.feature_columns}, f, indent=2)

    def load(self, directory: Path) -> None:
        meta_path = directory / "gbm_meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_columns = list(meta.get("feature_columns", self.feature_columns))
        self.model_home = lgb.Booster(model_file=str(directory / "gbm_home.txt"))
        self.model_away = lgb.Booster(model_file=str(directory / "gbm_away.txt"))


def train_from_features(
    features_path: Path,
    config: AppConfig,
    *,
    show_progress: bool = True,
) -> tuple[GBMPredictor, GBMMetrics, dict[str, float]]:
    df = pd.read_parquet(features_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    predictor = GBMPredictor(config.gbm)
    metrics = predictor.fit(train_df, val_df, show_progress=show_progress)
    test_metrics = predictor.evaluate(test_df)
    return predictor, metrics, test_metrics

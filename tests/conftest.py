"""Shared pytest fixtures."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from worldcup_predictor.config import (
    AppConfig,
    BayesianConfig,
    CalibrationConfig,
    ClubConfig,
    EloConfig,
    FeatureConfig,
    GBMConfig,
    NNConfig,
    PipelineConfig,
    SimulationConfig,
    SplitConfig,
)


def pytest_configure(config: pytest.Config) -> None:
    # macOS: LightGBM and PyTorch both link OpenMP; allow coexistence in one process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig(
        elo=EloConfig(),
        features=FeatureConfig(),
        splits=SplitConfig(
            train_end=date(2010, 1, 1),
            val_end=date(2018, 1, 1),
        ),
        gbm=GBMConfig(num_boost_round=10, early_stopping_rounds=5),
        simulation=SimulationConfig(n_sims=100, max_goals=10),
        calibration=CalibrationConfig(),
        club=ClubConfig(),
        nn=NNConfig(
            pretrain_epochs=2,
            finetune_epochs=2,
            pretrain_batch_size=32,
            finetune_batch_size=32,
            device="cpu",
        ),
        bayesian=BayesianConfig(draws=50, tune=50, chains=1),
        pipeline=PipelineConfig(),
        team_aliases={"Czech Republic": "Czechia"},
    )


def make_match(
    d: str,
    home: str,
    away: str,
    hs: int,
    aws: int,
    tournament: str = "Friendly",
    neutral: bool = False,
) -> dict:
    return {
        "date": date.fromisoformat(d),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": aws,
        "tournament": tournament,
        "neutral": neutral,
    }


def matches_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run torch-dependent tests last to avoid LightGBM/OpenMP conflicts on macOS."""
    torch_tests = [
        item
        for item in items
        if any(
            token in item.nodeid
            for token in ("test_nn_", "test_device.py", "test_nn_predictor")
        )
    ]
    other = [item for item in items if item not in torch_tests]
    items[:] = other + torch_tests

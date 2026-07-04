"""Bayesian model training orchestration."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from worldcup_predictor.config import AppConfig
from worldcup_predictor.models.bayesian.artifacts import BayesianArtifacts
from worldcup_predictor.models.bayesian.model import build_pymc_model, prepare_match_data


def _ensure_pytensor_cache() -> None:
    root = Path(__file__).resolve().parents[4]
    cache = root / ".pytensor"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={cache}")


def filter_bayesian_matches(
    df: pd.DataFrame,
    *,
    min_date: date,
    val_end: date,
    split_scope: str = "train",
) -> pd.DataFrame:
    """Select matches from min_date onward for Bayesian fit.

    split_scope:
      - ``train``: train split only (fair ensemble comparison)
      - ``train_val``: train + validation (legacy rho-only experiments)
    """
    if split_scope not in {"train", "train_val"}:
        raise ValueError(f"Unknown split_scope: {split_scope}")

    splits = ["train"] if split_scope == "train" else ["train", "val"]
    dates = pd.to_datetime(df["date"]).dt.date
    mask = (
        df["split"].isin(splits)
        & (dates >= min_date)
        & (dates < val_end)
    )
    return df.loc[mask].reset_index(drop=True)


def fit_bayesian(
    df: pd.DataFrame,
    config: AppConfig,
    *,
    show_progress: bool = True,
) -> BayesianArtifacts:
    """Fit hierarchical Dixon-Coles model and return artifact summary."""
    _ensure_pytensor_cache()
    import arviz as az
    import pymc as pm

    bayesian_cfg = config.bayesian
    cal_cfg = config.calibration

    fit_df = filter_bayesian_matches(
        df,
        min_date=bayesian_cfg.min_date,
        val_end=config.splits.val_end,
        split_scope=bayesian_cfg.split_scope,
    )
    if fit_df.empty:
        raise ValueError("No matches available for Bayesian fit after filtering")

    data = prepare_match_data(fit_df)
    model = build_pymc_model(
        data,
        rho_min=cal_cfg.rho_min,
        rho_max=cal_cfg.rho_max,
    )

    with model:
        idata = pm.sample(
            draws=bayesian_cfg.draws,
            tune=bayesian_cfg.tune,
            chains=bayesian_cfg.chains,
            target_accept=bayesian_cfg.target_accept,
            random_seed=bayesian_cfg.random_seed,
            progressbar=show_progress,
        )

    summary_df = az.summary(idata, var_names=["rho", "intercept", "beta"])
    summary = summary_df.to_dict(orient="index")

    att_post = idata.posterior["att"].mean(dim=("chain", "draw")).values
    def_post = idata.posterior["def"].mean(dim=("chain", "draw")).values
    index_to_team = {idx: team for team, idx in data.team_index.items()}
    att_mean = {index_to_team[i]: float(att_post[i]) for i in range(len(att_post))}
    def_mean = {index_to_team[i]: float(def_post[i]) for i in range(len(def_post))}

    return BayesianArtifacts.from_posterior_summary(
        summary,
        data=data,
        att_mean=att_mean,
        def_mean=def_mean,
        chains=bayesian_cfg.chains,
        draws=bayesian_cfg.draws,
        tune=bayesian_cfg.tune,
    )


def train_from_features(
    features_path: Path,
    config: AppConfig,
    *,
    show_progress: bool = True,
) -> BayesianArtifacts:
    df = pd.read_parquet(features_path).reset_index(drop=True)
    return fit_bayesian(df, config, show_progress=show_progress)

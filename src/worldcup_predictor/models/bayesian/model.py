"""Hierarchical Dixon-Coles Poisson model for PyMC."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from worldcup_predictor.features.pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class BayesianMatchData:
    """Prepared arrays for PyMC fitting."""

    home_idx: np.ndarray
    away_idx: np.ndarray
    y_home: np.ndarray
    y_away: np.ndarray
    x: np.ndarray
    team_index: dict[str, int]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    n_teams: int
    n_matches: int


def dc_tau(i: int, j: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if rho == 0.0:
        return 1.0
    if i == 0 and j == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if i == 0 and j == 1:
        return 1.0 + lambda_home * rho
    if i == 1 and j == 0:
        return 1.0 + lambda_away * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _poisson_logpmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-9)
    return k * np.log(lam) - lam - math.lgamma(k + 1)


def match_log_prob_numpy(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    tau = dc_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
    if tau <= 0:
        return -np.inf
    return (
        _poisson_logpmf(home_goals, lambda_home)
        + _poisson_logpmf(away_goals, lambda_away)
        + np.log(tau)
    )


def dixon_coles_logp_numpy(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    y_h = np.asarray(y_home, dtype=int)
    y_a = np.asarray(y_away, dtype=int)
    lh = np.asarray(lambda_home, dtype=float)
    la = np.asarray(lambda_away, dtype=float)
    out = np.zeros(len(y_h), dtype=float)
    for i in range(len(y_h)):
        out[i] = match_log_prob_numpy(y_h[i], y_a[i], lh[i], la[i], rho)
    return out


def prepare_match_data(df) -> BayesianMatchData:
    """Build team indices and standardized feature matrix from a match DataFrame."""
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    team_index = {team: idx for idx, team in enumerate(teams)}

    home_idx = df["home_team"].map(team_index).to_numpy(dtype=np.int32)
    away_idx = df["away_team"].map(team_index).to_numpy(dtype=np.int32)
    y_home = df["home_score"].to_numpy(dtype=np.int32)
    y_away = df["away_score"].to_numpy(dtype=np.int32)

    x_raw = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    feature_means = x_raw.mean(axis=0)
    feature_stds = x_raw.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    x = (x_raw - feature_means) / feature_stds

    return BayesianMatchData(
        home_idx=home_idx,
        away_idx=away_idx,
        y_home=y_home,
        y_away=y_away,
        x=x,
        team_index=team_index,
        feature_means=feature_means,
        feature_stds=feature_stds,
        n_teams=len(teams),
        n_matches=len(df),
    )


def build_pymc_model(
    data: BayesianMatchData,
    *,
    rho_min: float,
    rho_max: float,
):
    """Build PyMC hierarchical Dixon-Coles model."""
    import pymc as pm
    import pytensor.tensor as pt

    coords = {
        "team": np.arange(data.n_teams),
        "feature": list(FEATURE_COLUMNS),
        "match": np.arange(data.n_matches),
    }

    with pm.Model(coords=coords) as model:
        home_idx = pm.Data("home_idx", data.home_idx, dims="match")
        away_idx = pm.Data("away_idx", data.away_idx, dims="match")
        x = pm.Data("x", data.x, dims=("match", "feature"))
        y_home = pm.Data("y_home", data.y_home, dims="match")
        y_away = pm.Data("y_away", data.y_away, dims="match")

        intercept = pm.Normal("intercept", 0, 1)
        sigma_team = pm.HalfNormal("sigma_team", 1)
        att_raw = pm.Normal("att_raw", 0, sigma_team, dims="team")
        def_raw = pm.Normal("def_raw", 0, sigma_team, dims="team")
        att = pm.Deterministic("att", att_raw - pt.mean(att_raw), dims="team")
        defense = pm.Deterministic("def", def_raw - pt.mean(def_raw), dims="team")
        beta = pm.Normal("beta", 0, 1, dims="feature")
        rho = pm.Uniform("rho", lower=rho_min, upper=rho_max)

        lin = pt.dot(x, beta)
        eta_home = intercept + att[home_idx] - defense[away_idx] + lin
        eta_away = intercept + att[away_idx] - defense[home_idx] + lin
        lambda_home = pm.Deterministic("lambda_home", pt.exp(eta_home), dims="match")
        lambda_away = pm.Deterministic("lambda_away", pt.exp(eta_away), dims="match")

        pois_home = pm.logp(pm.Poisson.dist(mu=lambda_home), y_home)
        pois_away = pm.logp(pm.Poisson.dist(mu=lambda_away), y_away)

        h0 = pt.eq(y_home, 0)
        h1 = pt.eq(y_home, 1)
        a0 = pt.eq(y_away, 0)
        a1 = pt.eq(y_away, 1)

        tau = pt.ones_like(lambda_home)
        tau = pt.switch(h0 & a0, 1.0 - lambda_home * lambda_away * rho, tau)
        tau = pt.switch(h0 & a1, 1.0 + lambda_home * rho, tau)
        tau = pt.switch(h1 & a0, 1.0 + lambda_away * rho, tau)
        tau = pt.switch(h1 & a1, 1.0 - rho, tau)
        log_tau = pt.log(pt.maximum(tau, 1e-12))

        pm.Potential("dc_ll", pt.sum(pois_home + pois_away + log_tau))

    return model

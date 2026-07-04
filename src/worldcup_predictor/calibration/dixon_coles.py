"""Dixon-Coles rho estimation via MLE on validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from worldcup_predictor.simulation.score_grid import build_score_grid


@dataclass(frozen=True)
class DixonColesParams:
    rho: float = 0.0


def _match_log_prob(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = 10,
) -> float:
    grid = build_score_grid(
        lambda_home,
        lambda_away,
        max_goals=max_goals,
        rho=rho,
    )
    i = min(int(home_goals), max_goals)
    j = min(int(away_goals), max_goals)
    prob = grid[i, j]
    return float(np.log(max(prob, 1e-15)))


def fit_dixon_coles_rho(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    rho_min: float = -0.2,
    rho_max: float = 0.1,
    max_goals: int = 10,
) -> DixonColesParams:
    y_h = np.asarray(y_home, dtype=int)
    y_a = np.asarray(y_away, dtype=int)
    lh = np.asarray(lambda_home, dtype=float)
    la = np.asarray(lambda_away, dtype=float)

    def neg_log_likelihood(rho: float) -> float:
        total = 0.0
        for i in range(len(y_h)):
            total += _match_log_prob(y_h[i], y_a[i], lh[i], la[i], rho, max_goals)
        return -total

    try:
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(rho_min, rho_max),
            method="bounded",
        )
        if result.success:
            return DixonColesParams(rho=float(result.x))
    except (ValueError, FloatingPointError):
        pass
    return DixonColesParams(rho=0.0)

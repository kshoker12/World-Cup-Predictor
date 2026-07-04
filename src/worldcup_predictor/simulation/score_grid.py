"""Poisson score probability grid with optional Dixon-Coles adjustment."""

from __future__ import annotations

import math

import numpy as np


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _dc_tau(i: int, j: int, lambda_home: float, lambda_away: float, rho: float) -> float:
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


def build_score_grid(
    lambda_home: float,
    lambda_away: float,
    *,
    max_goals: int = 10,
    rho: float = 0.0,
) -> np.ndarray:
    """Return (max_goals+1) x (max_goals+1) probability grid, normalized to sum 1."""
    lam_h = max(float(lambda_home), 1e-9)
    lam_a = max(float(lambda_away), 1e-9)
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] = (
                _poisson_pmf(i, lam_h)
                * _poisson_pmf(j, lam_a)
                * _dc_tau(i, j, lam_h, lam_a, rho)
            )
    total = grid.sum()
    if total <= 0:
        raise ValueError("Score grid sums to zero")
    return grid / total


def wdl_probabilities(grid: np.ndarray) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) from score grid."""
    p_draw = float(np.trace(grid))
    p_home = float(np.tril(grid, k=-1).sum())
    p_away = float(np.triu(grid, k=1).sum())
    return p_home, p_draw, p_away

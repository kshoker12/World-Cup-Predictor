"""Multiplicative lambda calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from worldcup_predictor.models.metrics import poisson_deviance


@dataclass(frozen=True)
class ScalingParams:
    s_home: float = 1.0
    s_away: float = 1.0

    def apply(
        self,
        lambda_home: np.ndarray | float,
        lambda_away: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lh = np.asarray(lambda_home, dtype=float) * self.s_home
        la = np.asarray(lambda_away, dtype=float) * self.s_away
        return lh, la


def _fit_one_scale(
    y_true: np.ndarray,
    lambda_raw: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    def objective(s: float) -> float:
        return poisson_deviance(y_true, lambda_raw * s)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x)


def fit_scaling(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.5, 2.0),
) -> ScalingParams:
    s_home = _fit_one_scale(y_home, lambda_home, bounds)
    s_away = _fit_one_scale(y_away, lambda_away, bounds)
    return ScalingParams(s_home=s_home, s_away=s_away)

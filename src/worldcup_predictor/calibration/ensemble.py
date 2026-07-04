"""Fit ensemble weights with optional minimum per-model floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldcup_predictor.models.metrics import poisson_deviance


@dataclass(frozen=True)
class EnsembleParams:
    w_gbm: float = 1.0
    w_nn: float = 0.0
    w_bayesian: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> EnsembleParams:
        return cls(
            w_gbm=float(data.get("w_gbm", 1.0)),
            w_nn=float(data.get("w_nn", 0.0)),
            w_bayesian=float(data.get("w_bayesian", 0.0)),
        )


def combine_lambda(
    lambda_home_gbm: np.ndarray,
    lambda_away_gbm: np.ndarray,
    weights: EnsembleParams,
    lambda_home_nn: np.ndarray | None = None,
    lambda_away_nn: np.ndarray | None = None,
    lambda_home_bayes: np.ndarray | None = None,
    lambda_away_bayes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lh = weights.w_gbm * np.asarray(lambda_home_gbm, dtype=float)
    la = weights.w_gbm * np.asarray(lambda_away_gbm, dtype=float)
    if lambda_home_nn is not None and lambda_away_nn is not None and weights.w_nn > 0:
        lh = lh + weights.w_nn * np.asarray(lambda_home_nn, dtype=float)
        la = la + weights.w_nn * np.asarray(lambda_away_nn, dtype=float)
    if (
        lambda_home_bayes is not None
        and lambda_away_bayes is not None
        and weights.w_bayesian > 0
    ):
        lh = lh + weights.w_bayesian * np.asarray(lambda_home_bayes, dtype=float)
        la = la + weights.w_bayesian * np.asarray(lambda_away_bayes, dtype=float)
    return lh, la


def _deviance(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lh: np.ndarray,
    la: np.ndarray,
) -> float:
    return poisson_deviance(y_home, lh) + poisson_deviance(y_away, la)


def _normalize_three(w_gbm: float, w_nn: float, w_bay: float) -> EnsembleParams:
    total = w_gbm + w_nn + w_bay
    if total <= 0:
        return EnsembleParams(w_gbm=1 / 3, w_nn=1 / 3, w_bayesian=1 / 3)
    return EnsembleParams(
        w_gbm=w_gbm / total,
        w_nn=w_nn / total,
        w_bayesian=w_bay / total,
    )


def _apply_min_weight(
    weights: EnsembleParams,
    *,
    min_weight: float,
    active: tuple[bool, bool, bool],
) -> EnsembleParams:
    """Project weights onto simplex with per-active-model floor."""
    if min_weight <= 0:
        return weights

    n_active = sum(active)
    if n_active == 0:
        return weights
    if n_active * min_weight >= 1.0:
        equal = 1.0 / n_active
        return EnsembleParams(
            w_gbm=equal if active[0] else 0.0,
            w_nn=equal if active[1] else 0.0,
            w_bayesian=equal if active[2] else 0.0,
        )

    raw = np.array(
        [
            weights.w_gbm if active[0] else 0.0,
            weights.w_nn if active[1] else 0.0,
            weights.w_bayesian if active[2] else 0.0,
        ],
        dtype=float,
    )
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raw = np.array([1.0 if flag else 0.0 for flag in active], dtype=float)

    remaining = 1.0 - n_active * min_weight
    scaled = raw / raw.sum() * remaining
    adjusted = scaled + np.array(
        [min_weight if flag else 0.0 for flag in active], dtype=float
    )
    return EnsembleParams(
        w_gbm=float(adjusted[0]),
        w_nn=float(adjusted[1]),
        w_bayesian=float(adjusted[2]),
    )


def fit_ensemble_weights(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home_gbm: np.ndarray,
    lambda_away_gbm: np.ndarray,
    lambda_home_nn: np.ndarray | None = None,
    lambda_away_nn: np.ndarray | None = None,
    lambda_home_bayes: np.ndarray | None = None,
    lambda_away_bayes: np.ndarray | None = None,
    *,
    min_weight: float = 0.0,
) -> EnsembleParams:
    """Optimize weights on simplex minimizing validation Poisson deviance."""
    has_nn = lambda_home_nn is not None and lambda_away_nn is not None
    has_bayes = lambda_home_bayes is not None and lambda_away_bayes is not None

    if not has_nn and not has_bayes:
        return EnsembleParams(w_gbm=1.0, w_nn=0.0, w_bayesian=0.0)

    from scipy.optimize import minimize

    if has_nn and not has_bayes:
        def objective(w_nn: float) -> float:
            w_gbm = 1.0 - w_nn
            lh = w_gbm * lambda_home_gbm + w_nn * lambda_home_nn
            la = w_gbm * lambda_away_gbm + w_nn * lambda_away_nn
            return _deviance(y_home, y_away, lh, la)

        lo = min_weight if min_weight > 0 else 0.0
        hi = 1.0 - lo
        result = minimize(objective, x0=0.5, bounds=[(lo, hi)], method="L-BFGS-B")
        w_nn = float(np.clip(result.x[0], lo, hi))
        weights = EnsembleParams(w_gbm=1.0 - w_nn, w_nn=w_nn, w_bayesian=0.0)
        return _apply_min_weight(weights, min_weight=min_weight, active=(True, True, False))

    if has_bayes and not has_nn:
        def objective(w_bay: float) -> float:
            w_gbm = 1.0 - w_bay
            lh = w_gbm * lambda_home_gbm + w_bay * lambda_home_bayes
            la = w_gbm * lambda_away_gbm + w_bay * lambda_away_bayes
            return _deviance(y_home, y_away, lh, la)

        lo = min_weight if min_weight > 0 else 0.0
        hi = 1.0 - lo
        result = minimize(objective, x0=0.5, bounds=[(lo, hi)], method="L-BFGS-B")
        w_bay = float(np.clip(result.x[0], lo, hi))
        weights = EnsembleParams(w_gbm=1.0 - w_bay, w_nn=0.0, w_bayesian=w_bay)
        return _apply_min_weight(weights, min_weight=min_weight, active=(True, False, True))

    assert has_nn and has_bayes

    lo = min_weight if min_weight > 0 else 0.0
    hi = max(lo, 1.0 - 2 * lo)

    def objective(x: np.ndarray) -> float:
        w_nn, w_bay = float(x[0]), float(x[1])
        w_gbm = 1.0 - w_nn - w_bay
        if w_gbm < lo - 1e-9:
            return 1e12
        lh = (
            w_gbm * lambda_home_gbm
            + w_nn * lambda_home_nn
            + w_bay * lambda_home_bayes
        )
        la = (
            w_gbm * lambda_away_gbm
            + w_nn * lambda_away_nn
            + w_bay * lambda_away_bayes
        )
        return _deviance(y_home, y_away, lh, la)

    result = minimize(
        objective,
        x0=np.array([1 / 3, 1 / 3]),
        bounds=[(lo, hi), (lo, hi)],
        constraints={"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1] - lo},
        method="SLSQP",
    )
    w_nn = float(np.clip(result.x[0], lo, hi))
    w_bay = float(np.clip(result.x[1], lo, hi))
    weights = _normalize_three(1.0 - w_nn - w_bay, w_nn, w_bay)
    return _apply_min_weight(
        weights,
        min_weight=min_weight,
        active=(True, True, True),
    )

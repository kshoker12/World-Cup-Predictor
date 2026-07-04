"""Evaluation metrics for goal prediction models."""

from __future__ import annotations

import numpy as np

from worldcup_predictor.simulation.score_grid import build_score_grid, wdl_probabilities


def poisson_deviance(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    """Mean Poisson deviance (2 * per-obs NLL up to constant)."""
    y = np.asarray(y_true, dtype=float)
    lam = np.clip(np.asarray(lambda_pred, dtype=float), 1e-9, None)
    term = np.zeros_like(y)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / lam[positive])
    return float(2.0 * np.mean(lam - y + term))


def goal_rmse(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    lam = np.asarray(lambda_pred, dtype=float)
    return float(np.sqrt(np.mean((y - lam) ** 2)))


def goal_mae(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    lam = np.asarray(lambda_pred, dtype=float)
    return float(np.mean(np.abs(y - lam)))


def wdl_log_loss(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    p_home_win: np.ndarray,
    p_draw: np.ndarray,
    p_away_win: np.ndarray,
) -> float:
    """Multiclass log loss for win/draw/loss from score-grid probabilities."""
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    p_h = np.clip(np.asarray(p_home_win, dtype=float), 1e-15, 1.0)
    p_d = np.clip(np.asarray(p_draw, dtype=float), 1e-15, 1.0)
    p_a = np.clip(np.asarray(p_away_win, dtype=float), 1e-15, 1.0)

    outcome_h = (hg > ag).astype(float)
    outcome_d = (hg == ag).astype(float)
    outcome_a = (hg < ag).astype(float)
    ll = outcome_h * np.log(p_h) + outcome_d * np.log(p_d) + outcome_a * np.log(p_a)
    return float(-np.mean(ll))


def brier_score_wdl(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    p_home_win: np.ndarray,
    p_draw: np.ndarray,
    p_away_win: np.ndarray,
) -> float:
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    p_h = np.asarray(p_home_win, dtype=float)
    p_d = np.asarray(p_draw, dtype=float)
    p_a = np.asarray(p_away_win, dtype=float)

    y_h = (hg > ag).astype(float)
    y_d = (hg == ag).astype(float)
    y_a = (hg < ag).astype(float)
    return float(
        np.mean((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2)
    )


def expected_calibration_error_wdl(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    p_home_win: np.ndarray,
    p_draw: np.ndarray,
    p_away_win: np.ndarray,
    n_bins: int = 10,
) -> float:
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    probs = np.stack(
        [
            np.asarray(p_home_win, dtype=float),
            np.asarray(p_draw, dtype=float),
            np.asarray(p_away_win, dtype=float),
        ],
        axis=1,
    )
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    outcomes = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if not mask.any():
            continue
        acc = float(np.mean(predictions[mask] == outcomes[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def wdl_from_lambdas(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    rho: float = 0.0,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_h_list: list[float] = []
    p_d_list: list[float] = []
    p_a_list: list[float] = []
    for lh, la in zip(lambda_home, lambda_away):
        grid = build_score_grid(lh, la, max_goals=max_goals, rho=rho)
        p_h, p_d, p_a = wdl_probabilities(grid)
        p_h_list.append(p_h)
        p_d_list.append(p_d)
        p_a_list.append(p_a)
    return (
        np.asarray(p_h_list),
        np.asarray(p_d_list),
        np.asarray(p_a_list),
    )


def evaluate_goals(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> dict[str, float]:
    return {
        "poisson_deviance_home": poisson_deviance(y_home, lambda_home),
        "poisson_deviance_away": poisson_deviance(y_away, lambda_away),
        "poisson_deviance_total": poisson_deviance(y_home, lambda_home)
        + poisson_deviance(y_away, lambda_away),
        "rmse_home": goal_rmse(y_home, lambda_home),
        "rmse_away": goal_rmse(y_away, lambda_away),
        "mae_home": goal_mae(y_home, lambda_home),
        "mae_away": goal_mae(y_away, lambda_away),
    }


def evaluate_wdl(
    y_home: np.ndarray,
    y_away: np.ndarray,
    p_home_win: np.ndarray,
    p_draw: np.ndarray,
    p_away_win: np.ndarray,
) -> dict[str, float]:
    return {
        "wdl_log_loss": wdl_log_loss(y_home, y_away, p_home_win, p_draw, p_away_win),
        "wdl_brier": brier_score_wdl(y_home, y_away, p_home_win, p_draw, p_away_win),
        "wdl_ece": expected_calibration_error_wdl(
            y_home, y_away, p_home_win, p_draw, p_away_win
        ),
    }


def evaluate_full(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    *,
    rho: float = 0.0,
    max_goals: int = 10,
) -> dict[str, float]:
    metrics = evaluate_goals(y_home, y_away, lambda_home, lambda_away)
    p_h, p_d, p_a = wdl_from_lambdas(
        lambda_home, lambda_away, rho=rho, max_goals=max_goals
    )
    metrics.update(evaluate_wdl(y_home, y_away, p_h, p_d, p_a))
    return metrics

"""Elo rating computation."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expected score for team A vs team B (400-point scale)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_rating(rating: float, expected: float, actual: float, k_factor: float) -> float:
    """Apply one Elo update. actual in {0, 0.5, 1}."""
    return rating + k_factor * (actual - expected)


def match_result_points(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5

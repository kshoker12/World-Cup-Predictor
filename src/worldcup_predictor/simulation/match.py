"""Single-match simulation with knockout tie resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldcup_predictor.simulation.score_grid import build_score_grid


@dataclass(frozen=True)
class MatchOutcome:
    home_goals: int
    away_goals: int
    winner: str
    loser: str
    decided_by: str  # regulation | extra_time | penalties


def penalty_home_win_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _sample_from_grid(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = grid.ravel()
    idx = rng.choice(flat.size, p=flat)
    return divmod(int(idx), grid.shape[1])


def simulate_match(
    home: str,
    away: str,
    lambda_home: float,
    lambda_away: float,
    elo_diff: float,
    *,
    knockout: bool = False,
    max_goals: int = 10,
    rho: float = 0.0,
    rng: np.random.Generator,
) -> MatchOutcome:
    grid = build_score_grid(
        lambda_home, lambda_away, max_goals=max_goals, rho=rho
    )
    home_goals, away_goals = _sample_from_grid(grid, rng)
    decided_by = "regulation"

    if knockout and home_goals == away_goals:
        et_lh = lambda_home / 3.0
        et_la = lambda_away / 3.0
        et_grid = build_score_grid(et_lh, et_la, max_goals=max_goals, rho=rho)
        et_h, et_a = _sample_from_grid(et_grid, rng)
        home_goals += et_h
        away_goals += et_a
        decided_by = "extra_time"

        if home_goals == away_goals:
            p_home = penalty_home_win_prob(elo_diff)
            if rng.random() < p_home:
                home_goals += 1
            else:
                away_goals += 1
            decided_by = "penalties"

    if home_goals > away_goals:
        winner, loser = home, away
    else:
        winner, loser = away, home

    return MatchOutcome(
        home_goals=home_goals,
        away_goals=away_goals,
        winner=winner,
        loser=loser,
        decided_by=decided_by,
    )

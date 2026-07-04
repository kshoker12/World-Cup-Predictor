import numpy as np

from worldcup_predictor.simulation.score_grid import build_score_grid, wdl_probabilities


def test_grid_sums_to_one():
    grid = build_score_grid(1.3, 1.1, max_goals=10, rho=0.0)
    assert abs(grid.sum() - 1.0) < 1e-9


def test_wdl_sums_to_one():
    grid = build_score_grid(1.5, 1.2, max_goals=10)
    p_h, p_d, p_a = wdl_probabilities(grid)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-9


def test_rho_zero_independent():
    grid = build_score_grid(1.0, 1.0, max_goals=5, rho=0.0)
    p_h, p_d, p_a = wdl_probabilities(grid)
    assert p_d > 0.2

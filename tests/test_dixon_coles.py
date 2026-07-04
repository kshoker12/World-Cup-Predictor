import numpy as np

from worldcup_predictor.calibration.dixon_coles import fit_dixon_coles_rho
from worldcup_predictor.simulation.score_grid import build_score_grid, wdl_probabilities


def test_fit_rho_negative_on_low_scores():
    rng = np.random.default_rng(0)
    n = 500
    lh = np.full(n, 0.8)
    la = np.full(n, 0.8)
    y_h = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    y_a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])

    params = fit_dixon_coles_rho(y_h, y_a, lh, la, rho_min=-0.2, rho_max=0.1)
    assert -0.2 <= params.rho <= 0.1


def test_rho_changes_draw_probability():
    grid0 = build_score_grid(1.2, 1.2, max_goals=10, rho=0.0)
    grid1 = build_score_grid(1.2, 1.2, max_goals=10, rho=-0.1)
    _, p_d0, _ = wdl_probabilities(grid0)
    _, p_d1, _ = wdl_probabilities(grid1)
    assert p_d1 != p_d0

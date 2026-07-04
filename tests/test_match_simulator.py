import numpy as np

from worldcup_predictor.simulation.match import penalty_home_win_prob, simulate_match


def test_penalty_formula():
    p = penalty_home_win_prob(0.0)
    assert abs(p - 0.5) < 1e-9
    assert penalty_home_win_prob(400.0) > 0.5


def test_knockout_never_draw():
    rng = np.random.default_rng(0)
    for _ in range(50):
        outcome = simulate_match(
            "A",
            "B",
            0.01,
            0.01,
            0.0,
            knockout=True,
            rng=rng,
        )
        assert outcome.home_goals != outcome.away_goals


def test_regulation_can_draw():
    rng = np.random.default_rng(1)
    draws = 0
    for _ in range(200):
        outcome = simulate_match(
            "A", "B", 0.3, 0.3, 0.0, knockout=False, rng=rng
        )
        if outcome.home_goals == outcome.away_goals:
            draws += 1
    assert draws > 0

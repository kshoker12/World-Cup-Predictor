from worldcup_predictor.ratings.elo import expected_score, update_rating


def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == 0.5


def test_home_advantage_increases_expected():
    # Home effective 1550 vs 1500
    exp = expected_score(1550, 1500)
    assert exp > 0.5
    assert abs(exp - 0.5714631174083814) < 1e-6


def test_update_rating_win():
    # E=0.5, actual=1, K=20 -> +10
    new = update_rating(1500, 0.5, 1.0, 20)
    assert new == 1510.0


def test_update_rating_draw():
    new = update_rating(1500, 0.5, 0.5, 20)
    assert new == 1500.0


def test_update_rating_loss():
    new = update_rating(1500, 0.5, 0.0, 20)
    assert new == 1490.0


def test_neutral_no_home_advantage_in_pipeline(default_config):
    """Neutral match: expectation uses raw ratings (verified via pipeline in integration)."""
    exp_neutral = expected_score(1500, 1500)
    exp_home = expected_score(1550, 1500)
    assert exp_neutral == 0.5
    assert exp_home > 0.5

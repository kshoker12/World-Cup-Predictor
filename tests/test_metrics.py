import numpy as np

from worldcup_predictor.models.metrics import goal_rmse, poisson_deviance


def test_poisson_deviance_perfect():
    y = np.array([2.0, 1.0, 3.0])
    assert poisson_deviance(y, y) == 0.0


def test_goal_rmse():
    y = np.array([2.0, 0.0])
    pred = np.array([2.0, 1.0])
    assert abs(goal_rmse(y, pred) - np.sqrt(0.5)) < 1e-9

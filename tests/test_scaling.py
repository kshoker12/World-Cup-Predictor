import numpy as np

from worldcup_predictor.calibration.scaling import ScalingParams, fit_scaling
from worldcup_predictor.models.metrics import poisson_deviance


def test_scaling_improves_deviance():
    y = np.array([2.0, 1.0, 0.0, 3.0, 1.0])
    lam = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    raw_dev = poisson_deviance(y, lam)
    params = fit_scaling(y, y, lam, lam, bounds=(0.5, 2.0))
    lh, _ = params.apply(lam, lam)
    cal_dev = poisson_deviance(y, lh)
    assert params.s_home >= 0.5
    assert cal_dev <= raw_dev


def test_scaling_apply():
    params = ScalingParams(s_home=1.2, s_away=0.8)
    lh, la = params.apply(np.array([1.0]), np.array([2.0]))
    assert lh[0] == 1.2
    assert la[0] == 1.6

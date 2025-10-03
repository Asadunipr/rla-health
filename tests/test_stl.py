import numpy as np

from rla.robust.stl import stl_detrend


def test_stl_reduces_trend_rmse():
    rng = np.random.default_rng(0)
    T = 512
    t = np.arange(T)
    trend = 0.01 * t  # slow drift
    noise = rng.normal(scale=0.1, size=T)
    x = trend + noise
    x_dt = stl_detrend(x, period=None, robust=True, axis=0)
    # detrending should reduce correlation with t (proxy for drift)
    corr_orig = np.corrcoef(x, t)[0, 1]
    corr_dt = np.corrcoef(x_dt, t)[0, 1]
    assert abs(corr_dt) < abs(corr_orig)

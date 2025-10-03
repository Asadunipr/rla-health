import numpy as np

from rla.robust.hampel import hampel_filter


def test_hampel_1d_repairs_spike():
    x = np.zeros(101, dtype=float)
    x[50] = 100.0
    y = hampel_filter(x, window=9, k=3.0, axis=0)
    # For a flat baseline, the local median is exactly 0
    assert abs(y[50]) < 1e-9  # numerical noise tolerance


def test_hampel_multichannel_time_major():
    rng = np.random.default_rng(0)
    T, D = 200, 2
    x = rng.normal(scale=0.1, size=(T, D))
    x[80, 0] = 50.0  # spike in channel 0

    y = hampel_filter(x, window=9, k=3.0, axis=0)

    # Replaced value should be close to the local neighborhood median.
    # With noise σ≈0.1, the local median is typically within a few hundredths of 0.
    assert abs(y[80, 0]) < 2e-2

    # Other channel should be effectively untouched at that index
    assert abs(y[80, 1] - x[80, 1]) < 1e-12

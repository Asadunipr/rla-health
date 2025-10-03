import numpy as np

from rla.robust.mad import mad_scale


def test_mad_basic_1d():
    x = np.array([0, 0, 0, 10])  # outlier
    s = mad_scale(x, axis=0)
    assert np.all(s > 0)


def test_mad_is_robust_to_single_spike():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 3))

    # Baseline MAD per channel over time (axis=0 for time-major (T, D))
    s_clean = mad_scale(x, axis=0)  # shape (1, 3)

    # Inject one huge spike in channel 1
    x_spike = x.copy()
    x_spike[100, 1] += 50
    s_spike = mad_scale(x_spike, axis=0)

    ratio = s_spike / s_clean  # expected ~1 for a single outlier (robustness)
    assert ratio.shape == (1, 3)

    # Channel with spike should not inflate MAD much (<= +20% is a reasonable tolerance)
    assert ratio[0, 1] < 1.2

    # Other channels should be essentially unchanged (within ±10%)
    assert abs(ratio[0, 0] - 1.0) < 0.1
    assert abs(ratio[0, 2] - 1.0) < 0.1

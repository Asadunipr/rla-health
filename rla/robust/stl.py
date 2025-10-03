from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL


def stl_detrend(
    x: np.ndarray,
    period: int | None = None,
    robust: bool = True,
    axis: int = 0,
    return_trend: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Apply STL decomposition to remove low-frequency trend (and optional seasonality).

    Parameters
    ----------
    x : np.ndarray
        Time series, supports (T,), (T, D), (B, T, D) etc.
    period : int | None
        Seasonality period. For ECG baseline wander removal you can set None (trend only)
        or a large period relative to sampling. For wearable periodic channels, set period.
    robust : bool
        Use robust STL (less sensitive to outliers).
    axis : int
        Time axis.
    return_trend : bool
        If True, return (detrended, trend).

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Detrended signal (same shape as x), and optional trend.
    """
    x = np.asarray(x, dtype=float)
    x_moved = np.moveaxis(x, axis, 0)  # time-first
    rest_shape = x_moved.shape[1:]
    T = x_moved.shape[0]
    C = int(np.prod(rest_shape)) if rest_shape else 1
    x2 = x_moved.reshape(T, C)

    detrended = np.empty_like(x2)
    trend = np.empty_like(x2)

    for c in range(C):
        series = x2[:, c]
        # statsmodels expects 1D array
        if period is None:
            # trick: set a large period so seasonal ~0, trend picks low-freq
            stl = STL(
                series, robust=robust, seasonal=7, trend=13, period=7
            )  # mild defaults
        else:
            stl = STL(series, period=period, robust=robust)
        res = stl.fit()
        trend[:, c] = res.trend
        detrended[:, c] = series - res.trend  # keep residual + seasonal

    detrended_full = detrended.reshape((T, *rest_shape))
    trend_full = trend.reshape((T, *rest_shape))

    detrended_out = np.moveaxis(detrended_full, 0, axis)
    trend_out = np.moveaxis(trend_full, 0, axis)
    return (detrended_out, trend_out) if return_trend else detrended_out

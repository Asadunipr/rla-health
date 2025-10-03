from __future__ import annotations

import numpy as np


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Fast-ish rolling median for 1D using padding + sliding windows.
    For multichannel, apply per-channel in Python loop (n_channels is small).
    """
    assert window % 2 == 1, "Hampel window must be odd"
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    # build strided view
    shape = (arr.shape[0], window)
    strides = (padded.strides[0], padded.strides[0])
    as_strided = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.median(as_strided, axis=1)


def hampel_filter_1d(x: np.ndarray, window: int = 17, k: float = 3.0) -> np.ndarray:
    """
    Hampel spike repair for a 1D array. Replaces outliers by local median.
    """
    med = _rolling_median(x, window)
    # local MAD
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    shape = (x.shape[0], window)
    strides = (padded.strides[0], padded.strides[0])
    as_strided = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    local_mad = 1.4826 * np.median(np.abs(as_strided - med[:, None]), axis=1)
    local_mad = np.maximum(local_mad, 1e-9)

    y = x.copy()
    mask = np.abs(x - med) > (k * local_mad)
    y[mask] = med[mask]
    return y


def hampel_filter(
    x: np.ndarray, window: int = 17, k: float = 3.0, axis: int = 0
) -> np.ndarray:
    """
    Multichannel Hampel filter.

    If x is (T,) -> returns (T,)
    If x is (T, D) with axis=0 -> applies Hampel independently on each column D.
    If x is (B, T, D) and axis=1 -> applies per (B, D) along time.

    Parameters
    ----------
    x : np.ndarray
    window : int
    k : float
    axis : int
        Axis representing time dimension to filter along.

    Returns
    -------
    np.ndarray
        Filtered array with spikes replaced by local medians.
    """
    x = np.asarray(x)
    x_moved = np.moveaxis(x, axis, 0)  # time-first
    rest_shape = x_moved.shape[1:]
    y_moved = np.empty_like(x_moved)

    # collapse other dims to a single channel dim: (T, C)
    T = x_moved.shape[0]
    C = int(np.prod(rest_shape)) if rest_shape else 1
    x2 = x_moved.reshape(T, C)
    y2 = np.empty_like(x2)

    for c in range(C):
        y2[:, c] = hampel_filter_1d(x2[:, c], window=window, k=k)

    y_moved = y2.reshape((T, *rest_shape))
    y = np.moveaxis(y_moved, 0, axis)
    return y

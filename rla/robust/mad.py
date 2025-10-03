from __future__ import annotations

import numpy as np


def mad_scale(x: np.ndarray, axis: int | None = 0, eps: float = 1e-9) -> np.ndarray:
    """
    Compute Gaussian-consistent Median Absolute Deviation (MAD) scale.

    Parameters
    ----------
    x : np.ndarray
        Input array. Supports shapes like (T,), (T, D), (B, T, D).
    axis : int | None
        Axis along which to compute MAD. If None, MAD over the flattened array.
        Common choices:
          - axis=0 for time-major (T, D)
          - axis=1 for (B, T, D) to scale per-sample and per-channel across time
    eps : float
        Minimum scale to avoid divide-by-zero.

    Returns
    -------
    np.ndarray
        Scale array broadcastable to x along the chosen axis.
    """
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    scale = 1.4826 * mad  # Gaussian consistency factor
    return np.maximum(scale, eps)

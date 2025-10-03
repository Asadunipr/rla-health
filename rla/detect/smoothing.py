from __future__ import annotations

import numpy as np


def rolling_median(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    z = np.pad(x, (pad, pad), mode="edge")
    shape = (x.shape[0], win)
    strides = (z.strides[0], z.strides[0])
    v = np.lib.stride_tricks.as_strided(z, shape=shape, strides=strides)
    return np.median(v, axis=1)

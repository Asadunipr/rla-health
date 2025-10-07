from __future__ import annotations

import numpy as np


def robust_mad_threshold(scores: np.ndarray, kappa: float = 2.5) -> float:
    """
    τ = median(scores) + κ * MAD(scores) with Gaussian consistency (1.4826 factor).
    Unsupervised; robust to a small fraction of outliers.
    """
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med))
    mad = max(mad, 1e-12)
    return float(med + kappa * mad)


def tau_from_quantile(scores: np.ndarray, q: float = 0.995) -> float:
    """
    τ as the q-quantile of the validation-normal scores.
    This directly controls expected false positive rate under stationarity.
    """
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(scores, q))

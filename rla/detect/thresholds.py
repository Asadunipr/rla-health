from __future__ import annotations

import numpy as np


def robust_mad_threshold(scores: np.ndarray, kappa: float = 2.5) -> float:
    """τ = median(scores) + κ * MAD(scores) with Gaussian consistency."""
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med))
    mad = max(mad, 1e-12)
    return float(med + kappa * mad)

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) == 1:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, tau: float) -> float:
    y_hat = (scores >= tau).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, tau_at_best) for reference (not used to set τ; we set τ on val)."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    idx = int(np.nanargmax(f1))
    return float(f1[idx]), float(t[max(idx - 1, 0)])  # thresholds array shorter by 1

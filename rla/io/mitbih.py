from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb

# AHA heartbeat annotation symbols considered "normal"
NORMAL_BEATS = {"N", "L", "R", "e", "j"}


@dataclass
class RecordData:
    record_id: str
    fs: int
    signal: np.ndarray  # shape (T, C)
    ann_sample: np.ndarray  # sample indices of annotations
    ann_symbol: List[str]  # corresponding symbols


def load_record(
    record_id: str,
    db: str = "mitdb",
    source: str = "remote",
    base_dir: str | None = None,
) -> RecordData:
    """
    Load a MIT-BIH record either from PhysioNet (remote) or locally.

    source:
      - "remote": stream via PhysioNet (requires internet) using pn_dir=db
      - "local": read from base_dir/<db>/<record_id>.* (requires files on disk)
    """
    if source == "remote":
        rec = wfdb.rdrecord(record_id, pn_dir=db)
        ann = wfdb.rdann(record_id, "atr", pn_dir=db)
    else:
        assert base_dir is not None, "base_dir must be provided for source='local'"
        p = Path(base_dir) / db / record_id
        rec = wfdb.rdrecord(str(p))
        ann = wfdb.rdann(str(p), "atr")

    sig = rec.p_signal.astype(np.float32)  # (T, C)
    fs = int(rec.fs)
    return RecordData(
        record_id=record_id,
        fs=fs,
        signal=sig,
        ann_sample=ann.sample.astype(int),
        ann_symbol=list(ann.symbol),
    )


def downsample(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Simple integer-ratio decimation (e.g., 360 -> 180)."""
    if target_fs == orig_fs:
        return sig
    assert orig_fs % target_fs == 0, "Use integer decimation for now."
    ratio = orig_fs // target_fs
    return sig[::ratio]


def window_and_label(
    sig: np.ndarray,
    anns: Tuple[np.ndarray, List[str]],
    fs: int,
    window: int,
    stride: int,
    anomaly_if_any_non_normal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide windows over (T, C) -> (N, window, C); label window as 1 if any arrhythmia occurs inside.
    """
    T, C = sig.shape
    ann_idx, ann_sym = anns
    arrhythmic = np.zeros(T, dtype=bool)
    for s, sym in zip(ann_idx, ann_sym):
        if 0 <= s < T and (anomaly_if_any_non_normal and sym not in NORMAL_BEATS):
            arrhythmic[s] = True

    windows, labels = [], []
    for start in range(0, max(T - window + 1, 0), stride):
        end = start + window
        w = sig[start:end, :]
        if w.shape[0] < window:
            break
        windows.append(w)
        labels.append(bool(arrhythmic[start:end].any()))

    if not windows:
        return np.empty((0, window, C), dtype=sig.dtype), np.empty((0,), dtype=np.int64)
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb

from rla.io.mitbih import downsample, load_record, window_and_label


def get_record_list() -> List[str]:
    return wfdb.get_record_list("mitdb")


def split_records(
    records: List[str], val_frac: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    recs = records.copy()
    rng.shuffle(recs)
    n_val = max(1, int(len(recs) * val_frac))
    return recs[n_val:], recs[:n_val]  # train, val


def main(out_dir: str, target_fs: int, window: int, stride: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = get_record_list()
    train_recs, val_recs = split_records(records, val_frac=0.2, seed=42)

    def process_split(rec_ids: List[str]):
        X_list, y_list, recid_list = [], [], []
        for rid in rec_ids:
            # REMOTE read from PhysioNet
            rd = load_record(rid, db="mitdb", source="remote", base_dir=None)
            sig = downsample(rd.signal, rd.fs, target_fs)
            sig = sig[:, : min(sig.shape[1], 2)]
            X, y = window_and_label(
                sig, (rd.ann_sample, rd.ann_symbol), target_fs, window, stride
            )
            if X.shape[0] == 0:
                continue
            X_list.append(X)
            y_list.append(y)
            recid_list.extend([rid] * X.shape[0])
        if not X_list:
            return (
                np.empty((0, window, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )
        return (
            np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            recid_list,
        )

    print(
        f"Processing {len(train_recs)} train records and {len(val_recs)} val records…"
    )
    X_tr_all, y_tr_all, _ = process_split(train_recs)
    X_va_all, y_va_all, _ = process_split(val_recs)

    X_train = X_tr_all[y_tr_all == 0]
    X_val = X_va_all[y_va_all == 0]

    X_test = (
        np.concatenate([X_tr_all, X_va_all], axis=0)
        if X_tr_all.size and X_va_all.size
        else (X_tr_all if X_va_all.size == 0 else X_va_all)
    )
    y_test = (
        np.concatenate([y_tr_all, y_va_all], axis=0)
        if y_tr_all.size and y_va_all.size
        else (y_tr_all if y_va_all.size == 0 else y_va_all)
    )
    rec_ids = (
        ["train"] * len(y_tr_all) + ["val"] * len(y_va_all)
        if (len(y_tr_all) and len(y_va_all))
        else (["train"] * len(y_tr_all) if len(y_tr_all) else ["val"] * len(y_va_all))
    )

    np.savez_compressed(out / "train_normals.npz", X=X_train)
    np.savez_compressed(out / "val_normals.npz", X=X_val)
    np.savez_compressed(
        out / "test_all.npz", X=X_test, y=y_test, rec_ids=np.array(rec_ids)
    )

    print(f"Saved: {out/'train_normals.npz'}  ({X_train.shape})")
    print(f"Saved: {out/'val_normals.npz'}    ({X_val.shape})")
    print(
        f"Saved: {out/'test_all.npz'}       ({X_test.shape}, labels={y_test.sum()}/{len(y_test)} anomalies)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=int, default=180)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()
    main(args.out, args.fs, args.window, args.stride)

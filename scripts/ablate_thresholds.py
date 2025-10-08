import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from rla.detect.smoothing import rolling_median
from rla.detect.thresholds import robust_mad_threshold, tau_from_quantile
from rla.eval.metrics import auprc, auroc, f1_at_threshold
from rla.io.npzloader import NPZWindows
from rla.models.tiny_tcn import TinyTCN
from rla.robust.hampel import hampel_filter

torch.set_num_threads(4)


def robust_standardize_with_stats(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values
    scale = (1.4826 * mad).clamp_min(eps)
    return (x - med) / scale, med, scale


def unstandardize(
    y_std: torch.Tensor, med: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return y_std * scale + med


def hampel_torch(x: torch.Tensor, window: int = 9, k: float = 3.0) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    for b in range(x_np.shape[0]):
        for c in range(x_np.shape[2]):
            x_np[b, :, c] = hampel_filter(x_np[b, :, c], window=window, k=k, axis=0)
    return torch.from_numpy(x_np).to(x.device)


def compute_scores_raw(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str,
    use_hampel: bool,
    hw: int,
    hk: float,
) -> np.ndarray:
    model.eval()
    scores = np.zeros((X.shape[0],), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i : i + bs]).to(device)
            if use_hampel:
                xb = hampel_torch(xb, window=hw, k=hk)
            xb_std, med, scale = robust_standardize_with_stats(xb)
            yb_std = model(xb_std)
            yb_raw = unstandardize(yb_std, med, scale)
            e = torch.abs(yb_raw - xb).mean(dim=(1, 2))
            scores[i : i + bs] = e.cpu().numpy()
    return scores


def parse_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--kappas", type=str, default="0.9,1.0,1.1,1.2")
    ap.add_argument("--quantiles", type=str, default="")
    ap.add_argument("--smooth_list", type=str, default="7,9")
    ap.add_argument(
        "--out", type=str, default="experiments/results/mitbih/ablate_thresholds.csv"
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    data_dir = Path(cfg.get("data_dir", "data/mitbih"))
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Data
    va_ds = NPZWindows(data_dir, split="val")
    te_npz = np.load(data_dir / "test_all.npz")
    X_test, y_test = te_npz["X"].astype("float32"), te_npz["y"].astype(int)

    # Model
    d_in = va_ds.X.shape[-1]
    channels = int(cfg["model"]["tcn"]["channels"])
    depth = int(cfg["model"]["tcn"].get("depth", 6))
    model = TinyTCN(d_in=d_in, d_out=d_in, channels=channels, depth=depth)
    device = "cpu"
    model.to(device)

    best_path = Path(cfg.get("out_dir", "experiments/results/mitbih")) / "model_best.pt"
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Preproc flags (must match training/eval)
    use_hampel = bool(cfg.get("preprocess", {}).get("hampel", False))
    hw = int(cfg.get("preprocess", {}).get("hampel_win", 9))
    hk = float(cfg.get("preprocess", {}).get("hampel_k", 3.0))

    # Scores in RAW space (once)
    print("Scoring validation (raw)…")
    val_scores = compute_scores_raw(model, va_ds.X, device, use_hampel, hw, hk)
    print("Scoring test (raw)…")
    test_scores_raw = compute_scores_raw(model, X_test, device, use_hampel, hw, hk)

    kappas = parse_floats(args.kappas)
    quantiles = parse_floats(args.quantiles)
    smooth_list = parse_ints(args.smooth_list)

    rows: list[dict] = []
    for sm in smooth_list:
        test_scores = (
            rolling_median(test_scores_raw, win=sm) if sm > 1 else test_scores_raw
        )
        for k in kappas:
            tau = robust_mad_threshold(val_scores, kappa=k)
            rows.append(
                {
                    "method": "mad",
                    "param": k,
                    "tau": tau,
                    "smooth": sm,
                    "auroc": auroc(y_test, test_scores),
                    "auprc": auprc(y_test, test_scores),
                    "f1_at_tau": f1_at_threshold(y_test, test_scores, tau),
                }
            )
        for q in quantiles:
            tau = tau_from_quantile(val_scores, q=q)
            rows.append(
                {
                    "method": "quantile",
                    "param": q,
                    "tau": tau,
                    "smooth": sm,
                    "auroc": auroc(y_test, test_scores),
                    "auprc": auprc(y_test, test_scores),
                    "f1_at_tau": f1_at_threshold(y_test, test_scores, tau),
                }
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

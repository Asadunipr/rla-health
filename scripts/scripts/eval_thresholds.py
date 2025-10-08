# @"
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
    device: str = "cpu",
    use_hampel: bool = False,
    hampel_win: int = 9,
    hampel_k: float = 3.0,
) -> np.ndarray:
    model.eval()
    scores = np.zeros((X.shape[0],), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            x_np = X[i : i + bs]
            xb = torch.from_numpy(x_np).to(device)
            if use_hampel:
                xb = hampel_torch(xb, window=hampel_win, k=hampel_k)
            xb_std, med, scale = robust_standardize_with_stats(xb)
            yb_std = model(xb_std)
            yb_raw = unstandardize(yb_std, med, scale)
            e = torch.abs(yb_raw - xb).mean(dim=(1, 2))
            scores[i : i + bs] = e.cpu().numpy()
    return scores


def parse_list_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--kappas", type=str, default="0.8,1.0,1.2,1.5")
    ap.add_argument("--quantiles", type=str, default="0.99,0.995,0.997")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument(
        "--out", type=str, default="experiments/results/mitbih/threshold_sweep.csv"
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    data_dir = Path(cfg.get("data_dir", "data/mitbih"))
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Datasets
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

    # Load best checkpoint
    best_path = Path(cfg.get("out_dir", "experiments/results/mitbih")) / "model_best.pt"
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Hampel settings (must match training/eval)
    use_hampel = bool(cfg.get("preprocess", {}).get("hampel", False))
    hampel_win = int(cfg.get("preprocess", {}).get("hampel_win", 9))
    hampel_k = float(cfg.get("preprocess", {}).get("hampel_k", 3.0))

    # Scores
    print("Computing validation raw scores…")
    val_scores = compute_scores_raw(
        model,
        va_ds.X,
        device=device,
        use_hampel=use_hampel,
        hampel_win=hampel_win,
        hampel_k=hampel_k,
    )
    print("Computing test raw scores…")
    test_scores = compute_scores_raw(
        model,
        X_test,
        device=device,
        use_hampel=use_hampel,
        hampel_win=hampel_win,
        hampel_k=hampel_k,
    )
    if args.smooth > 1:
        test_scores = rolling_median(test_scores, win=args.smooth)

    kappas = parse_list_floats(args.kappas)
    quantiles = parse_list_floats(args.quantiles)

    rows: list[dict] = []
    for k in kappas:
        tau = robust_mad_threshold(val_scores, kappa=k)
        rows.append(
            {
                "method": "mad",
                "param": k,
                "tau": tau,
                "auroc": auroc(y_test, test_scores),
                "auprc": auprc(y_test, test_scores),
                "f1_at_tau": f1_at_threshold(y_test, test_scores, tau),
                "smooth": args.smooth,
            }
        )
    for q in quantiles:
        tau = tau_from_quantile(val_scores, q=q)
        rows.append(
            {
                "method": "quantile",
                "param": q,
                "tau": tau,
                "auroc": auroc(y_test, test_scores),
                "auprc": auprc(y_test, test_scores),
                "f1_at_tau": f1_at_threshold(y_test, test_scores, tau),
                "smooth": args.smooth,
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep to: {out_csv}")


if __name__ == "__main__":
    main()
# "@ | Set-Content -Encoding UTF8 scripts\eval_thresholds.py

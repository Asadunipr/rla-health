import argparse
import json
from pathlib import Path
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from rla.detect.smoothing import rolling_median
from rla.detect.thresholds import robust_mad_threshold
from rla.eval.metrics import auprc, auroc, f1_at_threshold
from rla.io.npzloader import NPZWindows
from rla.losses.huber import huber_loss
from rla.losses.trimmed_mse import trimmed_mse
from rla.models.tiny_tcn import TinyTCN

# Keep CPU thread usage sane on Windows; tune for your CPU
torch.set_num_threads(4)


# --------------------------
# Robust standardization
# --------------------------
def robust_standardize_with_stats(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust per-window, per-channel standardization with returned stats:
      x' = (x - med_t) / (1.4826 * MAD_t + eps)
    Shapes: x (B, L, C) -> (x_std, med, scale), all broadcastable.
    """
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values
    scale = (1.4826 * mad).clamp_min(eps)
    return (x - med) / scale, med, scale


def unstandardize(
    y_std: torch.Tensor, med: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Invert robust standardization: y = y_std * scale + med."""
    return y_std * scale + med


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def mean_abs_error_batched_raw(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    bs: int = 256,
) -> float:
    """
    Mean |x_raw - y_raw| computed in mini-batches.
    Train may use standardized inputs, but we evaluate in RAW space:
      1) standardize x -> (x_std, med, scale)
      2) y_std = model(x_std)
      3) y_raw = unstandardize(y_std, med, scale)
      4) MAE(y_raw, x_raw)
    """
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            x_np = X[i : i + bs]
            xb = torch.from_numpy(x_np).to(device)  # (B, L, C)
            xb_std, med, scale = robust_standardize_with_stats(xb)
            yb_std = model(xb_std)
            yb_raw = unstandardize(yb_std, med, scale)
            e = torch.abs(yb_raw - xb).mean(dim=(1, 2))  # per-window MAE in RAW space
            tot += float(e.sum().cpu().item())
            n += e.shape[0]
    return tot / max(n, 1)


def compute_scores_raw(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Reconstruction residual scores per window, **in RAW space**:
      score = mean(|x_raw - y_raw|)
    """
    model.eval()
    scores = np.zeros((X.shape[0],), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            x_np = X[i : i + bs]
            xb = torch.from_numpy(x_np).to(device)
            xb_std, med, scale = robust_standardize_with_stats(xb)
            yb_std = model(xb_std)
            yb_raw = unstandardize(yb_std, med, scale)
            e = torch.abs(yb_raw - xb).mean(dim=(1, 2))
            scores[i : i + bs] = e.cpu().numpy()
    return scores


# --------------------------
# Training
# --------------------------
def main(cfg_path: str) -> None:
    # ---- Config & paths ----
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    data_dir = Path(cfg.get("data_dir", "data/mitbih"))
    out_dir = Path(cfg.get("out_dir", "experiments/results/mitbih"))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))

    # ---- Data ----
    tr_ds = NPZWindows(data_dir, split="train")
    va_ds = NPZWindows(data_dir, split="val")
    te_npz = np.load(data_dir / "test_all.npz")
    X_test, y_test = te_npz["X"].astype("float32"), te_npz["y"].astype(int)

    # ---- Model ----
    d_in = tr_ds.X.shape[-1]
    channels = int(cfg["model"]["tcn"]["channels"])
    depth = int(cfg["model"]["tcn"].get("depth", 6))
    model = TinyTCN(d_in=d_in, d_out=d_in, channels=channels, depth=depth)
    device = "cpu"  # set "cuda" if available/desired
    model.to(device)

    # ---- Optimizer & training params ----
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    bs = int(cfg["training"]["batch_size"])
    val_bs = int(cfg["training"].get("val_batch_size", min(bs, 256)))
    epochs = int(cfg["training"]["epochs"])
    loss_name = cfg["training"]["loss"]
    huber_delta = float(cfg.get("huber_delta", 1.0))
    trimmed_p = float(cfg.get("trimmed_p", 0.05))

    # Curriculum: sort train windows by per-window std (proxy for hardness)
    stds = tr_ds.X.std(axis=(1, 2))
    order = np.argsort(stds)  # easiest -> hardest
    X_train_sorted = tr_ds.X[order]
    start_frac = float(cfg["training"].get("curriculum_start", 0.5))
    end_frac = 1.0

    def make_loader(X: np.ndarray) -> DataLoader:
        tensor = torch.from_numpy(X)
        ds = torch.utils.data.TensorDataset(tensor)
        return DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0
        )

    # Early stopping
    patience = int(cfg["training"].get("early_stop_patience", 15))
    no_improve = 0
    best_val = float("inf")
    best_path = out_dir / "model_best.pt"

    # ---- Training loop ----
    for epoch in range(1, epochs + 1):
        # curriculum slice
        frac = start_frac + (end_frac - start_frac) * (epoch - 1) / max(epochs - 1, 1)
        n_use = max(int(frac * X_train_sorted.shape[0]), bs)
        X_tr_use = X_train_sorted[:n_use]
        tr_loader = make_loader(X_tr_use)

        model.train()
        running = 0.0
        n_steps = 0
        t0 = time.time()

        for (xb,) in tr_loader:
            xb = xb.to(device)
            # Train on standardized inputs; reconstruct standardized target
            xb_std, med, scale = robust_standardize_with_stats(xb)
            yb_std = model(xb_std)
            target = xb_std

            if loss_name == "huber":
                loss = huber_loss(yb_std, target, delta=huber_delta)
            elif loss_name == "trimmed":
                loss = trimmed_mse(yb_std, target, trim_p=trimmed_p)
            else:
                loss = torch.mean((yb_std - target) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_steps += 1

        # Validation **in RAW space** (memory-safe)
        vloss = mean_abs_error_batched_raw(
            model,
            va_ds.X,
            device=device,
            bs=val_bs,
        )

        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{epochs}] train_loss={running/max(n_steps,1):.4f}  val_l1={vloss:.4f}  time={dt:.1f}s"
        )

        # checkpoint + early stopping
        improved = vloss < (best_val - 1e-5)
        if improved:
            best_val = vloss
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    print(f"Saved best model to: {best_path}")

    # ---- Evaluation on test ----
    # reload best
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # scores on val normals (RAW) -> τ (no leakage)
    val_scores = compute_scores_raw(model, va_ds.X, device=device)
    kappa = float(cfg["decision"]["kappa"])
    tau = robust_mad_threshold(val_scores, kappa=kappa)
    print(f"Robust threshold τ (kappa={kappa}) = {tau:.6f}")

    # scores on test (RAW), optionally smoothed
    test_scores = compute_scores_raw(model, X_test, device=device)
    smooth_len = int(cfg["decision"].get("smooth_median_len", 5))
    if smooth_len > 1:
        test_scores = rolling_median(test_scores, win=smooth_len)

    # metrics
    metrics = {
        "auroc": auroc(y_test, test_scores),
        "auprc": auprc(y_test, test_scores),
        "f1_at_tau": f1_at_threshold(y_test, test_scores, tau),
        "tau": float(tau),
        "val_score_median": float(np.median(val_scores)),
        "val_score_mad": float(
            1.4826 * np.median(np.abs(val_scores - np.median(val_scores)))
        ),
        "best_val_l1": float(best_val),
    }
    print("Test metrics:", metrics)

    # ---- Save artifacts ----
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config_snapshot.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    np.savez_compressed(out_dir / "scores_test.npz", scores=test_scores, y=y_test)
    print(f"Artifacts saved under: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    main(args.config)

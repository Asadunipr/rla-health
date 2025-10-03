import argparse
import json
from pathlib import Path
import time

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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_scores(
    model: torch.nn.Module, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Reconstruction residual scores: mean(|x - x_hat|) per window."""
    model.eval()
    scores = np.zeros((X.shape[0],), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i : i + bs]).to(device)  # (B, L, C)
            yb = model(xb)
            e = torch.abs(yb - xb).mean(dim=(1, 2))  # mean L1 per window
            scores[i : i + bs] = e.cpu().numpy()
    return scores


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # ---- Paths & seed ----
    data_dir = Path(cfg.get("data_dir", "data/mitbih"))
    out_dir = Path(cfg.get("out_dir", "experiments/results/mitbih"))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))

    # ---- Datasets ----
    tr_ds = NPZWindows(data_dir, split="train")
    va_ds = NPZWindows(data_dir, split="val")
    te_npz = np.load(data_dir / "test_all.npz")
    X_test, y_test = te_npz["X"].astype("float32"), te_npz["y"].astype(int)

    # ---- Model ----
    d_in = tr_ds.X.shape[-1]
    model = TinyTCN(d_in=d_in, d_out=d_in, channels=cfg["model"]["tcn"]["channels"])
    device = "cpu"  # edge-first; can set to "cuda" if available
    model.to(device)

    # ---- Optimizer ----
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    bs = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    loss_name = cfg["training"]["loss"]
    huber_delta = float(cfg.get("huber_delta", 1.0))
    trimmed_p = float(cfg.get("trimmed_p", 0.05))

    # ---- Curriculum: sort train windows by Hampel-like roughness (here proxy = per-window std)
    stds = tr_ds.X.std(axis=(1, 2))
    order = np.argsort(stds)  # easiest -> hardest
    X_train_sorted = tr_ds.X[order]
    # schedule: start with 50% easiest, linearly to 100%
    start_frac = float(cfg["training"].get("curriculum_start", 0.5))
    end_frac = 1.0

    def make_loader(X: np.ndarray) -> DataLoader:
        tensor = torch.from_numpy(X)
        ds = torch.utils.data.TensorDataset(tensor)
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    # ---- Training loop ----
    best_val = 1e9
    best_path = out_dir / "model_best.pt"
    for epoch in range(1, epochs + 1):
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
            yb = model(xb)
            if loss_name == "huber":
                loss = huber_loss(yb, xb, delta=huber_delta)
            elif loss_name == "trimmed":
                loss = trimmed_mse(yb, xb, trim_p=trimmed_p)
            else:
                loss = torch.mean((yb - xb) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_steps += 1

        # validation proxy: mean recon error on val normals (lower is better)
        with torch.no_grad():
            model.eval()
            vb = torch.from_numpy(va_ds.X).to(device)
            vy = model(vb)
            vloss = torch.mean(torch.abs(vy - vb)).item()

        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{epochs}] train_loss={running/max(n_steps,1):.4f}  val_l1={vloss:.4f}  time={dt:.1f}s"
        )

        # save best
        if vloss < best_val:
            best_val = vloss
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, best_path)

    print(f"Saved best model to: {best_path}")

    # ---- Evaluation on test set ----
    # reload best
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # scores on val normals -> fit τ
    val_scores = compute_scores(model, va_ds.X, device=device)
    tau = robust_mad_threshold(val_scores, kappa=float(cfg["decision"]["kappa"]))
    print(f"Robust threshold τ = {tau:.6f}")

    # scores on test
    test_scores = compute_scores(model, X_test, device=device)
    # optional smoothing (window=5 by default)
    if int(cfg["decision"].get("smooth_median_len", 5)) > 1:
        test_scores = rolling_median(
            test_scores, win=int(cfg["decision"]["smooth_median_len"])
        )

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

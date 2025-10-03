from __future__ import annotations

import torch


def trimmed_mse(
    y_hat: torch.Tensor, y: torch.Tensor, trim_p: float = 0.05
) -> torch.Tensor:
    """
    Elementwise squared errors, drop the largest trim_p fraction, mean the rest.
    Works on (B, L, C).
    """
    e2 = (y_hat - y) ** 2
    flat = e2.flatten(start_dim=0)
    k = int(flat.numel() * (1 - trim_p))
    if k <= 0:
        return torch.mean(flat)
    topk_vals, _ = torch.topk(flat, k, largest=False)
    return torch.mean(topk_vals)

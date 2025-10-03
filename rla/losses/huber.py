import torch


def huber_loss(y_hat, y, delta: float = 1.0, reduction: str = "mean"):
    e = y_hat - y
    abs_e = e.abs()
    quad = 0.5 * (e**2)
    lin = delta * (abs_e - 0.5 * delta)
    out = torch.where(abs_e <= delta, quad, lin)
    return out.mean() if reduction == "mean" else out

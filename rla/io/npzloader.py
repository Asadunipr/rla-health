from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZWindows(Dataset):
    def __init__(self, path: str | Path, split: Literal["train", "val", "test"]):
        p = Path(path)
        if split == "train":
            data = np.load(p / "train_normals.npz")
            self.X = data["X"]  # (N, L, C)
            self.y = None
        elif split == "val":
            data = np.load(p / "val_normals.npz")
            self.X = data["X"]
            self.y = None
        else:
            data = np.load(p / "test_all.npz")
            self.X = data["X"]
            self.y = data["y"].astype(int)
        self.X = self.X.astype("float32")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.from_numpy(self.X[idx])  # (L, C)
        return x, (
            None if self.y is None else torch.tensor(self.y[idx], dtype=torch.long)
        )

from __future__ import annotations

import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, d=1):
        super().__init__()
        self.dw = nn.Conv1d(
            in_ch,
            in_ch,
            k,
            padding=d * (k - 1) // 2,
            dilation=d,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class TinyTCN(nn.Module):
    def __init__(self, d_in: int, d_out: int, channels: int = 48, depth: int = 6):
        super().__init__()
        layers = [nn.Conv1d(d_in, channels, 1)]
        for i in range(depth):
            layers += [DepthwiseSeparableConv1d(channels, channels, k=3, d=2**i)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, d_out, 1)

    def forward(self, x):  # x: (B, L, d_in)
        x = x.transpose(1, 2)
        h = self.net(x)
        y = self.head(h).transpose(1, 2)
        return y

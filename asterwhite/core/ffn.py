"""
AsterWhite :: core.ffn

SwiGLU (Shazeer 2020), the dense gated-MLP block used both as
HybridBlock's ordinary feed-forward (layers.py) and as the per-expert
feed-forward inside a sparse mixture-of-experts layer (core.moe.
MoEFeedForward -- an "expert" IS just a smaller SwiGLU). Split into its
own module purely so moe.py can import it too without layers.py <-> moe.py
becoming a circular import (layers.py already imports MoEFeedForward
from moe.py to build a HybridBlock's MLP).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

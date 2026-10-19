"""
AsterWhite :: core.norm

RMSNorm (Zhang & Sennrich 2019), the one normalization primitive every
other core module needs -- HybridBlock's two per-block norms (layers.py),
the backbone's final norm (model.py), and GroupedQueryAttention's optional
per-head query/key norm (attention.py, gated by BackboneConfig.qk_norm).
Split into its own module (rather than living in layers.py, which is
where it originated) purely so attention.py can import it too without
layers.py <-> attention.py becoming a circular import (layers.py already
imports GroupedQueryAttention from attention.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The reduction (mean-of-squares + rsqrt) is upcast to fp32 before
        # computing the norm, then cast back to x's own dtype -- standard
        # practice for bf16/fp16 training (mirrors production RMSNorm
        # implementations, e.g. Llama's). bf16 has only 7 mantissa bits;
        # accumulating pow(2).mean() natively in it loses real precision
        # and is a real (if secondary) contributor to early-training
        # instability, not just a theoretical concern -- torch.autocast
        # does NOT do this automatically for hand-written reduction code
        # like this (only for specific built-in ops), so it has to be
        # explicit here.
        input_dtype = x.dtype
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(input_dtype)) * self.weight

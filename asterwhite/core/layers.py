"""AsterWhite :: core.layers -- the hybrid attention/SSM residual block
used to assemble AsterWhiteCore. Normalization (RMSNorm) and the dense
gated-MLP (SwiGLU) live in norm.py/ffn.py respectively -- re-exported
here since this was their original home and other modules still import
them from this path."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .ffn import SwiGLU  # noqa: F401 -- re-exported; this module is SwiGLU's original/canonical import path
from .moe import MoEFeedForward
from .norm import RMSNorm  # noqa: F401 -- re-exported; this module is RMSNorm's original/canonical import path
from .ssm import SelectiveSSM


def build_mlp(d_model: int, d_ff: int, moe_cfg, layer_index: int) -> nn.Module:
    """Dense SwiGLU, unless moe_cfg says this particular layer_index
    should be sparse (moe_cfg.enabled and layer_index is a multiple of
    every_n_layers) -- the one place HybridBlock decides dense-vs-MoE, so
    the two call sites (a HybridBlock's own construction, and any test
    that wants the exact same rule) never drift apart."""
    if moe_cfg is not None and moe_cfg.enabled and (layer_index % moe_cfg.every_n_layers == 0):
        return MoEFeedForward(
            d_model=d_model, num_experts=moe_cfg.num_experts, num_experts_per_tok=moe_cfg.num_experts_per_tok,
            num_shared_experts=moe_cfg.num_shared_experts, d_ff_expert=moe_cfg.d_ff_expert,
            router_noise_std=moe_cfg.router_noise_std,
        )
    return SwiGLU(d_model, d_ff)


class HybridBlock(nn.Module):
    """One residual block. `kind='A'` -> grouped-query attention mixer,
    `kind='S'` -> selective-SSM mixer. Both share the same pre-norm +
    gated-MLP residual structure so BackboneConfig.layer_pattern can freely
    interleave them (e.g. "ASSA" repeated across n_layers). The MLP itself
    is dense SwiGLU unless moe_cfg routes this layer_index to a sparse
    MoEFeedForward instead (see build_mlp) -- orthogonal to, and freely
    combinable with, the attention/SSM mixer choice."""

    def __init__(self, kind: str, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, max_seq_len: int, rope_theta: float, ssm_state_dim: int,
                 ssm_conv_kernel: int, norm_eps: float, dropout: float,
                 rope_scaling=None, qk_norm: bool = False, moe_cfg=None, layer_index: int = 0):
        super().__init__()
        assert kind in ("A", "S")
        self.kind = kind
        self.norm1 = RMSNorm(d_model, norm_eps)
        if kind == "A":
            self.mixer = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len, rope_theta, dropout,
                                                rope_scaling=rope_scaling, qk_norm=qk_norm, norm_eps=norm_eps)
        else:
            self.mixer = SelectiveSSM(d_model, state_dim=ssm_state_dim, conv_kernel=ssm_conv_kernel)
        self.norm2 = RMSNorm(d_model, norm_eps)
        self.mlp = build_mlp(d_model, d_ff, moe_cfg, layer_index)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        if self.kind == "A":
            x = x + self.mixer(self.norm1(x), kv_cache=cache)
        else:
            x = x + self.mixer(self.norm1(x), ssm_state=cache)
        x = x + self.mlp(self.norm2(x))
        return x

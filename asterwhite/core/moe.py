"""
AsterWhite :: core.moe

Sparse mixture-of-experts feed-forward layer (Shazeer et al. 2017 top-k
routing, DeepSeek-V2/V3-style optional always-on shared experts) --
BackboneConfig.moe / config.MoEConfig gates this on; a HybridBlock whose
MLP is an MoEFeedForward has far more total parameters than a dense
SwiGLU of the same d_model, at roughly num_experts_per_tok/num_experts
the FLOPs/token, since only the top-k routed experts (plus any always-on
shared experts) actually run on a given token.

This is a plain-PyTorch reference implementation -- a Python loop over
experts using boolean-mask gather/scatter, correct and hardware-
independent, exactly the same honesty core.ssm.SelectiveSSM's docstring
gives for the same reason: a fused/grouped-GEMM expert-dispatch kernel
(the standard MoE production technique -- sort tokens by assigned expert,
one batched matmul per expert on a contiguous slice, unsort) can be
dropped in behind MoEFeedForward.forward's exact interface for large-
batch training/serving throughput without touching any caller.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU


def load_balancing_loss(router_probs: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer/GShard/Mixtral-style auxiliary loss: minimized
    when tokens are spread evenly across experts (both by selection
    frequency AND by router-probability mass), which is what keeps
    routing from collapsing onto a handful of "winner" experts early in
    training. 0 when perfectly balanced across `num_experts` experts,
    growing as routing concentrates.

    router_probs: (N, E) softmax router probabilities for N tokens over
        E experts (BEFORE top-k selection).
    selected_experts: (N, k) long, the top-k expert indices actually
        routed to (AFTER top-k selection) -- may include fewer than E
        distinct experts per token.
    """
    _n_tokens, num_experts_dim = router_probs.shape
    assert num_experts_dim == num_experts
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()  # (N, k, E)
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # (E,) mean top-k-membership count per expert
    router_prob_per_expert = router_probs.mean(dim=0)       # (E,) mean router probability mass per expert
    return num_experts * torch.sum(tokens_per_expert * router_prob_per_expert) / max(1, selected_experts.shape[1])


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 8, num_experts_per_tok: int = 2,
                 num_shared_experts: int = 1, d_ff_expert: Optional[int] = None,
                 router_noise_std: float = 0.0):
        super().__init__()
        assert 0 < num_experts_per_tok <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.router_noise_std = router_noise_std
        d_ff_expert = d_ff_expert or max(1, (d_model * 4) // num_experts_per_tok)
        self.d_ff_expert = d_ff_expert

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff_expert) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([SwiGLU(d_model, d_ff_expert) for _ in range(num_shared_experts)])

        # Populated on every forward() call -- training.trainer.Trainer (and
        # core.model.AsterWhiteCore.collect_moe_aux_loss) read this AFTER a
        # forward pass rather than threading it through every caller's
        # return signature, so this stays a drop-in MLP replacement for
        # HybridBlock (same forward(x) -> x shape as dense SwiGLU).
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (N, D) where N = B*T

        router_logits = self.router(x_flat)  # (N, E)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x_flat.dtype)  # (N, E)

        topk_probs, topk_idx = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)  # (N, k) each
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # renormalize to sum 1

        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            slot_mask = topk_idx == expert_id                 # (N, k) -- which k-slot (if any) picked this expert
            token_mask = slot_mask.any(dim=-1)                 # (N,)   -- which tokens route here at all
            if not torch.any(token_mask):
                continue
            expert_out = expert(x_flat[token_mask])            # (n_sel, D)
            weight = (topk_probs * slot_mask.to(topk_probs.dtype)).sum(dim=-1)[token_mask]  # (n_sel,)
            out = out.index_copy(
                0, token_mask.nonzero(as_tuple=True)[0],
                out[token_mask] + expert_out * weight.unsqueeze(-1),
            )

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        self.last_aux_loss = load_balancing_loss(router_probs, topk_idx, self.num_experts)
        return out.reshape(B, T, D)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, num_experts={self.num_experts}, "
                f"num_experts_per_tok={self.num_experts_per_tok}, num_shared_experts={self.num_shared_experts}, "
                f"d_ff_expert={self.d_ff_expert}")


def expert_param_count(d_model: int, d_ff_expert: int) -> int:
    """Parameter count of ONE SwiGLU expert (or shared expert) -- shared
    with core.autosizer so MoE-enabled configs get an exact (not
    approximated) parameter count without instantiating the model."""
    return 3 * d_model * d_ff_expert  # w1, w2, w3

"""
AsterWhite :: core.attention

Rotary-embedded, grouped-query causal attention. Uses
torch.nn.functional.scaled_dot_product_attention, which dispatches to fused
Flash-Attention-2/3 kernels automatically on supported CUDA hardware (and
falls back to a memory-efficient or math kernel elsewhere) -- no hard
dependency on a separately-compiled flash-attn package that would tie this
codebase to one exact CUDA/torch build.

Two additive, config-gated (default-off, byte-identical-when-unused)
extensions live here:
  - RoPE scaling (BackboneConfig.rope_scaling / RoPEScalingConfig) for
    context-length extension beyond what a checkpoint was originally
    trained at -- "none" (default), "linear" (position interpolation),
    "ntk" (NTK-aware static rescaling), or "yarn".
  - QK-norm (BackboneConfig.qk_norm): RMSNorm on each head's query/key
    vector before RoPE is applied, a cheap real training-stability
    improvement at scale (Gemma 2 / Qwen 3 / OLMo 2).
"""
from __future__ import annotations

import math
from dataclasses import replace as _dataclass_replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """Inverse of RoPE's frequency formula: at which (even) channel index
    does a rotation complete `num_rotations` full turns over a sequence of
    length `max_position_embeddings`? YaRN uses this twice (at
    beta_fast/beta_slow "rotation budgets") to find the channel range that
    should ramp between interpolation and extrapolation."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float,
                                 max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int, device) -> torch.Tensor:
    if low == high:
        high += 1e-3  # avoid a zero-width ramp (division by zero) in the degenerate case
    ramp = (torch.arange(dim, device=device, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(ramp, 0.0, 1.0)


def compute_rope_frequencies(head_dim: int, theta: float, scaling, device) -> Tuple[torch.Tensor, float]:
    """Returns (inv_freq: (head_dim//2,) float32, attention_scaling: float).
    `scaling` is a config.RoPEScalingConfig or None; None / strategy="none"
    reproduces this module's original, unscaled behavior exactly (and
    attention_scaling=1.0 makes it a no-op everywhere it's applied).
    `scaling.original_max_position_embeddings` MUST already be resolved
    (non-None) by the caller for "ntk"/"yarn" -- see RotaryCache.__init__,
    which resolves it from `initial_len` if the config left it unset."""
    if scaling is None or scaling.strategy == "none":
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        return inv_freq, 1.0

    factor = scaling.factor

    if scaling.strategy == "linear":
        # Position interpolation (Chen et al. 2023) scales POSITIONS, not
        # frequencies -- see build_rope_cache, which divides `t` by factor.
        # inv_freq itself is unscaled here.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        return inv_freq, 1.0

    if scaling.strategy == "ntk":
        # NTK-aware static scaling (community formula, "NTK-Aware Scaled
        # RoPE"): rescale theta itself so high frequencies (fine local
        # resolution) barely move while low frequencies stretch to cover
        # the extended range -- typically needs less continued-pretraining
        # than plain linear interpolation to recover quality.
        adjusted_theta = theta * (factor ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (adjusted_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        return inv_freq, 1.0

    # "yarn" (Peng et al. 2023): per-dimension ramp between extrapolation
    # (long-wavelength/low-frequency dims, left alone -- they already
    # cover the extended range) and interpolation (short-wavelength/high-
    # frequency dims, linearly compressed), plus an attention-temperature
    # correction for the entropy shift of a longer context.
    original_len = scaling.original_max_position_embeddings
    assert original_len is not None, "RoPEScalingConfig.original_max_position_embeddings must be resolved " \
                                      "before compute_rope_frequencies (RotaryCache does this automatically)"
    half = head_dim // 2
    pos_freqs = theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _yarn_find_correction_range(scaling.beta_fast, scaling.beta_slow, head_dim, theta, original_len)
    extrapolation_factor = 1.0 - _yarn_linear_ramp(low, high, half, device)
    inv_freq = (inv_freq_interpolation * (1.0 - extrapolation_factor)
                + inv_freq_extrapolation * extrapolation_factor)

    attention_scaling = scaling.attention_factor
    if attention_scaling is None:
        attention_scaling = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inv_freq, attention_scaling


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype, scaling=None):
    inv_freq, attention_scaling = compute_rope_frequencies(head_dim, theta, scaling, device)
    t = torch.arange(seq_len, device=device).float()
    if scaling is not None and scaling.strategy == "linear":
        t = t / scaling.factor
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos() * attention_scaling, freqs.sin() * attention_scaling
    return cos.to(dtype), sin.to(dtype)


class RotaryCache:
    """Lazily-grown cos/sin table. A cache fixed at `max_seq_len` (the
    original approach) silently runs out once offset + T exceeds it --
    `cos[offset:]` just returns fewer rows than `apply_rope` needs, which
    either errors or (worse) misaligns phase. This version extends itself
    (doubling, like a Python list) the first time a longer position is
    requested, so generation is not hard-capped at whatever max_seq_len
    happened to be configured at init -- the initial value is a
    preallocation hint, not a ceiling.

    `scaling`: a config.RoPEScalingConfig, or None for the original
    unscaled behavior. When scaling.strategy != "none" and
    original_max_position_embeddings was left unset, it's resolved to
    `initial_len` HERE (once, on a private copy -- the caller's config
    object is never mutated) since that's the length the caller
    constructed this cache to represent "the model's trained context" at."""

    def __init__(self, head_dim: int, theta: float, initial_len: int, scaling=None):
        self.head_dim = head_dim
        self.theta = theta
        self.scaling = scaling
        if scaling is not None and scaling.strategy != "none" and scaling.original_max_position_embeddings is None:
            self.scaling = _dataclass_replace(scaling, original_max_position_embeddings=max(initial_len, 1))
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._len = 0
        self._grow_to(max(initial_len, 1), device=torch.device("cpu"), dtype=torch.float32)

    def _grow_to(self, min_len: int, device, dtype) -> None:
        new_len = max(min_len, self._len * 2 if self._len else min_len)
        self._cos, self._sin = build_rope_cache(new_len, self.head_dim, self.theta, device, dtype, self.scaling)
        self._len = new_len

    def get(self, offset: int, length: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = offset + length
        if needed > self._len or self._cos.device != device:
            # Recompute in fp32 for precision, then rebuild at the needed
            # length on the target device -- growth is rare (amortized O(1)
            # via doubling), not a per-token cost.
            self._grow_to(needed, device=device, dtype=torch.float32)
        return (self._cos[offset:offset + length].to(dtype=dtype),
                self._sin[offset:offset + length].to(dtype=dtype))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with grouped-query KV heads + rotary position
    embeddings. Pass an (initially empty) dict as `kv_cache` to turn this
    into incremental/streaming attention for token-by-token generation."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int,
                 rope_theta: float = 10000.0, dropout: float = 0.0, rope_scaling=None,
                 qk_norm: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.dropout_p = dropout
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        if qk_norm:
            # Per-head RMSNorm on q/k BEFORE rotation (Gemma 2 / Qwen 3 /
            # OLMo 2): bounds attention-logit growth as depth/width scale up,
            # independent of and complementary to rope_scaling above.
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)

        # Not a persistent nn.Module buffer (never was -- the original was
        # registered with persistent=False, so this is checkpoint-compatible):
        # this cache grows past its initial size instead of hard-capping.
        self._rope_cache = RotaryCache(self.head_dim, rope_theta, max_seq_len, scaling=rope_scaling)

    def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = kv_cache.get("offset", 0) if kv_cache else 0
        cos, sin = self._rope_cache.get(offset, T, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        fresh_len = T
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
            kv_cache["offset"] = offset + T

        is_causal = k.shape[2] == fresh_len
        k_rep = k.repeat_interleave(self.n_rep, dim=1)
        v_rep = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

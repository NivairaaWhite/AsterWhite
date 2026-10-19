"""
AsterWhite :: core.autosizer

Hardware-adaptive sizing: estimates whether a given AsterWhiteConfig fits in
available memory *before* allocating the model, and classifies the machine
into one of the tiers the spec defines (smoke/tiny CPU up through
multi-node). AsterWhite does not claim one fixed full-size model trains on
every device -- this module is what makes that claim concrete instead of
aspirational: it estimates, it refuses unsafe configs by default, and an
operator has to explicitly opt in to override the refusal.

Estimates are analytic approximations (standard transformer/optimizer
memory formulas), not a substitute for an actual dry run -- they are
deliberately conservative (over-estimate) so the default behavior errs
toward refusing rather than OOMing mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil

from ..config import AsterWhiteConfig
from ..utils.error_tracing import AsterWhiteError, get_logger
from .device import DeviceInfo
from .moe import expert_param_count

logger = get_logger("asterwhite.core.autosizer")

_BYTES_PER_DTYPE = {"fp32": 4, "fp16": 2, "bf16": 2}
_ADAM_STATES_PER_PARAM = 2       # exp_avg + exp_avg_sq (fp32 master copies)
_SAFETY_RESERVE_FRACTION = 0.15  # held back from every estimate, per spec's "safety reserve" category


class HardwareTier(str, Enum):
    TIER_0_CPU_TINY = "tier_0_cpu_tiny"                 # 4-8GB RAM: tests, tokenization, tiny models
    TIER_1_SMALL_OR_UNIFIED = "tier_1_small_or_unified"  # 8-32GB RAM/unified: adapters, short-context, quantized
    TIER_2_ACCEL_SMALL = "tier_2_accel_small"           # 8-24GB accelerator: small/medium pretrain, SFT, alignment
    TIER_3_ACCEL_LARGE = "tier_3_accel_large"           # 24-80+GB accelerator: larger single-device training
    TIER_4_MULTI_DEVICE = "tier_4_multi_device"         # multi-GPU/multi-node: full production pretraining
    MOBILE_EMBEDDED = "mobile_embedded"                  # quantized inference / adapter-only


@dataclass
class MemoryEstimate:
    param_count: int
    param_memory_bytes: int
    gradient_memory_bytes: int
    optimizer_state_memory_bytes: int
    activation_memory_bytes: int
    kv_cache_memory_bytes: int
    dataloader_memory_bytes: int
    safety_reserve_bytes: int
    total_required_bytes: int
    available_bytes: int
    fits: bool

    def report(self) -> str:
        def mb(n: int) -> str:
            return f"{n / (1 << 20):.1f} MB"
        return (
            f"params={self.param_count:,} ({mb(self.param_memory_bytes)})  "
            f"grads={mb(self.gradient_memory_bytes)}  optim_state={mb(self.optimizer_state_memory_bytes)}  "
            f"activations~={mb(self.activation_memory_bytes)}  kv_cache~={mb(self.kv_cache_memory_bytes)}  "
            f"dataloader~={mb(self.dataloader_memory_bytes)}  safety_reserve={mb(self.safety_reserve_bytes)}  "
            f"TOTAL required~={mb(self.total_required_bytes)} vs available={mb(self.available_bytes)}  "
            f"fits={self.fits}"
        )


def count_parameters(cfg: AsterWhiteConfig) -> int:
    """Analytic parameter count from config alone (no model instantiation
    required) -- this is what lets the sizer refuse BEFORE allocating.

    Walks the exact same per-layer dense-vs-MoE decision core.layers.
    build_mlp makes (moe.enabled and layer_index % every_n_layers == 0),
    so a MoE-enabled config's parameter count reflects EVERY expert plus
    the router -- not a dense-SwiGLU approximation. This matters: MoE's
    whole point is far more total parameters (and therefore memory for
    weights/gradients/optimizer state) than a dense model of the same
    active-compute-per-token, so under-counting here would make
    guard_or_raise pass configs that don't actually fit."""
    b = cfg.backbone
    head_dim = b.d_model // b.n_heads
    attn_params = (b.n_heads * head_dim * b.d_model                     # q_proj
                   + 2 * b.n_kv_heads * head_dim * b.d_model             # k_proj, v_proj
                   + b.n_heads * head_dim * b.d_model)                   # o_proj
    ssm_d_inner = 2 * b.d_model
    ssm_params = (b.d_model * 2 * ssm_d_inner                            # in_proj.weight: (2*ssm_d_inner, d_model)
                  + ssm_d_inner * b.ssm_conv_kernel                      # conv1d.weight: (ssm_d_inner, 1, kernel)
                  + ssm_d_inner                                          # conv1d.bias: (ssm_d_inner,)
                  + ssm_d_inner * (b.ssm_state_dim * 2 + 1)              # x_proj.weight: (2*N+1, ssm_d_inner)
                  + ssm_d_inner                                          # dt_proj.weight: (ssm_d_inner, 1)
                  + ssm_d_inner                                          # dt_proj.bias: (ssm_d_inner,)
                  + ssm_d_inner * b.ssm_state_dim                        # A_log: (ssm_d_inner, N)
                  + ssm_d_inner                                          # D: (ssm_d_inner,)
                  + ssm_d_inner * b.d_model)                             # out_proj.weight: (d_model, ssm_d_inner)
    dense_mlp_params = 3 * b.d_model * b.d_ff  # SwiGLU: w1, w2, w3

    pattern = [b.layer_pattern[i % len(b.layer_pattern)] for i in range(b.n_layers)] if b.layer_pattern else ["A"] * b.n_layers
    moe = b.moe
    moe_d_ff_expert = moe.d_ff_expert or max(1, (b.d_model * 4) // moe.num_experts_per_tok)  # mirrors MoEFeedForward's own default
    moe_layer_params = (
        (moe.num_experts + moe.num_shared_experts) * expert_param_count(b.d_model, moe_d_ff_expert)
        + b.d_model * moe.num_experts  # router: nn.Linear(d_model, num_experts, bias=False)
    )

    mixer_params = 0
    mlp_params_total = 0
    for i, kind in enumerate(pattern):
        mixer_params += attn_params if kind == "A" else ssm_params
        mlp_params_total += moe_layer_params if (moe.enabled and i % moe.every_n_layers == 0) else dense_mlp_params

    norm_params = 2 * b.n_layers * b.d_model + b.d_model  # 2 norms/layer + final norm
    embed_params = b.vocab_size * b.d_model
    lm_head_params = 0 if b.tie_word_embeddings else b.vocab_size * b.d_model
    mtp_head_params = b.mtp_n_future_tokens * b.vocab_size * b.d_model
    qk_norm_params = (2 * pattern.count("A") * head_dim) if b.qk_norm else 0  # q_norm + k_norm weight per attn layer
    return int(mixer_params + mlp_params_total + norm_params + embed_params + lm_head_params
               + mtp_head_params + qk_norm_params)


def estimate_memory(cfg: AsterWhiteConfig, device_info: DeviceInfo, batch_size: Optional[int] = None,
                     seq_len: Optional[int] = None) -> MemoryEstimate:
    """Conservative analytic estimate across every category the spec asks
    for. `available_bytes` comes from the resolved device when known (GPU
    total memory) or from host RAM (psutil) otherwise -- CPU training is
    bounded by system RAM, not a separate accelerator pool."""
    n_params = count_parameters(cfg)
    param_bytes_per = _BYTES_PER_DTYPE[cfg.training.amp_dtype]
    param_memory = n_params * param_bytes_per
    gradient_memory = n_params * param_bytes_per
    optimizer_state_memory = n_params * 4 * _ADAM_STATES_PER_PARAM  # AdamW keeps fp32 master moments

    bsz = batch_size or cfg.training.micro_batch_size
    seq = seq_len or cfg.backbone.max_seq_len
    # Rough activation estimate: per-layer hidden states + MLP intermediate,
    # kept for backward -- a coarse but conservative multiplier.
    activation_memory = bsz * seq * cfg.backbone.d_model * cfg.backbone.n_layers * 6 * param_bytes_per
    # KV cache: only attention layers hold K/V; SSM state is O(1) per token.
    pattern = [cfg.backbone.layer_pattern[i % len(cfg.backbone.layer_pattern)] for i in range(cfg.backbone.n_layers)]
    n_attn_layers = pattern.count("A")
    kv_cache_memory = 2 * n_attn_layers * bsz * seq * cfg.backbone.n_kv_heads * (
        cfg.backbone.d_model // cfg.backbone.n_heads) * param_bytes_per
    dataloader_memory = bsz * seq * 8 * 4  # small, fixed-ish overhead for staged token-id batches

    subtotal = param_memory + gradient_memory + optimizer_state_memory + activation_memory + kv_cache_memory + dataloader_memory
    safety_reserve = int(subtotal * _SAFETY_RESERVE_FRACTION)
    total_required = subtotal + safety_reserve

    if device_info.total_memory_bytes:
        available = device_info.total_memory_bytes
    else:
        available = psutil.virtual_memory().available

    return MemoryEstimate(
        param_count=n_params, param_memory_bytes=param_memory, gradient_memory_bytes=gradient_memory,
        optimizer_state_memory_bytes=optimizer_state_memory, activation_memory_bytes=activation_memory,
        kv_cache_memory_bytes=kv_cache_memory, dataloader_memory_bytes=dataloader_memory,
        safety_reserve_bytes=safety_reserve, total_required_bytes=total_required,
        available_bytes=available, fits=(total_required <= available),
    )


def classify_tier(device_info: DeviceInfo, world_size: int = 1) -> HardwareTier:
    if world_size > 1:
        return HardwareTier.TIER_4_MULTI_DEVICE
    if device_info.backend == "cpu":
        total = psutil.virtual_memory().total
        return HardwareTier.TIER_0_CPU_TINY if total < 8 * (1 << 30) else HardwareTier.TIER_1_SMALL_OR_UNIFIED
    if device_info.backend == "mps":
        return HardwareTier.TIER_1_SMALL_OR_UNIFIED
    if device_info.backend in ("cuda", "rocm"):
        gb = (device_info.total_memory_bytes or 0) / (1 << 30)
        if gb <= 0:
            return HardwareTier.TIER_1_SMALL_OR_UNIFIED
        if gb < 24:
            return HardwareTier.TIER_2_ACCEL_SMALL
        return HardwareTier.TIER_3_ACCEL_LARGE
    return HardwareTier.TIER_1_SMALL_OR_UNIFIED


def guard_or_raise(estimate: MemoryEstimate, override: bool = False) -> None:
    """MUST refuse configurations predicted to exceed safe memory unless the
    operator explicitly overrides (--allow-oversized-config / override=True)."""
    if estimate.fits or override:
        if not estimate.fits:
            logger.warning(f"config estimated to exceed available memory but override=True was passed. {estimate.report()}")
        return
    raise AsterWhiteError(
        "This configuration is estimated to exceed available memory and was refused before "
        "allocating the model (safer than OOMing mid-run):\n  " + estimate.report() +
        "\nReduce batch/sequence/model size, choose a smaller configs/ profile, or pass "
        "--allow-oversized-config to override this check."
    )

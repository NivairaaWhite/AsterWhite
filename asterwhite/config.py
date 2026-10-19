"""
AsterWhite :: core configuration.

MODEL_NAME is the single source of truth for the product/model name. It
must remain exactly "AsterWhite" (case-sensitive) everywhere it surfaces --
banners, logs, checkpoints, exported character cards, etc.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Literal, Optional, get_type_hints

MODEL_NAME: str = "AsterWhite"
CONFIG_SCHEMA_VERSION: int = 2


@dataclass
class RoPEScalingConfig:
    """Context-extension scaling for the attention blocks' rotary
    embeddings (core.attention.RotaryCache) -- lets a checkpoint trained
    at max_seq_len be continued-pretrained or served at a longer length
    without the attention layers' positional signal degrading the way
    plain unscaled RoPE does past its trained range. SSM blocks need no
    such treatment (their O(1) recurrent state has no positional
    encoding to extrapolate), so this only ever touches "A" blocks.

    strategy:
      "none"   -- unscaled RoPE (the historical default; byte-identical
                  to this repo's original behavior).
      "linear" -- position interpolation (Chen et al. 2023): compress
                  positions by 1/factor before computing angles. Cheap,
                  needs brief continued-pretraining at the new length to
                  recover full quality.
      "ntk"    -- NTK-aware scaling (rescales rope_theta itself by the
                  community-standard formula so low frequencies barely
                  move and high frequencies interpolate) -- typically
                  usable with LESS continued-pretraining than "linear".
      "yarn"   -- YaRN (Peng et al. 2023): per-dimension ramp between
                  NTK-style extrapolation (long-wavelength dims, left
                  alone) and interpolation (short-wavelength dims,
                  scaled), plus an attention-temperature correction
                  (attention_factor) for the entropy shift of a longer
                  context. The long-context recipe behind most
                  128K+-context open checkpoints as of this repo's
                  knowledge; the most work to get right, the least
                  continued-pretraining needed of the three.
    factor: target_max_seq_len / original_max_position_embeddings.
    original_max_position_embeddings: the length the model was actually
        trained at pre-scaling; defaults to backbone.max_seq_len at
        construction time if left None (see AsterWhiteCore.__init__).
    beta_fast/beta_slow: YaRN-only ramp boundaries (in "number of
        rotations"), values from the original paper's reference config.
    attention_factor: YaRN-only logit temperature correction; None ->
        the paper's closed-form default (0.1*ln(factor)+1.0).
    """
    strategy: Literal["none", "linear", "ntk", "yarn"] = "none"
    factor: float = 1.0
    original_max_position_embeddings: Optional[int] = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: Optional[float] = None


@dataclass
class MoEConfig:
    """Sparse mixture-of-experts feed-forward layers (Shazeer et al. 2017
    top-k routing; shared-expert design per DeepSeek-V2/V3). Swaps the
    dense SwiGLU MLP in every `every_n_layers`-th HybridBlock for a
    MoEFeedForward (asterwhite.core.moe): far more total parameters at
    roughly the same active-compute-per-token cost as a dense model of
    d_ff_expert's width -- the standard lever frontier labs use to scale
    capacity without scaling FLOPs/token 1:1.

    enabled: false (default) -> every MLP is the original dense SwiGLU;
        zero behavior change from this repo's original architecture.
    num_experts / num_experts_per_tok: total routed experts and how many
        fire per token (top-k). num_experts_per_tok must be <= num_experts.
    num_shared_experts: experts that run on EVERY token unconditionally,
        added to the routed output (DeepSeek-style) -- captures common
        patterns so routed experts can specialize instead of each
        re-learning generic behavior. 0 disables (Mixtral-style pure
        routing).
    d_ff_expert: hidden width of EACH expert's SwiGLU. None -> d_ff //
        num_experts_per_tok, so active compute per token stays roughly
        equal to a dense d_ff-wide MLP even though many more parameters
        exist in total.
    aux_loss_coef: weight on the Switch-Transformer-style load-balancing
        auxiliary loss (asterwhite.core.moe.load_balancing_loss) that
        training.trainer.Trainer adds to the main loss whenever any
        layer's MoEFeedForward.last_aux_loss is populated -- without
        this, routing tends to collapse onto a handful of experts.
    every_n_layers: apply MoE to every Nth eligible layer, dense
        elsewhere (1 = every layer). A common practice for keeping early
        layers dense while concentrating capacity later in the stack.
    router_noise_std: optional Gaussian noise added to router logits
        during training only (0.0 = off) -- a standard exploration aid
        against early routing collapse (Shazeer et al.'s "noisy top-k").
    capacity_factor: None (default) means no-drop routing -- every
        token's chosen experts always run on it, which is correct but
        (on real distributed hardware, not this repo's reference
        implementation) sacrifices the throughput a fixed per-expert
        capacity + token-dropping would buy. Correctness over throughput
        is the deliberate default; set a float (e.g. 1.25) only once
        you've implemented/measured capacity-based dropping for your
        deployment -- the reference MoEFeedForward here does not drop
        tokens regardless of this value (see its docstring).
    """
    enabled: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    d_ff_expert: Optional[int] = None
    aux_loss_coef: float = 0.01
    every_n_layers: int = 1
    router_noise_std: float = 0.0
    capacity_factor: Optional[float] = None


@dataclass
class BackboneConfig:
    """AsterWhiteCore hybrid attention/state-space decoder."""
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4                  # grouped-query attention
    d_ff: int = 4096
    max_seq_len: int = 8192
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # Hybrid layer interleave. "A" = attention block, "S" = selective-SSM
    # block. Repeats/truncates to fill n_layers, e.g. "ASSA" with
    # n_layers=8 -> A S S A A S S A. Attention gives precise local/global
    # mixing; SSM gives O(1)-memory-per-token long-range compression.
    layer_pattern: str = "ASSA"
    ssm_state_dim: int = 16
    ssm_conv_kernel: int = 4
    tie_word_embeddings: bool = True
    # Long-context RoPE extrapolation/interpolation -- see RoPEScalingConfig.
    rope_scaling: RoPEScalingConfig = field(default_factory=RoPEScalingConfig)
    # Sparse mixture-of-experts MLPs -- see MoEConfig.
    moe: MoEConfig = field(default_factory=MoEConfig)
    # Query/Key RMSNorm before RoPE (Gemma 2 / Qwen 3 / OLMo 2 style): a
    # cheap, real stability improvement at scale -- bounds attention logit
    # growth as d_model/n_layers increase, at the cost of two tiny extra
    # RMSNorms per attention block. Off by default (byte-identical
    # original behavior); worth turning on for any config past roughly
    # the nvidia_single_gpu tier.
    qk_norm: bool = False
    # Multi-token prediction (Gloeckle et al. 2024 parallel-head variant):
    # M extra untied linear heads, each predicting a further future
    # offset (+2, +3, ..., +1+M) from the SAME final hidden state the
    # standard lm_head already predicts +1 from -- zero extra trunk
    # forward passes. 0 (default) = disabled, zero behavior/param change.
    # training.trainer.Trainer adds an auxiliary loss for these
    # automatically whenever this is > 0 (see TrainingConfig.mtp_loss_coef);
    # asterwhite.inference.speculative uses the same heads as a
    # zero-extra-pass self-speculative-decoding drafter at inference time.
    mtp_n_future_tokens: int = 0


@dataclass
class NarrativeConfig:
    """Universe/agent/generation subsystem dimensions."""
    world_state_dim: int = 512
    character_dim: int = 256
    plan_dim: int = 512
    working_memory_slots: int = 64            # agent dual-tier memory: fast RAM tier
    episodic_memory_capacity: int = 100_000    # agent dual-tier memory: vectorized tier
    max_plan_beats: int = 64
    revision_passes: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 8
    micro_batch_size: int = 1                 # per-forward batch; accumulated up to batch_size
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    amp_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    checkpoint_every: int = 1000
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 20
    distributed_strategy: Literal["ddp", "fsdp"] = "ddp"
    # Wraps each HybridBlock in torch.utils.checkpoint (training.trainer.
    # Trainer wires this for real -- see core.model.AsterWhiteCore.forward):
    # recomputes activations during backward instead of storing them,
    # trading ~30% more compute time for activation memory that stops
    # scaling with n_layers, which is what lets a deeper model fit in the
    # same VRAM. No effect on eval/generation (only engages when the
    # model is in .train() mode with grad enabled).
    activation_checkpointing: bool = False
    eval_every: int = 500
    grad_norm_clip_report_every: int = 20
    # Wraps the model with torch.compile (PyTorch 2.x graph capture +
    # fusion) before training -- a real, standard throughput lever, but
    # ONLY actually attempted when the resolved device is CUDA: compilation
    # is lazy (it happens on the first real forward call, not the wrap
    # itself), so a try/except around the wrap can't catch a slow/hanging
    # first compilation -- only an immediate construction-time error. On a
    # CPU-only sandbox without a full build toolchain, compilation has been
    # observed to hang rather than fail fast, so CPU/MPS/XPU get a logged
    # no-op instead of that risk; CUDA (where this is mature and reliably
    # fast to build) still gets a try/except around the wrap itself as a
    # second layer of safety.
    compile_model: bool = False
    # Exponential moving average of trained weights (Polyak averaging).
    # None (default) = disabled. A float in (0, 1), e.g. 0.999, maintains
    # a shadow copy of every parameter updated as
    # ema = decay*ema + (1-decay)*param after every optimizer step;
    # Trainer.save_checkpoint stores it alongside the raw weights
    # (Trainer.ema_state_dict()) -- the EMA weights are what most modern
    # pretraining recipes actually ship, since they're consistently as-good
    # -or-better and smoother than the raw last-step weights.
    ema_decay: Optional[float] = None
    # Weight on the multi-token-prediction auxiliary loss (see
    # BackboneConfig.mtp_n_future_tokens); irrelevant/unused when that is 0.
    mtp_loss_coef: float = 0.3
    # Attempt torch.optim.AdamW(..., fused=True) (a real, single-kernel
    # fused optimizer step -- meaningfully faster on CUDA) and transparently
    # fall back to the unfused implementation if this torch build/device
    # doesn't support it, rather than erroring.
    optimizer_fused: bool = True


@dataclass
class GRPOConfig:
    group_size: int = 8              # completions sampled per prompt
    kl_coef: float = 0.02
    clip_eps: float = 0.2
    learning_rate: float = 1e-5
    max_new_tokens: int = 512
    temperature: float = 1.0
    reward_clip: float = 10.0        # rewards are clipped to [-reward_clip, +reward_clip] before normalization


@dataclass
class InferenceConfig:
    """Serving-time defaults -- kept in the same validated config system as
    training so a production_inference/edge_inference profile is one YAML
    file, not a second untracked settings surface."""
    max_total_tokens: int = 4096
    default_max_new_tokens: int = 256
    default_temperature: float = 0.9
    default_top_p: float = 0.9
    default_top_k: int = 0            # 0 -> disabled
    default_repetition_penalty: float = 1.0
    quantize: bool = False            # dynamic INT8 quantization for CPU inference
    max_concurrent_requests: int = 8
    request_timeout_seconds: float = 60.0


@dataclass
class AsterWhiteConfig:
    name: str = MODEL_NAME
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    narrative: NarrativeConfig = field(default_factory=NarrativeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @staticmethod
    def tiny_for_testing() -> "AsterWhiteConfig":
        """Same code paths as production, small enough to run in seconds on
        CPU. Used by tests/test_smoke.py and by train.py's --dry-run."""
        cfg = AsterWhiteConfig()
        cfg.backbone.vocab_size = 256
        cfg.backbone.d_model = 32
        cfg.backbone.n_layers = 4
        cfg.backbone.n_heads = 4
        cfg.backbone.n_kv_heads = 2
        cfg.backbone.d_ff = 64
        cfg.backbone.max_seq_len = 128
        cfg.backbone.ssm_state_dim = 8
        cfg.narrative.world_state_dim = 16
        cfg.narrative.character_dim = 16
        cfg.narrative.plan_dim = 16
        cfg.narrative.working_memory_slots = 4
        cfg.narrative.episodic_memory_capacity = 128
        cfg.narrative.max_plan_beats = 8
        cfg.training.batch_size = 2
        cfg.training.micro_batch_size = 1
        cfg.grpo.group_size = 2
        cfg.grpo.max_new_tokens = 8
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["schema_version"] = CONFIG_SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AsterWhiteConfig":
        """Reconstructs nested dataclasses from a plain dict (as loaded
        from YAML or a checkpoint's stored metadata) -- dataclasses.asdict
        is one-way, this is the inverse. Unknown keys are rejected (typo
        protection) rather than silently ignored."""
        d = {k: v for k, v in d.items() if k != "schema_version"}
        return _dict_to_dataclass(cls, d)


def _dict_to_dataclass(cls, d: Dict[str, Any]):
    # `from __future__ import annotations` makes every field's .type a
    # string (e.g. "BackboneConfig"), not the class object -- resolve real
    # types via get_type_hints before checking is_dataclass, or nested
    # configs would never be reconstructed (every branch would fall to the
    # `else` and store a plain dict where a dataclass instance belongs).
    resolved_types = get_type_hints(cls)
    kwargs = {}
    valid_names = {f.name for f in fields(cls)}
    unknown = set(d.keys()) - valid_names
    if unknown:
        raise AsterWhiteConfigError(f"unknown config key(s) for {cls.__name__}: {sorted(unknown)} -- "
                                     f"check for typos; known keys are {sorted(valid_names)}")
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        field_type = resolved_types.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


class AsterWhiteConfigError(Exception):
    """Raised by validate_config / from_dict on a malformed configuration.
    Deliberately NOT AsterWhiteError -- config.py is intentionally
    dependency-free (no import of utils.error_tracing) so it can be
    imported first, before anything else, with zero risk of a circular
    import."""


def validate_config(cfg: AsterWhiteConfig) -> None:
    """MUST reject invalid head counts, dimensions, vocabulary sizes,
    sequence lengths and precision combinations BEFORE allocating a model
    (spec 4.2). Call this right after resolving a config from YAML/CLI
    overrides, before constructing AsterWhiteCore."""
    b, n, t, g = cfg.backbone, cfg.narrative, cfg.training, cfg.grpo
    i = cfg.inference
    errors = []

    if b.d_model % b.n_heads != 0:
        errors.append(f"backbone.d_model ({b.d_model}) must be divisible by backbone.n_heads ({b.n_heads})")
    if b.n_heads % b.n_kv_heads != 0:
        errors.append(f"backbone.n_heads ({b.n_heads}) must be divisible by backbone.n_kv_heads ({b.n_kv_heads}) "
                       f"for grouped-query attention")
    if b.vocab_size <= 0:
        errors.append(f"backbone.vocab_size must be positive, got {b.vocab_size}")
    if b.d_model <= 0 or b.n_layers <= 0 or b.n_heads <= 0 or b.n_kv_heads <= 0:
        errors.append("backbone.d_model/n_layers/n_heads/n_kv_heads must all be positive")
    if b.max_seq_len <= 0:
        errors.append(f"backbone.max_seq_len must be positive, got {b.max_seq_len}")
    if not b.layer_pattern or any(ch not in ("A", "S") for ch in b.layer_pattern):
        errors.append(f"backbone.layer_pattern must be non-empty and contain only 'A'/'S', got {b.layer_pattern!r}")
    if b.ssm_state_dim <= 0 or b.ssm_conv_kernel <= 0:
        errors.append("backbone.ssm_state_dim/ssm_conv_kernel must be positive")

    rs = b.rope_scaling
    if rs.strategy not in ("none", "linear", "ntk", "yarn"):
        errors.append(f"backbone.rope_scaling.strategy must be one of none/linear/ntk/yarn, got {rs.strategy!r}")
    if rs.strategy != "none" and rs.factor <= 1.0:
        errors.append(f"backbone.rope_scaling.factor must be > 1.0 when strategy={rs.strategy!r}, got {rs.factor}")
    if rs.original_max_position_embeddings is not None and rs.original_max_position_embeddings <= 0:
        errors.append("backbone.rope_scaling.original_max_position_embeddings must be positive when set")
    if rs.strategy == "yarn" and rs.beta_fast <= rs.beta_slow:
        errors.append(f"backbone.rope_scaling.beta_fast ({rs.beta_fast}) must be > beta_slow ({rs.beta_slow})")

    m = b.moe
    if m.enabled:
        if m.num_experts <= 0:
            errors.append(f"backbone.moe.num_experts must be positive, got {m.num_experts}")
        if not (0 < m.num_experts_per_tok <= m.num_experts):
            errors.append(f"backbone.moe.num_experts_per_tok ({m.num_experts_per_tok}) must be in "
                           f"[1, num_experts={m.num_experts}]")
        if m.num_shared_experts < 0:
            errors.append("backbone.moe.num_shared_experts must be >= 0")
        if m.d_ff_expert is not None and m.d_ff_expert <= 0:
            errors.append("backbone.moe.d_ff_expert must be positive when set")
        if m.aux_loss_coef < 0:
            errors.append("backbone.moe.aux_loss_coef must be >= 0")
        if m.every_n_layers <= 0:
            errors.append(f"backbone.moe.every_n_layers must be positive, got {m.every_n_layers}")
        if m.router_noise_std < 0:
            errors.append("backbone.moe.router_noise_std must be >= 0")
        if m.capacity_factor is not None and m.capacity_factor < 1.0:
            errors.append(f"backbone.moe.capacity_factor must be >= 1.0 when set, got {m.capacity_factor}")

    if b.mtp_n_future_tokens < 0:
        errors.append(f"backbone.mtp_n_future_tokens must be >= 0, got {b.mtp_n_future_tokens}")

    if t.ema_decay is not None and not (0.0 < t.ema_decay < 1.0):
        errors.append(f"training.ema_decay must be in (0, 1) when set, got {t.ema_decay}")
    if t.mtp_loss_coef < 0:
        errors.append("training.mtp_loss_coef must be >= 0")

    if t.amp_dtype not in ("bf16", "fp16", "fp32"):
        errors.append(f"training.amp_dtype must be one of bf16/fp16/fp32, got {t.amp_dtype!r}")
    if t.micro_batch_size <= 0 or t.batch_size <= 0:
        errors.append("training.batch_size/micro_batch_size must be positive")
    if t.micro_batch_size > t.batch_size:
        errors.append(f"training.micro_batch_size ({t.micro_batch_size}) cannot exceed "
                       f"training.batch_size ({t.batch_size})")
    if t.learning_rate <= 0:
        errors.append(f"training.learning_rate must be positive, got {t.learning_rate}")
    if t.grad_clip <= 0:
        errors.append(f"training.grad_clip must be positive, got {t.grad_clip}")

    if t.distributed_strategy not in ("ddp", "fsdp"):
        errors.append(f"training.distributed_strategy must be 'ddp' or 'fsdp', got {t.distributed_strategy!r}")

    if i.max_total_tokens <= 0:
        errors.append(f"inference.max_total_tokens must be positive, got {i.max_total_tokens}")
    if i.default_max_new_tokens <= 0 or i.default_max_new_tokens > i.max_total_tokens:
        errors.append("inference.default_max_new_tokens must be positive and <= inference.max_total_tokens")
    if not (0.0 < i.default_top_p <= 1.0):
        errors.append(f"inference.default_top_p must be in (0, 1], got {i.default_top_p}")
    if i.max_concurrent_requests <= 0:
        errors.append(f"inference.max_concurrent_requests must be positive, got {i.max_concurrent_requests}")

    if n.world_state_dim <= 0 or n.character_dim <= 0 or n.plan_dim <= 0:
        errors.append("narrative.world_state_dim/character_dim/plan_dim must all be positive")
    if n.working_memory_slots <= 0 or n.episodic_memory_capacity <= 0:
        errors.append("narrative.working_memory_slots/episodic_memory_capacity must be positive")

    if g.group_size <= 1:
        errors.append(f"grpo.group_size must be > 1 (need at least 2 completions to form a within-group "
                       f"baseline), got {g.group_size}")
    if not (0.0 < g.clip_eps < 1.0):
        errors.append(f"grpo.clip_eps must be in (0, 1), got {g.clip_eps}")
    if g.reward_clip <= 0:
        errors.append(f"grpo.reward_clip must be positive, got {g.reward_clip}")

    if errors:
        raise AsterWhiteConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def resolve_secret(name: str) -> Optional[str]:
    """Secrets MUST come from a secret manager or environment injection,
    never source control (spec 4.2). This is the one place a secret value
    is read from in this codebase -- always ASTERWHITE_SECRET_<NAME> from
    the environment. There is deliberately no YAML/config-file code path
    for secret VALUES (profiles may reference a secret by NAME, never
    embed one)."""
    return os.environ.get(f"ASTERWHITE_SECRET_{name.upper()}")

"""AsterWhite :: tests.test_attention_extensions -- RoPE scaling
(config.RoPEScalingConfig / core.attention.compute_rope_frequencies) and
QK-norm (BackboneConfig.qk_norm), both default-off and both threaded
through core.attention.GroupedQueryAttention."""
from __future__ import annotations

import torch

from asterwhite.config import AsterWhiteConfig, RoPEScalingConfig, validate_config
from asterwhite.core.attention import GroupedQueryAttention, build_rope_cache, compute_rope_frequencies
from asterwhite.core.model import AsterWhiteCore


def test_rope_none_strategy_is_byte_identical_to_unscaled_formula():
    head_dim, theta, seq_len = 16, 10000.0, 32
    cos, sin = build_rope_cache(seq_len, head_dim, theta, torch.device("cpu"), torch.float32, scaling=None)
    inv_freq_ref = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs_ref = torch.outer(torch.arange(seq_len).float(), inv_freq_ref)
    assert torch.allclose(cos, freqs_ref.cos())
    assert torch.allclose(sin, freqs_ref.sin())


def test_rope_scaling_strategies_produce_finite_differing_frequencies():
    head_dim, theta = 16, 10000.0
    cos_none, _ = build_rope_cache(128, head_dim, theta, torch.device("cpu"), torch.float32, scaling=None)
    for strategy in ("linear", "ntk", "yarn"):
        scaling = RoPEScalingConfig(strategy=strategy, factor=4.0, original_max_position_embeddings=32)
        cos, sin = build_rope_cache(128, head_dim, theta, torch.device("cpu"), torch.float32, scaling=scaling)
        assert cos.shape == (128, head_dim // 2)
        assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
        assert not torch.allclose(cos[:32], cos_none[:32]), f"{strategy} should differ from unscaled"


def test_yarn_continuity_near_factor_one_matches_unscaled():
    """A real correctness property, not just 'runs without crashing': YaRN
    at factor~1 (no real extension requested) must reduce to ~unscaled
    RoPE, since YaRN's ramp interpolates between exactly that and full
    interpolation as factor grows."""
    head_dim, theta = 16, 10000.0
    scaling = RoPEScalingConfig(strategy="yarn", factor=1.0001, original_max_position_embeddings=32)
    inv_freq_yarn, _ = compute_rope_frequencies(head_dim, theta, scaling, torch.device("cpu"))
    inv_freq_none, _ = compute_rope_frequencies(head_dim, theta, None, torch.device("cpu"))
    assert torch.allclose(inv_freq_yarn, inv_freq_none, atol=1e-3)


def test_rope_scaling_default_off_config_is_unchanged():
    cfg = AsterWhiteConfig.tiny_for_testing()
    assert cfg.backbone.rope_scaling.strategy == "none"
    validate_config(cfg)  # must not raise


def test_rope_scaling_end_to_end_through_attention_module():
    scaling = RoPEScalingConfig(strategy="yarn", factor=4.0)  # original_max_position_embeddings auto-resolved
    attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, max_seq_len=16, rope_scaling=scaling)
    assert attn._rope_cache.scaling.original_max_position_embeddings == 16
    x = torch.randn(2, 8, 32)
    out = attn(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_rope_scaling_validation_rejects_bad_config():
    import pytest
    from asterwhite.config import AsterWhiteConfigError

    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.rope_scaling.strategy = "yarn"
    cfg.backbone.rope_scaling.factor = 0.5  # must be > 1.0 for a real scaling strategy
    with pytest.raises(AsterWhiteConfigError):
        validate_config(cfg)


def test_qk_norm_forward_and_gradients():
    attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, max_seq_len=16, qk_norm=True)
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = attn(x)
    assert out.shape == (2, 8, 32) and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert attn.q_norm.weight.grad is not None and attn.k_norm.weight.grad is not None


def test_qk_norm_off_by_default_has_no_norm_submodules():
    attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, max_seq_len=16)
    assert not hasattr(attn, "q_norm")
    assert not hasattr(attn, "k_norm")


def test_kitchen_sink_moe_qknorm_yarn_mtp_forward_backward_and_generate():
    """All four frontier additions enabled simultaneously: forward,
    backward (every parameter must receive a gradient), and generate()
    all still work together."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.moe.enabled = True
    cfg.backbone.moe.num_experts = 4
    cfg.backbone.moe.num_experts_per_tok = 2
    cfg.backbone.moe.d_ff_expert = 16
    cfg.backbone.qk_norm = True
    cfg.backbone.rope_scaling.strategy = "yarn"
    cfg.backbone.rope_scaling.factor = 2.0
    cfg.backbone.mtp_n_future_tokens = 2
    validate_config(cfg)

    torch.manual_seed(0)
    model = AsterWhiteCore(cfg.backbone)
    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, cfg.backbone.vocab_size)
    assert torch.isfinite(logits).all()
    assert model._last_mtp_logits is not None and len(model._last_mtp_logits) == 2

    loss = logits.float().pow(2).mean() + sum(t.float().pow(2).mean() for t in model._last_mtp_logits)
    loss = loss + 0.01 * model.collect_moe_aux_loss()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no gradient: {missing}"

    out, _cache = model.generate(ids[:, :3], max_new_tokens=5, temperature=0.8)
    assert out.shape == (2, 8)

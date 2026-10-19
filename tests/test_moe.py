"""AsterWhite :: tests.test_moe -- core.moe.MoEFeedForward and its wiring
into HybridBlock/AsterWhiteCore via BackboneConfig.moe."""
from __future__ import annotations

import torch

from asterwhite.config import AsterWhiteConfig, validate_config
from asterwhite.core.autosizer import count_parameters
from asterwhite.core.layers import HybridBlock
from asterwhite.core.model import AsterWhiteCore
from asterwhite.core.moe import MoEFeedForward, load_balancing_loss


def test_moe_forward_shape_and_finite():
    moe = MoEFeedForward(d_model=16, num_experts=4, num_experts_per_tok=2, num_shared_experts=1, d_ff_expert=8)
    x = torch.randn(2, 5, 16)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_moe_every_parameter_gets_a_gradient_with_a_large_enough_batch():
    torch.manual_seed(0)
    moe = MoEFeedForward(d_model=16, num_experts=4, num_experts_per_tok=2, num_shared_experts=1, d_ff_expert=8)
    x = torch.randn(4, 16, 16, requires_grad=True)  # 64 tokens -- enough to route through every expert
    out = moe(x)
    loss = out.pow(2).mean() + 0.01 * moe.last_aux_loss
    loss.backward()
    missing = [n for n, p in moe.named_parameters() if p.grad is None]
    assert not missing, f"params with no gradient: {missing}"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_moe_aux_loss_is_minimized_at_perfect_balance():
    """Switch-Transformer-style load-balancing loss: exactly 1.0 at perfect
    balance (E experts, uniform 1/E routing), and strictly higher once
    routing is forced onto a single winner -- the property the aux loss
    exists to penalize."""
    torch.manual_seed(1)
    moe = MoEFeedForward(d_model=8, num_experts=4, num_experts_per_tok=1, num_shared_experts=0, d_ff_expert=4)
    x = torch.randn(1, 64, 8)

    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] += 10.0  # force everything onto expert 0
    moe(x)
    imbalanced = moe.last_aux_loss.item()

    with torch.no_grad():
        moe.router.weight.zero_()  # uniform routing
    moe(x)
    balanced = moe.last_aux_loss.item()

    assert abs(balanced - 1.0) < 1e-4
    assert imbalanced > balanced


def test_load_balancing_loss_standalone():
    # Router probs consistently concentrated on expert 0 (not just the
    # top-1 *selection* -- the aux loss weights by router PROBABILITY
    # mass, so a test needs concentrated probs, not just a concentrated
    # argmax, to actually exceed the balanced-case minimum of 1.0).
    router_probs = torch.zeros(10, 4)
    router_probs[:, 0] = 0.85
    router_probs[:, 1:] = 0.05
    selected = torch.zeros(10, 1, dtype=torch.long)  # every token's top-1 pick is expert 0
    loss = load_balancing_loss(router_probs, selected, num_experts=4)
    assert loss.item() > 1.0  # concentrated routing scores above the balanced minimum


def test_every_n_layers_alternates_dense_and_moe_mlps():
    moe_cfg = AsterWhiteConfig.tiny_for_testing().backbone.moe
    moe_cfg.enabled = True
    moe_cfg.every_n_layers = 2
    moe_cfg.num_experts = 4
    moe_cfg.num_experts_per_tok = 2
    moe_cfg.d_ff_expert = 8
    kinds = []
    for i in range(4):
        block = HybridBlock("A", d_model=16, n_heads=4, n_kv_heads=2, d_ff=32, max_seq_len=32,
                             rope_theta=10000.0, ssm_state_dim=4, ssm_conv_kernel=4, norm_eps=1e-5,
                             dropout=0.0, moe_cfg=moe_cfg, layer_index=i)
        kinds.append(type(block.mlp).__name__)
    assert kinds == ["MoEFeedForward", "SwiGLU", "MoEFeedForward", "SwiGLU"]


def test_moe_default_off_is_byte_identical_dense_swiglu():
    """BackboneConfig.moe.enabled defaults to False -- an AsterWhiteCore
    built from an untouched config must have plain SwiGLU MLPs everywhere,
    zero behavior change from before MoE existed."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    assert cfg.backbone.moe.enabled is False
    model = AsterWhiteCore(cfg.backbone)
    for block in model.blocks:
        assert type(block.mlp).__name__ == "SwiGLU"


def test_count_parameters_exactly_matches_real_moe_model():
    """Regression test for a real bug caught during development: the
    analytic sizer must count EVERY expert + the router + any shared
    experts, not a dense-SwiGLU approximation -- under-counting here would
    make guard_or_raise pass MoE configs that don't actually fit."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.moe.enabled = True
    cfg.backbone.moe.num_experts = 6
    cfg.backbone.moe.num_experts_per_tok = 2
    cfg.backbone.moe.num_shared_experts = 1
    cfg.backbone.moe.d_ff_expert = 20
    validate_config(cfg)
    predicted = count_parameters(cfg)
    actual = sum(p.numel() for p in AsterWhiteCore(cfg.backbone).parameters())
    assert predicted == actual


def test_moe_config_validation_rejects_bad_values():
    from asterwhite.config import AsterWhiteConfigError
    import pytest

    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.moe.enabled = True
    cfg.backbone.moe.num_experts = 4
    cfg.backbone.moe.num_experts_per_tok = 8  # more than num_experts -- invalid
    with pytest.raises(AsterWhiteConfigError):
        validate_config(cfg)

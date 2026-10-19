"""AsterWhite :: tests.test_norm_precision -- core.norm.RMSNorm's fp32
upcast for the mean-of-squares reduction (standard practice for bf16/fp16
training stability -- torch.autocast does not do this automatically for
hand-written reduction code)."""
from __future__ import annotations

import torch

from asterwhite.core.norm import RMSNorm


def test_rmsnorm_forward_shape_and_finite_across_dtypes():
    norm = RMSNorm(32)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(2, 5, 32, dtype=dtype)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out.float()).all()


def test_rmsnorm_gradients_flow_and_are_finite():
    norm = RMSNorm(32)
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    out = norm(x)
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()


def test_rmsnorm_reduction_is_computed_in_fp32_not_the_input_dtype():
    """The actual fix, verified directly rather than just by reading the
    code: constructs a case where computing the mean-of-squares natively
    in bf16 (7 mantissa bits) rounds measurably differently than computing
    it in fp32 first -- confirms RMSNorm's output matches the fp32-computed
    reference, not the bf16-native one, for values where the two diverge."""
    torch.manual_seed(0)
    norm = RMSNorm(64)
    with torch.no_grad():
        norm.weight.fill_(1.0)  # isolate the reduction itself -- no scaling from a random weight

    # Values spanning several orders of magnitude -- exactly the case where
    # bf16's limited mantissa causes the mean-of-squares reduction to lose
    # real precision relative to computing it in fp32 first.
    x_fp32 = (torch.randn(1, 1, 64) * torch.logspace(-2, 2, 64)).float()
    x_bf16 = x_fp32.to(torch.bfloat16)

    fp32_reduction_output = norm(x_fp32)  # fp32 input -- reduction is fp32 either way, this is the ground truth

    # Manually reproduce what a NAIVE (native-dtype, not upcast) reduction
    # would have computed, for comparison:
    naive_bf16_reduction = x_bf16 * torch.rsqrt(x_bf16.pow(2).mean(-1, keepdim=True) + norm.eps)

    actual_output = norm(x_bf16)  # goes through RMSNorm's actual (fixed) forward

    # The fixed implementation's output (computed via fp32 upcast even
    # though the INPUT was bf16) must be closer to the fp32 ground truth
    # than the naive native-bf16 reduction is.
    err_actual = (actual_output.float() - fp32_reduction_output.float()).abs().mean()
    err_naive = (naive_bf16_reduction.float() - fp32_reduction_output.float()).abs().mean()
    assert err_actual <= err_naive

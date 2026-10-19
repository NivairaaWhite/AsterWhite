import pytest

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.autosizer import classify_tier, count_parameters, estimate_memory, guard_or_raise
from asterwhite.core.device import DeviceManager
from asterwhite.utils.error_tracing import AsterWhiteError


def test_count_parameters_positive_and_scales_with_size():
    small = AsterWhiteConfig.tiny_for_testing()
    big = AsterWhiteConfig.tiny_for_testing()
    big.backbone.d_model = 64
    big.backbone.n_layers = 8
    assert count_parameters(small) > 0
    assert count_parameters(big) > count_parameters(small)


def _assert_exact_match(cfg) -> None:
    """The real ground truth: count_parameters(cfg) must equal
    sum(p.numel() for p in AsterWhiteCore(cfg.backbone).parameters())
    exactly, not approximately -- this is what lets guard_or_raise refuse
    a config BEFORE allocating it. A prior version of this formula silently
    under-counted every SSM layer's conv1d.bias and dt_proj.bias (128
    params/SSM-layer), caught only by this kind of exact-instantiation
    diff, not by 'is count_parameters positive and increasing' checks."""
    from asterwhite.core.model import AsterWhiteCore
    predicted = count_parameters(cfg)
    actual = sum(p.numel() for p in AsterWhiteCore(cfg.backbone).parameters())
    assert predicted == actual, f"predicted={predicted:,} actual={actual:,} diff={actual - predicted:,}"


def test_count_parameters_exact_match_dense_baseline():
    _assert_exact_match(AsterWhiteConfig.tiny_for_testing())


def test_count_parameters_exact_match_qk_norm():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.qk_norm = True
    _assert_exact_match(cfg)


def test_count_parameters_exact_match_mtp_heads():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.mtp_n_future_tokens = 3
    _assert_exact_match(cfg)


def test_count_parameters_exact_match_untied_embeddings():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.tie_word_embeddings = False
    _assert_exact_match(cfg)


def test_count_parameters_exact_match_odd_layer_pattern():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.layer_pattern = "AASSSA"
    cfg.backbone.n_layers = 7
    _assert_exact_match(cfg)


def test_count_parameters_exact_match_all_pure_ssm_and_all_pure_attention():
    cfg_ssm = AsterWhiteConfig.tiny_for_testing()
    cfg_ssm.backbone.layer_pattern = "S"
    _assert_exact_match(cfg_ssm)
    cfg_attn = AsterWhiteConfig.tiny_for_testing()
    cfg_attn.backbone.layer_pattern = "A"
    _assert_exact_match(cfg_attn)


def test_estimate_memory_tiny_config_fits_on_cpu():
    cfg = AsterWhiteConfig.tiny_for_testing()
    info = DeviceManager.resolve("cpu")
    est = estimate_memory(cfg, info)
    assert est.total_required_bytes > 0
    assert est.available_bytes > 0
    assert est.fits is True  # a KB-scale model must fit in any real machine's RAM


def test_estimate_memory_report_is_readable_string():
    cfg = AsterWhiteConfig.tiny_for_testing()
    info = DeviceManager.resolve("cpu")
    report = estimate_memory(cfg, info).report()
    assert "params=" in report and "fits=" in report


def test_guard_or_raise_passes_when_it_fits():
    cfg = AsterWhiteConfig.tiny_for_testing()
    info = DeviceManager.resolve("cpu")
    guard_or_raise(estimate_memory(cfg, info))  # must not raise


def test_guard_or_raise_refuses_oversized_without_override():
    cfg = AsterWhiteConfig.tiny_for_testing()
    info = DeviceManager.resolve("cpu")
    est = estimate_memory(cfg, info)
    est.total_required_bytes = est.available_bytes * 1000  # force an oversized estimate
    est.fits = False
    with pytest.raises(AsterWhiteError, match="exceed available memory"):
        guard_or_raise(est)
    guard_or_raise(est, override=True)  # must not raise with explicit override


def test_classify_tier_multi_device_dominates():
    info = DeviceManager.resolve("cpu")
    assert classify_tier(info, world_size=4) == classify_tier(info, world_size=4)  # deterministic
    from asterwhite.core.autosizer import HardwareTier
    assert classify_tier(info, world_size=4) == HardwareTier.TIER_4_MULTI_DEVICE


def test_classify_tier_cpu_single_process():
    info = DeviceManager.resolve("cpu")
    from asterwhite.core.autosizer import HardwareTier
    assert classify_tier(info, world_size=1) in (HardwareTier.TIER_0_CPU_TINY, HardwareTier.TIER_1_SMALL_OR_UNIFIED)

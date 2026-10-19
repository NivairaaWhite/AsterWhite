import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.training.lora import (
    LoRALinear, apply_lora, load_lora_adapter, mark_only_lora_as_trainable,
    merge_lora_weights, save_lora_adapter,
)


def _tiny_model():
    cfg = AsterWhiteConfig.tiny_for_testing()
    return AsterWhiteCore(cfg.backbone), cfg


def test_apply_lora_wraps_target_layers_only():
    model, _ = _tiny_model()
    n = apply_lora(model, target_substrings=("q_proj", "v_proj"), r=4, alpha=8)
    assert n > 0
    wrapped = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == n
    for w in wrapped:
        assert w.base.weight.requires_grad is False  # base frozen at wrap time


def test_mark_only_lora_trainable_freezes_everything_else():
    model, _ = _tiny_model()
    apply_lora(model, target_substrings=("q_proj",), r=4, alpha=8)
    mark_only_lora_as_trainable(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable)
    assert len(trainable) > 0


def test_forward_runs_with_lora_applied():
    model, cfg = _tiny_model()
    apply_lora(model, target_substrings=("q_proj", "v_proj"), r=4, alpha=8)
    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, cfg.backbone.vocab_size)


def test_lora_starts_as_a_no_op_delta():
    # lora_B is zero-initialized -> the very first forward pass must be
    # numerically IDENTICAL to the unwrapped model (delta is exactly 0).
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 6))
    before = model(ids)
    apply_lora(model, target_substrings=("q_proj",), r=4, alpha=8)
    after = model(ids)
    assert torch.allclose(before, after, atol=1e-5)


def test_save_load_adapter_round_trip(tmp_path):
    model, cfg = _tiny_model()
    apply_lora(model, target_substrings=("q_proj",), r=4, alpha=8)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.1)
    save_lora_adapter(model, str(tmp_path))

    fresh, _ = _tiny_model()
    apply_lora(fresh, target_substrings=("q_proj",), r=4, alpha=8)
    # Two independently-constructed models have different random BASE
    # weights regardless of LoRA state -- copy the base (non-LoRA) state
    # across so this test isolates the LoRA adapter round trip specifically,
    # not base-weight equality (which save_lora_adapter never claims to handle).
    base_sd = {k: v for k, v in model.state_dict().items() if "lora_A" not in k and "lora_B" not in k}
    fresh.load_state_dict(base_sd, strict=False)
    load_lora_adapter(fresh, str(tmp_path))

    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 6))
    out_original = model(ids)
    out_loaded = fresh(ids)
    assert torch.allclose(out_original, out_loaded, atol=1e-5)


def test_merge_lora_weights_produces_plain_linear_with_same_output():
    torch.manual_seed(2)
    model, cfg = _tiny_model()
    apply_lora(model, target_substrings=("q_proj",), r=4, alpha=8)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.1)

    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 6))
    before = model(ids)
    merged_count = merge_lora_weights(model)
    assert merged_count > 0
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    after = model(ids)
    assert torch.allclose(before, after, atol=1e-4)

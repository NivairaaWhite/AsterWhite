"""AsterWhite :: tests.test_trainer_frontier_features -- the training.trainer.Trainer
wiring added for activation checkpointing, torch.compile, fused AdamW, EMA, and the
MTP/MoE auxiliary training losses. Each is opt-in (default off/unchanged); these tests
cover both "still exactly the old behavior when unused" and "actually does the right
thing when turned on"."""
from __future__ import annotations

import copy

import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.training import Trainer


def _batch(cfg):
    return torch.randint(0, cfg.backbone.vocab_size, (cfg.training.micro_batch_size, cfg.backbone.max_seq_len))


def test_activation_checkpointing_gradients_match_uncheckpointed():
    """The whole point of activation checkpointing is IDENTICAL math, just
    recomputed instead of stored -- so with the same seed and the same
    input, gradients with and without it must match (within float
    tolerance), not merely 'both run without error'."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    torch.manual_seed(0)
    model_plain = AsterWhiteCore(cfg.backbone)
    torch.manual_seed(0)
    model_ckpt = AsterWhiteCore(cfg.backbone)
    # constructed identically (same seed) -- sanity-check before proceeding
    for (n1, p1), (n2, p2) in zip(model_plain.named_parameters(), model_ckpt.named_parameters()):
        assert torch.equal(p1, p2), f"{n1} differs before any training -- seeds didn't line up"

    model_ckpt.activation_checkpointing = True
    model_plain.train()
    model_ckpt.train()

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 16))

    logits_plain = model_plain(ids)
    logits_plain.float().pow(2).mean().backward()
    logits_ckpt = model_ckpt(ids)
    logits_ckpt.float().pow(2).mean().backward()

    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(model_plain.named_parameters(), model_ckpt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), f"gradient mismatch for {n1}"


def test_activation_checkpointing_default_off_and_never_engages_with_a_cache():
    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone)
    assert model.activation_checkpointing is False  # default, set explicitly by Trainer only
    # Even if turned on, generation (which always passes a cache) must
    # never route through checkpoint() -- see forward_hidden's docstring
    # for why (cache mutation + checkpoint's double-forward would corrupt it).
    model.activation_checkpointing = True
    ids = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    out, cache = model.generate(ids, max_new_tokens=3)
    assert out.shape == (1, 7)


def test_trainer_wires_activation_checkpointing_flag_onto_raw_model():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.activation_checkpointing = True
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    assert trainer.raw_model.activation_checkpointing is True


def test_ema_updates_after_successful_step_and_is_smoother_than_raw():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.ema_decay = 0.9  # low decay so the effect is measurable in a few steps
    torch.manual_seed(0)
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    assert trainer.ema_state is not None
    ema_initial = copy.deepcopy(trainer.ema_state)

    for _ in range(5):
        trainer.train_step([_batch(cfg)])

    changed = sum(1 for k in ema_initial if not torch.equal(ema_initial[k], trainer.ema_state[k]))
    assert changed > 0
    # EMA must never equal the (rapidly-changing) raw weights exactly after
    # multiple real gradient steps with decay < 1 -- it's a lagging average.
    raw_sd = trainer.raw_model.state_dict()
    identical = all(torch.equal(trainer.ema_state[k], raw_sd[k]) for k in raw_sd)
    assert not identical


def test_ema_disabled_by_default():
    cfg = AsterWhiteConfig.tiny_for_testing()
    assert cfg.training.ema_decay is None
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    assert trainer.ema_state is None
    assert trainer.ema_state_dict() is None


def test_ema_round_trips_through_checkpoint(tmp_path):
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.ema_decay = 0.95
    torch.manual_seed(0)
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="h1")
    for _ in range(3):
        trainer.train_step([_batch(cfg)])

    trainer.save_checkpoint(str(tmp_path), config_dict=cfg.to_dict())

    trainer2 = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="h1")
    trainer2.load_checkpoint(str(tmp_path))
    assert trainer2.ema_state is not None
    assert all(torch.equal(trainer2.ema_state[k], trainer.ema_state[k]) for k in trainer.ema_state)


def test_fused_adamw_requested_never_crashes_construction():
    """fused=True is only actually usable on CUDA-like devices; on CPU it
    must transparently fall back to standard AdamW rather than raising."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.optimizer_fused = True
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    # must still be usable for a real step
    stats = trainer.train_step([_batch(cfg)])
    assert stats.step == 1


def test_compile_model_on_cpu_is_a_safe_logged_noop_not_a_hang_or_crash():
    """torch.compile is lazy (compiles on first forward, not at wrap time),
    so it is deliberately only attempted on CUDA -- see Trainer.__init__.
    On CPU this must construct quickly and train normally, uncompiled."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.compile_model = True
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    stats = trainer.train_step([_batch(cfg)])
    assert stats.step == 1
    assert not isinstance(trainer.model, torch._dynamo.OptimizedModule) if hasattr(torch, "_dynamo") else True


def test_mtp_loss_is_included_in_training_but_not_in_eval_loss():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.mtp_n_future_tokens = 2
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    micro = _batch(cfg)

    train_loss, train_breakdown = trainer._forward_loss(micro, include_aux_losses=True)
    eval_loss, eval_breakdown = trainer._forward_loss(micro, include_aux_losses=False)

    assert train_breakdown["mtp_loss"] != 0.0
    assert eval_breakdown["mtp_loss"] == 0.0
    # eval must be pure next-token CE, not entangled with the aux term
    assert abs(eval_loss.item() - eval_breakdown["main_loss"]) < 1e-5


def test_moe_aux_loss_is_included_in_training_but_not_in_eval_loss():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.moe.enabled = True
    cfg.backbone.moe.num_experts = 4
    cfg.backbone.moe.num_experts_per_tok = 2
    cfg.backbone.moe.d_ff_expert = 16
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    micro = _batch(cfg)

    _train_loss, train_breakdown = trainer._forward_loss(micro, include_aux_losses=True)
    _eval_loss, eval_breakdown = trainer._forward_loss(micro, include_aux_losses=False)

    assert train_breakdown["moe_aux_loss"] != 0.0
    assert eval_breakdown["moe_aux_loss"] == 0.0


def test_train_step_reports_loss_breakdown_in_stats():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.mtp_n_future_tokens = 1
    cfg.backbone.moe.enabled = True
    cfg.backbone.moe.num_experts = 4
    cfg.backbone.moe.num_experts_per_tok = 2
    cfg.backbone.moe.d_ff_expert = 16
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    stats = trainer.train_step([_batch(cfg)])
    assert stats.main_loss > 0
    assert stats.mtp_loss > 0
    assert stats.moe_aux_loss > 0
    assert abs(stats.loss - (stats.main_loss + cfg.training.mtp_loss_coef * stats.mtp_loss
                              + cfg.backbone.moe.aux_loss_coef * stats.moe_aux_loss)) < 1e-3

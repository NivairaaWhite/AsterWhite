import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.training import Trainer
from asterwhite.utils.error_tracing import AsterWhiteError


def _batch(cfg):
    return torch.randint(0, cfg.backbone.vocab_size, (2, 12))


def test_evaluate_reports_finite_loss_and_perplexity():
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    stats = trainer.evaluate([_batch(cfg), _batch(cfg)])
    assert torch.isfinite(torch.tensor(stats.val_loss))
    assert stats.perplexity > 0


def test_evaluate_tracks_best_val_loss_across_calls():
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    s1 = trainer.evaluate([_batch(cfg)])
    assert trainer.best_val_loss == s1.val_loss
    trainer.best_val_loss = -999.0  # force a value no real loss beats
    trainer.evaluate([_batch(cfg)])
    assert trainer.best_val_loss == -999.0  # unchanged: s2.val_loss can't beat it


def test_evaluate_restores_train_mode():
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    trainer.model.train()
    trainer.evaluate([_batch(cfg)])
    assert trainer.model.training is True


def test_checkpoint_resume_preserves_weights_and_tying(tmp_path):
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="h1")
    for _ in range(2):
        trainer.train_step([_batch(cfg)])
    trainer.save_checkpoint(str(tmp_path), config_dict=cfg.to_dict())

    resumed = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="h1")
    resumed.load_checkpoint(str(tmp_path))
    assert resumed.step_count == trainer.step_count
    for (n1, p1), (n2, p2) in zip(trainer.raw_model.named_parameters(), resumed.raw_model.named_parameters()):
        assert torch.allclose(p1, p2), n1
    assert resumed.raw_model.tok_emb.weight.data_ptr() == resumed.raw_model.lm_head.weight.data_ptr()


def test_checkpoint_resume_refuses_tokenizer_mismatch(tmp_path):
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="h1")
    trainer.train_step([_batch(cfg)])
    trainer.save_checkpoint(str(tmp_path), config_dict=cfg.to_dict())

    mismatched = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash="DIFFERENT")
    try:
        mismatched.load_checkpoint(str(tmp_path))
        assert False, "should have refused"
    except AsterWhiteError:
        pass


def test_nonfinite_step_is_skipped_not_crashed():
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"))
    orig = trainer._forward_loss
    call_count = {"n": 0}

    def poisoned(micro, include_aux_losses=False):
        call_count["n"] += 1
        loss, breakdown = orig(micro, include_aux_losses=include_aux_losses)
        return (loss * float("nan"), breakdown) if call_count["n"] == 1 else (loss, breakdown)

    trainer._forward_loss = poisoned
    step_before = trainer.step_count
    stats = trainer.train_step([_batch(cfg)])
    assert stats.skipped_nonfinite is True
    assert trainer.step_count == step_before + 1  # step counter still advances; the optimizer update was skipped


def test_repeated_nonfinite_steps_eventually_raise():
    cfg = AsterWhiteConfig.tiny_for_testing()
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), max_consecutive_nonfinite=3)
    empty_breakdown = {"main_loss": 0.0, "mtp_loss": 0.0, "moe_aux_loss": 0.0}
    trainer._forward_loss = lambda micro, include_aux_losses=False: (torch.tensor(float("nan")), empty_breakdown)
    raised = False
    try:
        for _ in range(10):
            trainer.train_step([_batch(cfg)])
    except AsterWhiteError:
        raised = True
    assert raised

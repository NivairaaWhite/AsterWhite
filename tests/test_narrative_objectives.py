import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.narrative.revision import QualityScorer
from asterwhite.narrative.style_consistency import StyleConsistencyChecker
from asterwhite.training.narrative_objectives import (
    NarrativeLossWeights, NarrativeObjectiveHeads, beat_order_loss, character_identity_loss,
    episodic_retrieval_loss, plan_consistency_loss, pooled_hidden_states,
    relationship_consistency_loss, revision_ranking_loss, safety_policy_loss,
    style_control_loss, timeline_consistency_loss,
)


def _setup():
    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone)
    heads = NarrativeObjectiveHeads(cfg.backbone.d_model, plan_dim=16, character_dim=16,
                                     world_state_dim=16, max_plan_beats=8)
    return cfg, model, heads


def test_pooled_hidden_states_carries_gradient():
    cfg, model, _ = _setup()
    ids = torch.randint(0, cfg.backbone.vocab_size, (3, 5))
    pooled = pooled_hidden_states(model, ids)
    assert pooled.shape == (3, cfg.backbone.d_model)
    assert pooled.requires_grad


def test_all_losses_are_finite_and_reduce_to_scalars():
    cfg, model, heads = _setup()
    B, T = 5, 6
    ids = torch.randint(0, cfg.backbone.vocab_size, (B, T))
    pooled = pooled_hidden_states(model, ids)

    beats = torch.randn(B, 16)
    beat_idx = torch.randint(0, 8, (B,))
    char_vecs = torch.randn(B, 16)
    sa, sb = torch.randn(B, 16), torch.randn(B, 16)
    target_aff = torch.rand(B) * 2 - 1
    a_before_b = torch.randint(0, 2, (B,)).float()
    mem_bank = torch.randn(20, 16)
    correct_idx = torch.randint(0, 20, (B,))
    safety_labels = torch.randint(0, 2, (B,)).float()

    losses = {
        "plan_consistency": plan_consistency_loss(heads, beats, pooled),
        "beat_order": beat_order_loss(heads, pooled, beat_idx),
        "character_identity": character_identity_loss(heads, char_vecs, pooled),
        "relationship_consistency": relationship_consistency_loss(heads, sa, sb, target_aff),
        "timeline_consistency": timeline_consistency_loss(heads, pooled, pooled.flip(0), a_before_b),
        "episodic_retrieval": episodic_retrieval_loss(char_vecs, mem_bank, correct_idx),
        "safety_policy": safety_policy_loss(heads, pooled, safety_labels),
    }
    for name, loss in losses.items():
        assert loss.dim() == 0, f"{name} did not reduce to a scalar"
        assert torch.isfinite(loss), f"{name} is not finite: {loss}"


def test_gradients_reach_both_backbone_and_heads():
    cfg, model, heads = _setup()
    B, T = 4, 5
    ids = torch.randint(0, cfg.backbone.vocab_size, (B, T))
    pooled = pooled_hidden_states(model, ids)
    beats = torch.randn(B, 16)
    beat_idx = torch.randint(0, 8, (B,))

    total = plan_consistency_loss(heads, beats, pooled) + beat_order_loss(heads, pooled, beat_idx)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads.parameters())


def test_revision_ranking_loss_prefers_higher_revised_score():
    scorer = QualityScorer(d_model=16)
    draft = scorer(torch.randn(5, 16))
    revised_better = draft.detach() + 1.0  # unambiguously higher
    loss_good = revision_ranking_loss(draft.detach(), revised_better)
    loss_bad = revision_ranking_loss(draft.detach(), draft.detach() - 1.0)  # revised WORSE than draft
    assert loss_good < loss_bad


def test_style_control_loss_requires_centroid_established_first():
    # Documented gotcha (found during development): drift_score returns a
    # constant 1.0 with NO gradient until update_centroid has been called
    # at least once -- verify both the bootstrap state and the real state.
    checker = StyleConsistencyChecker(style_dim=16, vocab_size=256)
    ids = torch.randint(0, 256, (3, 8))

    bootstrap_drift = checker.drift_score(ids)
    assert bootstrap_drift.requires_grad is False
    assert style_control_loss(bootstrap_drift).item() == 0.0

    checker.update_centroid(torch.randint(0, 256, (2, 10)))
    real_drift = checker.drift_score(ids)
    assert real_drift.requires_grad is True
    loss = style_control_loss(real_drift)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in checker.parameters())


def test_loss_weights_round_trip():
    w = NarrativeLossWeights(plan_consistency=2.0, safety_policy=0.1)
    restored = NarrativeLossWeights.from_dict(w.to_dict())
    assert restored == w

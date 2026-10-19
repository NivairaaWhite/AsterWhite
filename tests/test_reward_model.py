import torch

from asterwhite.alignment.grpo import GRPOConfig, GRPOTrainer
from asterwhite.alignment.reward_model import (
    HeuristicRewardComposite, HeuristicRewardWeights, RewardModel,
    bradley_terry_loss, train_reward_model_step,
)
from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.narrative.style_consistency import OriginalityGuard, StyleConsistencyChecker


def test_heuristic_reward_components_are_individually_inspectable():
    sc = StyleConsistencyChecker(style_dim=16, vocab_size=256)
    sc.update_centroid(torch.randint(0, 256, (2, 12)))
    reward_fn = HeuristicRewardComposite(sc, weights=HeuristicRewardWeights(target_length_tokens=8))
    prompt = torch.randint(0, 256, (5,))
    completion = torch.randint(0, 256, (8,))
    components = reward_fn.score_components(prompt, completion)
    assert set(components) == {"style_consistency", "originality", "length_penalty", "repetition_penalty"}
    r = reward_fn(prompt, completion)
    assert torch.isfinite(r)


def test_heuristic_reward_penalizes_repetition():
    sc = StyleConsistencyChecker(style_dim=16, vocab_size=256)
    sc.update_centroid(torch.randint(0, 256, (2, 12)))
    reward_fn = HeuristicRewardComposite(sc)
    varied = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    repetitive = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    prompt = torch.randint(0, 256, (3,))
    assert reward_fn(prompt, varied) > reward_fn(prompt, repetitive)


def test_heuristic_reward_uses_originality_guard_when_given():
    sc = StyleConsistencyChecker(style_dim=16, vocab_size=256)
    sc.update_centroid(torch.randint(0, 256, (2, 12)))
    og = OriginalityGuard(n=4)
    memorized = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    og.add(memorized)
    reward_fn = HeuristicRewardComposite(sc, og)
    prompt = torch.randint(0, 256, (3,))
    components = reward_fn.score_components(prompt, torch.tensor(memorized))
    assert components["originality"] < 1.0  # exact n-gram overlap with something already seen


def test_reward_model_training_step_reduces_bradley_terry_loss():
    torch.manual_seed(0)
    rm = RewardModel(d_model=16)
    opt = torch.optim.Adam(rm.parameters(), lr=1e-2)
    chosen = torch.randn(8, 16)
    rejected = torch.randn(8, 16)
    losses = [train_reward_model_step(rm, opt, chosen, rejected) for _ in range(30)]
    assert losses[-1] < losses[0]  # should learn to separate chosen from rejected on a fixed batch


def test_bradley_terry_loss_prefers_correctly_ordered_rewards():
    correct = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    incorrect = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert correct < incorrect


def test_reward_model_save_load_round_trip_and_tokenizer_hash_gate(tmp_path):
    rm = RewardModel(d_model=16)
    rm.save(str(tmp_path), tokenizer_hash="hash-abc")
    reloaded = RewardModel.load(str(tmp_path), expected_tokenizer_hash="hash-abc")
    x = torch.randn(4, 16)
    assert torch.allclose(rm(x), reloaded(x))

    import pytest
    from asterwhite.utils.error_tracing import AsterWhiteError
    with pytest.raises(AsterWhiteError):
        RewardModel.load(str(tmp_path), expected_tokenizer_hash="wrong-hash")


def test_grpo_clips_rewards_before_advantage():
    cfg = AsterWhiteConfig.tiny_for_testing()
    policy = AsterWhiteCore(cfg.backbone)
    grpo_cfg = GRPOConfig(group_size=4, max_new_tokens=4, reward_clip=1.0)

    def huge_reward_fn(prompt_ids, completion_ids):
        return torch.tensor(1000.0)  # far beyond reward_clip=1.0

    trainer = GRPOTrainer(policy, huge_reward_fn, grpo_cfg, device=torch.device("cpu"))
    prompt_ids = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    stats = trainer.step(prompt_ids)
    assert stats.mean_reward <= grpo_cfg.reward_clip + 1e-4
    assert stats.clip_fraction == 1.0  # every one of the group's rewards was clipped

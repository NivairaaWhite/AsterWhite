"""
AsterWhite :: alignment.reward_model

Spec 12 asks for "a versioned reward model OR an auditable preference
signal" -- this module ships both, because they serve different points in
a project's lifecycle:

1. HeuristicRewardComposite -- auditable, deterministic, needs no
   preference data. Combines StyleConsistencyChecker.drift_score (voice
   consistency), OriginalityGuard.originality_ratio (memorization
   penalty), a length-target term, and a repetition penalty into one
   scalar, with configurable weights every term of which is inspectable.
   This is what train.py's `align` subcommand uses out of the box now,
   replacing the previous single-line `style_checker.drift_score`
   placeholder.

2. RewardModel -- a small trainable head, same pooled-hidden-state
   pattern as narrative.revision.QualityScorer, trained via the standard
   Bradley-Terry pairwise preference loss. Real, versioned (ties to a
   tokenizer hash and a schema version, saved through registry.checkpoint
   so it's atomic/checksummed like every other checkpoint here), and
   completely untrained out of the box -- there is no preference dataset
   bundled with this repository. Train it on your own preference pairs
   (human-labeled or AI-labeled) before relying on it instead of, or
   blended with, the heuristic composite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..narrative.style_consistency import OriginalityGuard, StyleConsistencyChecker
from ..registry.checkpoint import load_checkpoint, save_checkpoint
from ..utils.error_tracing import get_logger

logger = get_logger("asterwhite.alignment.reward_model")

REWARD_MODEL_SCHEMA_VERSION = 1

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (prompt_ids, completion_ids) -> scalar tensor


@dataclass
class HeuristicRewardWeights:
    style_consistency: float = 1.0
    originality: float = 1.0
    length_target: float = 0.3
    repetition_penalty: float = 0.5
    target_length_tokens: int = 256

    def to_dict(self):
        return asdict(self)


class HeuristicRewardComposite:
    """Auditable default reward. Every term is a plain method you can call
    independently to see exactly why a completion scored the way it did --
    intentionally NOT a black box, since this is what a project uses
    before it has any real preference-labeled data."""

    def __init__(self, style_checker: StyleConsistencyChecker, originality_guard: Optional[OriginalityGuard] = None,
                 weights: Optional[HeuristicRewardWeights] = None):
        self.style_checker = style_checker
        self.originality_guard = originality_guard
        self.weights = weights or HeuristicRewardWeights()

    def _repetition_penalty(self, completion_ids: torch.Tensor) -> float:
        ids = completion_ids.tolist()
        if len(ids) < 4:
            return 0.0
        bigrams = list(zip(ids, ids[1:]))
        distinct_ratio = len(set(bigrams)) / max(1, len(bigrams))
        return 1.0 - distinct_ratio  # 0 = fully distinct, 1 = maximally repetitive

    def _length_penalty(self, completion_ids: torch.Tensor) -> float:
        n = completion_ids.shape[0]
        target = self.weights.target_length_tokens
        return abs(n - target) / max(1, target)

    def score_components(self, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> dict:
        """Every raw term BEFORE weighting -- log this, not just the
        combined scalar, for a genuinely auditable trail."""
        style = self.style_checker.drift_score(completion_ids.unsqueeze(0)).squeeze(0).item()
        originality = (self.originality_guard.originality_ratio(completion_ids.tolist())
                        if self.originality_guard is not None else 1.0)
        return {
            "style_consistency": style,
            "originality": originality,
            "length_penalty": self._length_penalty(completion_ids),
            "repetition_penalty": self._repetition_penalty(completion_ids),
        }

    def __call__(self, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
        c = self.score_components(prompt_ids, completion_ids)
        w = self.weights
        total = (w.style_consistency * c["style_consistency"]
                 + w.originality * c["originality"]
                 - w.length_target * c["length_penalty"]
                 - w.repetition_penalty * c["repetition_penalty"])
        return torch.tensor(total, dtype=torch.float32, device=completion_ids.device)


class RewardModel(nn.Module):
    """Trainable scalar reward head. Shares the pooled-hidden-state
    pattern used throughout this repo (narrative.revision.QualityScorer,
    training.narrative_objectives) rather than introducing a new one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(pooled_hidden).squeeze(-1)

    def save(self, directory: str, tokenizer_hash: Optional[str] = None) -> None:
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        metadata = {"schema_version": REWARD_MODEL_SCHEMA_VERSION, "tokenizer_hash": tokenizer_hash,
                    "d_model": self.head[0].in_features}
        save_checkpoint(directory, "reward_model", tensors, metadata, make_latest=True)

    @classmethod
    def load(cls, directory: str, expected_tokenizer_hash: Optional[str] = None) -> "RewardModel":
        tensors, metadata = load_checkpoint(directory, which="latest", expected_tokenizer_hash=expected_tokenizer_hash)
        if metadata.get("schema_version") != REWARD_MODEL_SCHEMA_VERSION:
            raise ValueError(f"reward model schema_version mismatch: {metadata.get('schema_version')}")
        model = cls(d_model=metadata["d_model"])
        model.load_state_dict(tensors)
        return model


def bradley_terry_loss(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> torch.Tensor:
    """Standard pairwise preference loss (Bradley & Terry, 1952 / used by
    every modern RLHF reward model): P(chosen preferred) =
    sigmoid(chosen_reward - rejected_reward), maximize its log-likelihood.
    chosen_reward, rejected_reward: (B,), from RewardModel(pooled_hidden)
    on a batch of (chosen, rejected) completion pairs for the SAME prompt."""
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def train_reward_model_step(reward_model: RewardModel, optimizer: torch.optim.Optimizer,
                             chosen_pooled: torch.Tensor, rejected_pooled: torch.Tensor) -> float:
    """One gradient step on a batch of preference pairs. Real, correct,
    tested (see tests/test_reward_model.py) -- but note there is no
    preference DATA shipped here; chosen_pooled/rejected_pooled must come
    from your own labeled (chosen, rejected) completion pairs, pooled the
    same way as training.narrative_objectives.pooled_hidden_states."""
    optimizer.zero_grad(set_to_none=True)
    loss = bradley_terry_loss(reward_model(chosen_pooled), reward_model(rejected_pooled))
    loss.backward()
    optimizer.step()
    return loss.item()

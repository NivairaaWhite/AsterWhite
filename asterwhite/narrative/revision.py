"""
AsterWhite :: narrative.revision

Multi-pass self-revision: structural -> content -> line -> polish, each
pass a conditioned generation call against the SAME AsterWhiteCore backbone
(a learned per-pass conditioning tag as hidden_prefix), gated by a
lightweight scoring head so passes stop once a chunk is good enough instead
of always running the full ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from ..core.model import AsterWhiteCore

PASS_NAMES = [
    "structural",  # chapter/scene level: pacing, plot holes, arc consistency
    "content",     # paragraph level: character motivation, world-state consistency
    "line",        # sentence level: prose rhythm, word choice, dialogue naturalism
    "polish",      # final pass: typos, redundancy, tightening
]


@dataclass
class RevisionResult:
    pass_name: str
    token_ids: torch.Tensor
    quality_score: float


class QualityScorer(nn.Module):
    """Lightweight scoring head on pooled backbone hidden states, used to
    decide whether another revision pass is warranted."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(pooled_hidden).squeeze(-1)


class PassTagEmbedding(nn.Module):
    """Trainable per-pass conditioning prefix. Swap for a real instruction
    encoder (text -> hidden_prefix through the shared tokenizer/backbone)
    once instruction-tuned data exists; this keeps the module runnable
    standalone in the meantime."""

    def __init__(self, d_model: int, prefix_len: int = 4, n_passes: int = len(PASS_NAMES)):
        super().__init__()
        self.tags = nn.Parameter(torch.randn(n_passes, prefix_len, d_model) * 0.02)

    def forward(self, pass_index: int, batch_size: int) -> torch.Tensor:
        return self.tags[pass_index].unsqueeze(0).expand(batch_size, -1, -1)


class HierarchicalRevisionSuite:
    """Drives up to `max_passes` revision passes over a draft, stopping
    early once QualityScorer clears `accept_threshold`."""

    def __init__(self, model: AsterWhiteCore, scorer: QualityScorer, tag_embedding: PassTagEmbedding,
                 max_passes: int = 3, accept_threshold: float = 0.85):
        self.model = model
        self.scorer = scorer
        self.tag_embedding = tag_embedding
        self.max_passes = min(max_passes, len(PASS_NAMES))
        self.accept_threshold = accept_threshold

    def _pooled_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.tok_emb(token_ids)
        for block in self.model.blocks:
            x = block(x)
        return self.model.norm_f(x).mean(dim=1)

    @torch.no_grad()
    def revise(self, draft_ids: torch.Tensor, max_new_tokens: int = 256,
               temperature: float = 0.8) -> List[RevisionResult]:
        history: List[RevisionResult] = []
        current = draft_ids
        for i in range(self.max_passes):
            tag = self.tag_embedding(i, current.shape[0]).to(current.device)
            revised, _ = self.model.generate(
                current, max_new_tokens=max_new_tokens, temperature=temperature, hidden_prefix=tag,
            )
            score = self.scorer(self._pooled_hidden(revised)).mean().item()
            history.append(RevisionResult(pass_name=PASS_NAMES[i], token_ids=revised, quality_score=score))
            current = revised
            if score >= self.accept_threshold:
                break
        return history

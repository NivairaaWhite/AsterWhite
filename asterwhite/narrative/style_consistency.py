"""
AsterWhite :: narrative.style_consistency

Two real, complementary checks bundled as "authorship verification":

1. StyleConsistencyChecker -- embeds a prose chunk and compares it (cosine
   similarity) against a running centroid of the target author/voice, so
   drift across a thousand-page manuscript gets flagged instead of quietly
   accumulating.
2. OriginalityGuard -- a rolling n-gram hash index over the training
   corpus; flags generated spans that overlap it too closely, as a
   safeguard against verbatim memorized reproduction rather than genuine
   generation. Built incrementally from the same streaming source
   data_pipeline.py uses, so it scales to GB/TB corpora without ever
   holding the whole thing in memory.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleConsistencyChecker(nn.Module):
    def __init__(self, style_dim: int = 256, vocab_size: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.style_dim = style_dim
        self.emb = nn.Embedding(vocab_size, style_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=style_dim, nhead=n_heads, dim_feedforward=style_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("centroid", torch.zeros(style_dim), persistent=True)
        self.register_buffer("centroid_count", torch.zeros(1), persistent=True)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) -> (B, style_dim) mean-pooled style embedding."""
        return self.encoder(self.emb(token_ids)).mean(dim=1)

    @torch.no_grad()
    def update_centroid(self, token_ids: torch.Tensor) -> None:
        """Fold a confirmed-on-voice reference chunk into the running
        centroid -- call this on the author's own sample chapters, or on
        earlier chapters already accepted during this manuscript."""
        vec = self.embed(token_ids).mean(dim=0)
        n = self.centroid_count.item()
        self.centroid.copy_((self.centroid * n + vec * token_ids.shape[0]) / (n + token_ids.shape[0]))
        self.centroid_count += token_ids.shape[0]

    def drift_score(self, token_ids: torch.Tensor) -> torch.Tensor:
        """1.0 = perfectly on-voice, -1.0 = maximally divergent. (B,) tensor."""
        vec = self.embed(token_ids)
        if self.centroid_count.item() == 0:
            return torch.ones(vec.shape[0], device=vec.device)
        return F.cosine_similarity(vec, self.centroid.unsqueeze(0).expand_as(vec), dim=-1)


class OriginalityGuard:
    """Rolling n-gram hash index over the training corpus. Not a neural
    module -- pure Python/set lookups, so it costs nothing extra during
    generation and scales to arbitrarily large corpora when backed by an
    on-disk hash index for TB-scale corpora."""

    def __init__(self, n: int = 8):
        self.n = n
        self._index: Set[int] = set()

    @staticmethod
    def _ngram_hashes(token_ids: Sequence[int], n: int) -> Iterable[int]:
        for i in range(len(token_ids) - n + 1):
            yield hash(tuple(token_ids[i:i + n]))

    def add(self, token_ids: Sequence[int]) -> None:
        self._index.update(self._ngram_hashes(token_ids, self.n))

    def overlap_spans(self, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
        """(start, end) index ranges of generated text whose n-grams match
        the training corpus -- candidates for near-verbatim reproduction
        rather than original prose."""
        spans: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        i = 0
        for i, h in enumerate(self._ngram_hashes(token_ids, self.n)):
            hit = h in self._index
            if hit and run_start is None:
                run_start = i
            elif not hit and run_start is not None:
                spans.append((run_start, i + self.n - 1))
                run_start = None
        if run_start is not None:
            spans.append((run_start, len(token_ids)))
        return spans

    def originality_ratio(self, token_ids: Sequence[int]) -> float:
        """Fraction of the sequence NOT covered by any overlap span. 1.0 = fully original."""
        if len(token_ids) < self.n:
            return 1.0
        covered = sum(end - start for start, end in self.overlap_spans(token_ids))
        return max(0.0, 1.0 - covered / len(token_ids))

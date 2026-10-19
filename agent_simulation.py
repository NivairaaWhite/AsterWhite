"""
AsterWhite :: agent_simulation

Agentic Character Simulator & Interaction Engine: each character is an
independent Agent wrapping a CharacterProfile plus a dual-tier memory --

  - WorkingMemory:  fixed-size, fast, recency-ordered (a ring buffer of
                    recent event embeddings) -- "what just happened."
  - EpisodicMemory: bounded-but-large vectorized store of every past event
                    an agent witnessed or took part in, retrievable by
                    similarity search -- "what happened, ever," so a
                    betrayal from 400 pages ago can still surface when
                    relevant even after it's scrolled out of working memory.

AgentSimulationLoop steps multiple Agents through a scene: each perceives
the others' last actions, retrieves relevant memory, and generates its next
action/dialogue line conditioned on its own profile + memory + goals --
characters that act from their own state rather than being puppeted line by
line from one authorial pass.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asterwhite.core.model import AsterWhiteCore
from asterwhite.narrative.character import CharacterEncoder, CharacterProfile
from asterwhite.utils.error_tracing import get_logger, trace_errors

logger = get_logger("asterwhite.agent_simulation")


@dataclass
class MemoryRecord:
    embedding: torch.Tensor
    text: str
    tick: int
    tags: List[str] = field(default_factory=list)  # e.g. ["betrayal", "c_joran"]


class WorkingMemory:
    """Fast tier: fixed-size, recency-ordered. O(1) push/evict."""

    def __init__(self, slots: int = 64):
        self.slots = slots
        self._buf: Deque[MemoryRecord] = deque(maxlen=slots)

    def add(self, record: MemoryRecord) -> None:
        self._buf.append(record)

    def recent(self, n: Optional[int] = None) -> List[MemoryRecord]:
        items = list(self._buf)
        return items[-n:] if n is not None else items

    def __len__(self) -> int:
        return len(self._buf)


class EpisodicMemoryStore:
    """High-capacity tier: every event an agent witnessed, retrievable by
    embedding similarity even long after it left working memory. Bounded at
    `capacity` (oldest evicted first) so memory cost stays flat over an
    arbitrarily long narrative instead of growing without limit -- a plain
    stacked-tensor + cosine-similarity search here; swap for an ANN index
    (FAISS, ScaNN) behind the same `query()` signature if capacity needs to
    grow past what brute-force search comfortably covers."""

    def __init__(self, capacity: int = 100_000, dim: int = 256):
        self.capacity = capacity
        self.dim = dim
        self._records: Deque[MemoryRecord] = deque(maxlen=capacity)

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def query(self, query_embedding: torch.Tensor, top_k: int = 5,
              tag_filter: Optional[str] = None) -> List[Tuple[MemoryRecord, float]]:
        """Cosine-similarity retrieval over the full episodic store -- the
        mechanism that lets a betrayal from hundreds of scenes ago surface
        again the moment it's contextually relevant."""
        candidates = [r for r in self._records if (tag_filter is None or tag_filter in r.tags)]
        if not candidates:
            return []
        mat = torch.stack([r.embedding for r in candidates], dim=0)
        sims = F.cosine_similarity(mat, query_embedding.unsqueeze(0).expand_as(mat), dim=-1)
        k = min(top_k, len(candidates))
        top_vals, top_idx = torch.topk(sims, k)
        return [(candidates[i], top_vals[j].item()) for j, i in enumerate(top_idx.tolist())]


class DualTierMemory:
    """Combines WorkingMemory + EpisodicMemoryStore behind one interface:
    every event is written to both tiers; recall prefers working-memory
    recency but falls back to episodic similarity search for anything that
    has scrolled out of the fast tier."""

    def __init__(self, working_slots: int = 64, episodic_capacity: int = 100_000, dim: int = 256):
        self.working = WorkingMemory(working_slots)
        self.episodic = EpisodicMemoryStore(episodic_capacity, dim)

    def witness(self, embedding: torch.Tensor, text: str, tick: int, tags: Optional[List[str]] = None) -> None:
        record = MemoryRecord(embedding=embedding.detach(), text=text, tick=tick, tags=tags or [])
        self.working.add(record)
        self.episodic.add(record)

    def recall(self, query_embedding: torch.Tensor, top_k: int = 5,
               tag_filter: Optional[str] = None) -> List[MemoryRecord]:
        recent_matches = [r for r in self.working.recent() if tag_filter is None or tag_filter in r.tags]
        seen = {id(r) for r in recent_matches}
        episodic_hits = self.episodic.query(query_embedding, top_k=top_k, tag_filter=tag_filter)
        combined = recent_matches + [r for r, _ in episodic_hits if id(r) not in seen]
        return combined[:top_k]


class Agent:
    """One simulated character: profile + dual-tier memory + the shared
    backbone/encoder it thinks and speaks through. Agents act independently
    of any single authorial narration path -- each decides its own next
    line from its own state."""

    def __init__(self, profile: CharacterProfile, model: AsterWhiteCore, encoder: CharacterEncoder,
                 working_slots: int = 64, episodic_capacity: int = 100_000):
        self.profile = profile
        self.model = model
        self.encoder = encoder
        self.memory = DualTierMemory(working_slots, episodic_capacity, dim=encoder.character_dim)
        self.project = nn.Linear(encoder.character_dim, model.cfg.d_model)

    def perceive(self, event_text: str, event_embedding: torch.Tensor, tick: int,
                 tags: Optional[List[str]] = None) -> None:
        self.memory.witness(event_embedding, event_text, tick, tags)

    def _conditioning_prefix(self, query_embedding: torch.Tensor) -> torch.Tensor:
        profile_vec = self.encoder(self.profile)                       # (character_dim,)
        recalled = self.memory.recall(query_embedding, top_k=3)
        mem_vecs = [r.embedding for r in recalled] or [torch.zeros_like(profile_vec)]
        mem_vec = torch.stack(mem_vecs, dim=0).mean(dim=0)
        stacked = torch.stack([profile_vec, mem_vec], dim=0)            # (2, character_dim)
        return self.project(stacked).unsqueeze(0)                        # (1, 2, d_model)

    @trace_errors
    def act(self, scene_context_ids: torch.Tensor, query_embedding: torch.Tensor,
            max_new_tokens: int = 64, temperature: float = 0.9) -> torch.Tensor:
        """Generate this agent's next action/dialogue line, conditioned on
        its own profile and whatever memory the current moment recalls.
        Each call reprocesses scene_context_ids from scratch (no carried
        KV-cache) since other agents' actions are spliced in between turns
        -- see generation_orchestrator for the single-writer streaming
        path, which DOES carry a cache across an uninterrupted sequence."""
        prefix = self._conditioning_prefix(query_embedding)
        out, _ = self.model.generate(
            scene_context_ids, max_new_tokens=max_new_tokens, temperature=temperature, hidden_prefix=prefix,
        )
        return out


class AgentSimulationLoop:
    """Steps multiple Agents through a shared scene: each tick, every agent
    acts, then every agent (including the actor) perceives what happened.
    This cross-agent perception pass is what keeps memories -- including
    grudges and betrayals -- consistent across every character's own
    dual-tier memory, not just the author's plot notes."""

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.tick = 0

    def step(self, scene_context_ids: torch.Tensor, active_ids: Sequence[str],
             query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        actions: Dict[str, torch.Tensor] = {}
        for agent_id in active_ids:
            actions[agent_id] = self.agents[agent_id].act(scene_context_ids, query_embedding)

        for observer_id in active_ids:
            for actor_id in actions:
                tags = [] if observer_id == actor_id else [actor_id]
                self.agents[observer_id].perceive(
                    event_text=f"tick {self.tick}: {actor_id} acted",
                    event_embedding=query_embedding, tick=self.tick, tags=tags,
                )
        self.tick += 1
        return actions

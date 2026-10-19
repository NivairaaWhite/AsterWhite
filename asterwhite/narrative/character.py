"""
AsterWhite :: narrative.character

CharacterProfile is the structured representation shared by
universe_tracker (character graph node), agent_simulation (the agent's own
self-model), and generation_orchestrator (conditioning for prose in this
character's voice). CharacterEncoder turns a profile into a fixed-size
vector AsterWhiteCore consumes as a hidden_prefix. export_to_game_card()
serializes a profile to plain JSON an external RPG/game engine can import
directly -- the same character model drives prose generation and gameplay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class CharacterProfile:
    character_id: str
    name: str
    traits: List[str] = field(default_factory=list)                # e.g. ["stoic", "loyal", "quick-tempered"]
    backstory: str = ""
    goals: List[str] = field(default_factory=list)                  # visible, in-universe-known goals
    hidden_agendas: List[str] = field(default_factory=list)         # secrets other characters don't know
    dialogue_style: List[str] = field(default_factory=list)         # e.g. ["terse", "sardonic", "archaic phrasing"]
    relationships: Dict[str, float] = field(default_factory=dict)   # other character_id -> affinity in [-1, 1]

    def export_to_game_card(self, reveal_secrets: bool = False) -> Dict:
        """Plain-JSON schema for import into an external RPG/game engine
        (Twine, Unity/Unreal dialogue systems, tabletop chargen tools, ...).
        `reveal_secrets=False` (default) omits hidden_agendas, matching how
        the character is actually known in-world before the plot reveals
        them."""
        card = {
            "id": self.character_id, "name": self.name, "traits": list(self.traits),
            "backstory": self.backstory, "goals": list(self.goals),
            "dialogue_style": list(self.dialogue_style), "relationships": dict(self.relationships),
        }
        if reveal_secrets:
            card["hidden_agendas"] = list(self.hidden_agendas)
        return card

    def to_json(self, reveal_secrets: bool = False) -> str:
        return json.dumps(self.export_to_game_card(reveal_secrets), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "CharacterProfile":
        return cls(
            character_id=d["id"], name=d["name"], traits=d.get("traits", []),
            backstory=d.get("backstory", ""), goals=d.get("goals", []),
            hidden_agendas=d.get("hidden_agendas", []), dialogue_style=d.get("dialogue_style", []),
            relationships=d.get("relationships", {}),
        )


class CharacterEncoder(nn.Module):
    """Encodes a CharacterProfile into a (character_dim,) conditioning
    vector: byte-level backstory/goal text pooled through an embedding
    table, plus a learned bag-of-traits embedding, projected together.
    Self-contained -- its own small embedding table -- so it's
    independently trainable/testable without the full AsterWhiteCore
    vocabulary."""

    def __init__(self, character_dim: int = 256, trait_vocab_size: int = 4096, text_vocab_size: int = 256):
        super().__init__()
        self.character_dim = character_dim
        self.text_vocab_size = text_vocab_size
        self.text_emb = nn.Embedding(text_vocab_size, character_dim)   # byte-level: no tokenizer dependency
        self.trait_emb = nn.EmbeddingBag(trait_vocab_size, character_dim, mode="mean")
        self.relationship_proj = nn.Linear(1, character_dim)
        self.combine = nn.Sequential(
            nn.Linear(character_dim * 3, character_dim * 2), nn.GELU(), nn.Linear(character_dim * 2, character_dim),
        )

    @staticmethod
    def _hash_terms(terms: List[str], vocab_size: int) -> torch.Tensor:
        if not terms:
            return torch.zeros(1, dtype=torch.long)
        return torch.tensor([abs(hash(t)) % vocab_size for t in terms], dtype=torch.long)

    def _pool_text(self, text: str) -> torch.Tensor:
        if not text:
            return self.text_emb.weight.new_zeros(self.character_dim)
        byte_ids = torch.tensor(list(text.encode("utf-8"))[:2048], dtype=torch.long) % self.text_vocab_size
        return self.text_emb(byte_ids.to(self.text_emb.weight.device)).mean(dim=0)

    def forward(self, profile: CharacterProfile) -> torch.Tensor:
        device = self.trait_emb.weight.device
        text = " ".join([profile.backstory] + profile.goals + profile.dialogue_style)
        text_vec = self._pool_text(text)

        trait_ids = self._hash_terms(profile.traits + profile.hidden_agendas, self.trait_emb.num_embeddings)
        trait_vec = self.trait_emb(trait_ids.to(device).unsqueeze(0)).squeeze(0)

        avg_affinity = (sum(profile.relationships.values()) / len(profile.relationships)
                         if profile.relationships else 0.0)
        rel_vec = self.relationship_proj(torch.tensor([[avg_affinity]], device=device, dtype=torch.float32)).squeeze(0)

        return self.combine(torch.cat([text_vec, trait_vec, rel_vec], dim=-1))

    def batch_forward(self, profiles: List[CharacterProfile]) -> torch.Tensor:
        return torch.stack([self.forward(p) for p in profiles], dim=0)

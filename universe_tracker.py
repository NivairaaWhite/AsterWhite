"""
AsterWhite :: universe_tracker

Hierarchical Universe State Engine:
  - Cosmic layer:    chronology, physical/magic/tech laws -- global, rarely mutated.
  - Local layer:     factions, locations, current scene assets -- mutated per scene.
  - Character graph: CharacterProfile nodes (asterwhite.narrative.character)
                      plus a relationship graph, mutated per beat.

A Universe holds one or more independent WorldState branches sharing this
schema -- multiverse / alternate-timeline storytelling: fork a branch at a
chosen point and let it diverge. Every WorldState exposes to_json()/
from_json() so an author can pause generation, hand-edit state as plain
JSON, and resume from it exactly (see generation_orchestrator's
checkpoint/resume hook).

DynamicStateTracker is the neural half: a learned, GRU-updated embedding
per entity (character/location/faction) that only changes when a beat
actually concerns that entity, otherwise persists untouched -- a compact
analogue of recurrent entity networks / neural checklist models, giving
the generator persistent, addressable memory instead of re-deriving state
from raw context every call.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from asterwhite.narrative.character import CharacterProfile
from asterwhite.utils.error_tracing import StateIntegrityError, get_logger, trace_errors

logger = get_logger("asterwhite.universe_tracker")


# --------------------------------------------------------------------------- #
# Structured (author-editable) state
# --------------------------------------------------------------------------- #

@dataclass
class CosmicRules:
    """Global, rarely-mutated cosmic-layer variables."""
    physical_laws: List[str] = field(default_factory=list)
    magic_or_tech_rules: List[str] = field(default_factory=list)
    calendar_system: str = "standard"


@dataclass
class Location:
    location_id: str
    name: str
    description: str = ""
    faction_control: Optional[str] = None


@dataclass
class Faction:
    faction_id: str
    name: str
    description: str = ""
    goals: List[str] = field(default_factory=list)


@dataclass
class ChronologyEvent:
    timestamp: str  # in-universe time label, author-defined format
    description: str
    location_id: Optional[str] = None
    participant_ids: List[str] = field(default_factory=list)


class WorldState:
    """One branch of a Universe. Every mutation method validates
    cross-references (an event can't name a character/location that
    doesn't exist) and raises StateIntegrityError with an exact message
    rather than silently corrupting the graph."""

    def __init__(self, universe_id: str, name: str = ""):
        self.universe_id = universe_id
        self.name = name or universe_id
        self.cosmic = CosmicRules()
        self.chronology: List[ChronologyEvent] = []
        self.locations: Dict[str, Location] = {}
        self.factions: Dict[str, Faction] = {}
        self.characters: Dict[str, CharacterProfile] = {}

    @trace_errors
    def add_location(self, loc: Location) -> None:
        self.locations[loc.location_id] = loc

    @trace_errors
    def add_faction(self, fac: Faction) -> None:
        if fac.faction_id in self.factions:
            raise StateIntegrityError(f"faction_id '{fac.faction_id}' already exists")
        self.factions[fac.faction_id] = fac

    @trace_errors
    def add_character(self, profile: CharacterProfile) -> None:
        if profile.character_id in self.characters:
            raise StateIntegrityError(f"character_id '{profile.character_id}' already exists")
        self.characters[profile.character_id] = profile

    @trace_errors
    def record_event(self, event: ChronologyEvent) -> None:
        for pid in event.participant_ids:
            if pid not in self.characters:
                raise StateIntegrityError(f"event at '{event.timestamp}' references unknown character_id '{pid}'")
        if event.location_id is not None and event.location_id not in self.locations:
            raise StateIntegrityError(
                f"event at '{event.timestamp}' references unknown location_id '{event.location_id}'"
            )
        self.chronology.append(event)

    @trace_errors
    def set_relationship(self, a: str, b: str, affinity: float, symmetric: bool = True) -> None:
        if a not in self.characters or b not in self.characters:
            raise StateIntegrityError(f"relationship references unknown character(s): '{a}', '{b}'")
        self.characters[a].relationships[b] = affinity
        if symmetric:
            self.characters[b].relationships[a] = affinity

    def validate(self) -> None:
        """Re-checks every cross-reference in the current state. Called
        automatically at the end of from_dict()/from_json() so a
        hand-edited (or otherwise untrusted) save file is verified BEFORE
        it's allowed to drive generation -- from_dict() previously
        assigned parsed records directly without re-running the same
        checks add_character()/record_event()/set_relationship() enforce
        on the normal write path, so a manually broken reference could
        silently pass through. Raises StateIntegrityError with the exact
        bad reference on any violation."""
        for loc in self.locations.values():
            if loc.faction_control is not None and loc.faction_control not in self.factions:
                raise StateIntegrityError(
                    f"location '{loc.location_id}' references unknown faction_control '{loc.faction_control}'"
                )
        for event in self.chronology:
            for pid in event.participant_ids:
                if pid not in self.characters:
                    raise StateIntegrityError(
                        f"chronology event at '{event.timestamp}' references unknown character_id '{pid}'"
                    )
            if event.location_id is not None and event.location_id not in self.locations:
                raise StateIntegrityError(
                    f"chronology event at '{event.timestamp}' references unknown location_id '{event.location_id}'"
                )
        for char_id, profile in self.characters.items():
            for other_id in profile.relationships:
                if other_id not in self.characters:
                    raise StateIntegrityError(
                        f"character '{char_id}' has a relationship to unknown character_id '{other_id}'"
                    )

    def to_dict(self) -> Dict:
        return {
            "universe_id": self.universe_id,
            "name": self.name,
            "cosmic": asdict(self.cosmic),
            "chronology": [asdict(e) for e in self.chronology],
            "locations": {k: asdict(v) for k, v in self.locations.items()},
            "factions": {k: asdict(v) for k, v in self.factions.items()},
            # reveal_secrets=True: this is an author/system save file, not a
            # player-facing export -- hidden_agendas must round-trip too.
            "characters": {k: v.export_to_game_card(reveal_secrets=True) for k, v in self.characters.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    @trace_errors
    def from_dict(cls, d: Dict) -> "WorldState":
        ws = cls(universe_id=d["universe_id"], name=d.get("name", ""))
        ws.cosmic = CosmicRules(**d.get("cosmic", {}))
        ws.locations = {k: Location(**v) for k, v in d.get("locations", {}).items()}
        ws.factions = {k: Faction(**v) for k, v in d.get("factions", {}).items()}
        ws.characters = {k: CharacterProfile.from_dict(v) for k, v in d.get("characters", {}).items()}
        ws.chronology = [ChronologyEvent(**e) for e in d.get("chronology", [])]
        ws.validate()
        return ws

    @classmethod
    def from_json(cls, s: str) -> "WorldState":
        return cls.from_dict(json.loads(s))


class Universe:
    """Holds one or more WorldState branches sharing this schema --
    multiverse / alternate-timeline storytelling: fork a branch at a chosen
    point, let it diverge, compare branches side by side."""

    def __init__(self, root_id: str = "prime"):
        self.root_id = root_id
        self.branches: Dict[str, WorldState] = {root_id: WorldState(root_id)}

    @trace_errors
    def fork(self, from_branch: str, new_branch_id: str) -> WorldState:
        if from_branch not in self.branches:
            raise StateIntegrityError(f"cannot fork unknown branch '{from_branch}'")
        if new_branch_id in self.branches:
            raise StateIntegrityError(f"branch_id '{new_branch_id}' already exists")
        forked = WorldState.from_dict(self.branches[from_branch].to_dict())
        forked.universe_id = new_branch_id
        self.branches[new_branch_id] = forked
        logger.info(f"forked universe branch '{from_branch}' -> '{new_branch_id}'")
        return forked

    def get(self, branch_id: Optional[str] = None) -> WorldState:
        return self.branches[branch_id or self.root_id]


# --------------------------------------------------------------------------- #
# Neural (learned) state -- Dynamic State Tracking
# --------------------------------------------------------------------------- #

class DynamicStateTracker(nn.Module):
    """Learned per-entity state vectors updated as the narrative
    progresses. A GRUCell-based recurrent update keyed by entity id: each
    entity's vector only updates when a beat concerns it, otherwise
    persists unchanged, giving the generator addressable memory of every
    character/location/faction without re-deriving it from raw context on
    every call."""

    def __init__(self, world_state_dim: int = 512):
        super().__init__()
        self.dim = world_state_dim
        self.update_cell = nn.GRUCell(world_state_dim, world_state_dim)
        self.entity_init = nn.Parameter(torch.randn(world_state_dim) * 0.02)
        self._states: Dict[str, torch.Tensor] = {}

    def get_state(self, entity_id: str) -> torch.Tensor:
        """_states holds plain tensors (not registered buffers), so
        nn.Module.to() does NOT move them -- re-align lazily to wherever
        the module's own parameters live right now."""
        ref = self.entity_init
        state = self._states.get(entity_id)
        if state is None:
            state = ref.clone()
        elif state.device != ref.device or state.dtype != ref.dtype:
            state = state.to(device=ref.device, dtype=ref.dtype)
        self._states[entity_id] = state
        return state

    def update(self, entity_id: str, beat_embedding: torch.Tensor) -> torch.Tensor:
        """Fold one narrative beat's embedding into entity_id's running
        state. Call once per entity actually involved in the beat --
        uninvolved entities are left untouched so their memory persists."""
        h = self.get_state(entity_id)
        h_new = self.update_cell(beat_embedding.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        self._states[entity_id] = h_new
        return h_new

    def pooled_context(self, entity_ids: Sequence[str]) -> torch.Tensor:
        """Mean-pooled state vector across several entities -- e.g. every
        character present in the current scene -- for use as part of
        generation_orchestrator's hidden_prefix."""
        if not entity_ids:
            return self.entity_init.new_zeros(self.dim)
        return torch.stack([self.get_state(e) for e in entity_ids], dim=0).mean(dim=0)

    def state_dict_snapshot(self) -> Dict[str, List[float]]:
        """Plain-Python snapshot of every entity's state vector -- the
        neural half of a checkpoint, saved alongside WorldState.to_json()
        so a paused run resumes with identical memory, not just identical
        facts."""
        return {k: v.detach().tolist() for k, v in self._states.items()}

    def load_snapshot(self, snapshot: Dict[str, List[float]]) -> None:
        ref = self.entity_init
        self._states = {k: torch.tensor(v, dtype=ref.dtype, device=ref.device) for k, v in snapshot.items()}

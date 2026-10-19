"""
AsterWhite :: generation_orchestrator

Long-Context Narrative Generator & Editable Checkpoint System.

Separates "Plot Calculation" from "Prose Synthesis":
  - PlotCalculator produces a sequence of latent plan-beat embeddings
    (hierarchical: act -> chapter -> scene) BEFORE any prose token is
    generated -- the same real technique as plan-and-write conditional
    generation, giving the model a fixed structural skeleton to stay
    faithful to instead of drifting wherever token-by-token sampling
    wanders.
  - Prose synthesis then runs on the shared AsterWhiteCore, conditioned on
    the current plan beat + universe_tracker's DynamicStateTracker context
    for whichever agent_simulation Agents are active in the scene.

Long-context strategy: attention blocks use a bounded sliding-window
KV-cache (trim_sliding_window drops the oldest keys/values past
`max_window` positions) for precise LOCAL context; SSM blocks carry
O(1)-memory recurrent state for unbounded GLOBAL compressed context (see
asterwhite.core.ssm). Together this is what lets generation keep going
across a very long document without attention memory growing without
bound.

GenerationCheckpoint bundles everything needed to pause and exactly
resume: generated token ids, the model's per-block cache (attention K/V +
SSM state), the current plan beat, and the WorldState/DynamicStateTracker
snapshot -- so an author can hand-edit the saved WorldState JSON
mid-manuscript and resume generation from the very next token with the
edited state in effect.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from asterwhite.core.model import AsterWhiteCore
from asterwhite.registry.checkpoint import flatten_tensors, load_checkpoint, save_checkpoint, unflatten_tensors
from asterwhite.utils.error_tracing import AsterWhiteError, get_logger, trace_errors
from universe_tracker import DynamicStateTracker, WorldState

logger = get_logger("asterwhite.generation_orchestrator")


# --------------------------------------------------------------------------- #
# Plot Calculation
# --------------------------------------------------------------------------- #

class PlotCalculator(nn.Module):
    """Produces latent plan-beat embeddings from a parent premise/beat
    embedding, BEFORE any prose token is generated -- a small planner
    operating in embedding space rather than token space (cheap relative
    to the main backbone) gives prose synthesis a fixed structural
    skeleton to stay faithful to.

    RECURSIVE, LAZY design: expand() is the one operation this module
    provides -- "given any single plan node, produce its N children one
    level down" -- and it is applied repeatedly to descend through a
    hierarchy (saga -> volume -> chapter -> scene, or however many levels
    a HierarchicalPlan is configured with), not just once against a flat
    list. A single earlier version of this module produced one FLAT list
    of at most max_beats (64) plan-beat embeddings for an entire work --
    which hard-caps a story at 64 planned units, nowhere near enough for
    tens of thousands of chapters, and the cap was baked directly into
    beat_query's PARAMETER SHAPE, not a config value that could just be
    raised. Composing THIS module recursively removes that ceiling
    entirely: max_children stays small and cheap to train (e.g. 20-64),
    but nesting levels multiplies out -- 50 volumes x 50 chapters/volume
    already reaches 2,500 chapters from a max_children of just 50, and a
    4th level reaches hundreds of thousands. See HierarchicalPlan, which
    is what actually navigates a tree built this way, expanding one node
    at a time and only ever holding the current root-to-leaf path in
    memory -- never the whole hypothetical tree.

    level_embedding lets the SAME weights specialize behavior by depth
    (a saga-level expansion can behave differently from a scene-level
    one) without needing separate parameter sets per level."""

    def __init__(self, plan_dim: int = 512, max_children: int = 64, n_layers: int = 2, n_heads: int = 4,
                 max_levels: int = 8):
        super().__init__()
        self.plan_dim = plan_dim
        self.max_children = max_children
        self.max_levels = max_levels
        layer = nn.TransformerEncoderLayer(d_model=plan_dim, nhead=n_heads, dim_feedforward=plan_dim * 2,
                                            batch_first=True, activation="gelu")
        self.planner = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.beat_query = nn.Parameter(torch.randn(max_children, plan_dim) * 0.02)
        self.level_embedding = nn.Embedding(max_levels, plan_dim)

    def forward(self, premise_embedding: torch.Tensor, n_beats: Optional[int] = None,
                child_level: int = 0) -> torch.Tensor:
        """premise_embedding: (B, plan_dim) -- the parent node's own beat
        embedding (or, for the saga root, an embedding of the premise/
        outline). Returns (B, n_beats, plan_dim): that many child beat
        embeddings one level down. n_beats is SILENTLY CLAMPED to
        max_children if larger (this module's beat_query parameter is
        sized exactly max_children, a hard ceiling on any single
        expansion) -- callers that need to know their request was honored
        exactly, not clamped, should compare the returned tensor's shape[1]
        against what they asked for; HierarchicalPlan validates every
        branching level against max_children upfront specifically so this
        clamping can never happen silently there. child_level selects
        level_embedding's row, clamped to max_levels - 1 (deep hierarchies
        reuse the last level's conditioning rather than erroring past a
        configured depth)."""
        n = min(n_beats or self.max_children, self.max_children)
        B = premise_embedding.shape[0]
        level_idx = torch.full((B,), min(child_level, self.max_levels - 1),
                                dtype=torch.long, device=premise_embedding.device)
        conditioned_premise = premise_embedding + self.level_embedding(level_idx)
        queries = self.beat_query[:n].unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([conditioned_premise.unsqueeze(1), queries], dim=1)
        out = self.planner(seq)
        return out[:, 1:, :]  # drop the premise position, keep the n beat embeddings

    def expand(self, node: "PlanNode", n_children: Optional[int] = None) -> List["PlanNode"]:
        """The operation HierarchicalPlan actually calls: expands ONE
        node into its children, lazily -- node.children stays None (and
        no compute happens, no memory is used for it) until this is
        called on that SPECIFIC node. Idempotent: calling this again on
        an already-expanded node just returns the existing children
        rather than recomputing/duplicating them."""
        if node.children is not None:
            return node.children
        child_level = node.level + 1
        beats = self.forward(node.beat_embedding, n_beats=n_children, child_level=child_level)
        n = beats.shape[1]
        node.children = [
            PlanNode(beat_embedding=beats[:, i, :], level=child_level, index_in_parent=i,
                      label=f"{node.label}.{i + 1}" if node.label else str(i + 1))
            for i in range(n)
        ]
        return node.children


@dataclass
class PlanNode:
    """One node in a lazily-materialized hierarchical story plan -- a
    saga, a volume within it, a chapter within a volume, or a scene
    within a chapter, depending on `level` (0 = the saga root). `children`
    is None until PlotCalculator.expand() is called on this SPECIFIC
    node -- most of a tens-of-thousands-of-chapters tree is never
    materialized at all; only the path HierarchicalPlan has actually
    walked exists as real objects in memory."""
    beat_embedding: torch.Tensor  # (1, plan_dim)
    level: int
    index_in_parent: int
    label: str
    children: Optional[List["PlanNode"]] = None


class HierarchicalPlan:
    """Navigates a PlotCalculator-expanded tree in document order (the
    saga's volume 1 -> its chapter 1 -> its chapter 2 -> ... -> volume 2's
    chapter 1 -> ...), expanding each level's children only the first
    time traversal actually reaches that node. This -- not a bigger
    max_beats -- is what makes tens of thousands of chapters tractable:
    at any moment, only the CURRENT root-to-chapter path (a handful of
    small embeddings, one per level) is held in memory, never the whole
    tree, and no upfront cost is paid for chapters that haven't been
    reached yet.

    `branching[level]` gives how many children a node AT that level
    expands into when first visited -- e.g. branching=[20, 50] means the
    saga root has 20 volumes, each volume has 50 chapters: 20*50 = 1,000
    chapters total. len(branching) is the tree's depth; the LAST entry's
    level is where chapters live (HierarchicalPlan always treats the
    deepest configured level as "a chapter" -- generate_next() writes
    prose for whatever chapter-level node current_beat() returns, across
    as many calls as that chapter needs; advance_chapter() is the
    explicit, caller-driven signal that this chapter is finished and the
    next one should begin)."""

    def __init__(self, plot_calculator: PlotCalculator, premise_embedding: torch.Tensor,
                 branching: Sequence[int], premise_label: str = "Saga"):
        if not branching:
            raise ValueError("branching must have at least one level (chapters)")
        too_large = [n for n in branching if n > plot_calculator.max_children]
        if too_large:
            raise ValueError(
                f"branching={list(branching)} requests up to {max(too_large)} children at some level, but this "
                f"PlotCalculator's max_children={plot_calculator.max_children} -- expand() would silently CLAMP "
                f"to max_children rather than error, which would make total_chapters() report a number the plan "
                f"can never actually reach. Either lower every value in branching to <= max_children, or "
                f"construct the PlotCalculator with a larger max_children."
            )
        self.plot_calculator = plot_calculator
        self.branching = list(branching)
        self.chapter_level = len(branching)  # root is level 0; branching[i] is level i+1's child count
        root = PlanNode(beat_embedding=premise_embedding, level=0, index_in_parent=0, label=premise_label)
        self.stack: List[PlanNode] = [root]
        self._descend_to_chapter()

    def _descend_to_chapter(self) -> None:
        while self.stack[-1].level < self.chapter_level:
            node = self.stack[-1]
            children = self.plot_calculator.expand(node, n_children=self.branching[node.level])
            self.stack.append(children[0])

    def current_beat(self) -> torch.Tensor:
        """(1, plan_dim) -- the CURRENT chapter's own beat embedding."""
        return self.stack[-1].beat_embedding

    def current_label(self) -> str:
        return self.stack[-1].label

    def advance_chapter(self) -> bool:
        """Moves to the next chapter in document order, transparently
        expanding new volumes/parts as needed. Returns False once EVERY
        level's branching is exhausted at the root -- the whole configured
        plan (product of branching) is complete -- True otherwise."""
        while True:
            if len(self.stack) == 1:
                return False  # exhausted the root itself: nothing left, ever
            parent = self.stack[-2]
            next_idx = self.stack[-1].index_in_parent + 1
            if next_idx < len(parent.children):
                self.stack[-1] = parent.children[next_idx]
                self._descend_to_chapter()
                return True
            self.stack.pop()  # this level's siblings are exhausted -- bubble up, try the parent's next sibling

    def total_chapters(self) -> int:
        """The product of every level's branching factor -- e.g. [20, 50]
        -> 1,000. What "tens of thousands of chapters" concretely means
        for a given branching config, without walking the tree."""
        total = 1
        for n in self.branching:
            total *= n
        return total

    def serialize_state(self) -> dict:
        """Just the current root-to-chapter PATH (a handful of small
        embeddings, one per level) -- not the tree. See GenerationCheckpoint,
        which persists this alongside token ids/cache/world state so a
        saga-scale generation run can pause and resume exactly, without
        ever having serialized (or held in memory) chapters it hasn't
        reached yet."""
        return {
            "branching": self.branching,
            "path": [{"level": n.level, "index_in_parent": n.index_in_parent, "label": n.label}
                     for n in self.stack],
        }

    @classmethod
    def restore(cls, plot_calculator: PlotCalculator, state: dict,
                path_beats: List[torch.Tensor]) -> "HierarchicalPlan":
        """Rebuilds a HierarchicalPlan from serialize_state()'s dict plus
        the corresponding beat embeddings (kept separately since
        GenerationCheckpoint routes tensors through registry.checkpoint's
        safetensors path, not JSON). Re-expands each ancestor's siblings
        exactly as the original run did (deterministic given the same
        weights + branching), so `advance_chapter()` continues correctly
        -- it does NOT recompute path_beats themselves, which is what
        makes this exact (a re-expand would give numerically-close but
        not bit-identical embeddings if the model changed at all)."""
        plan = cls.__new__(cls)
        plan.plot_calculator = plot_calculator
        plan.branching = state["branching"]
        plan.chapter_level = len(plan.branching)
        plan.stack = []
        for node_state, beat in zip(state["path"], path_beats):
            plan.stack.append(PlanNode(beat_embedding=beat, level=node_state["level"],
                                        index_in_parent=node_state["index_in_parent"],
                                        label=node_state["label"]))
        # Re-materialize each ancestor's sibling group (needed for
        # advance_chapter's parent.children lookups) WITHOUT overwriting
        # the exact restored embedding at the path's own index.
        for depth in range(len(plan.stack) - 1):
            parent = plan.stack[depth]
            child_on_path = plan.stack[depth + 1]
            children = plan.plot_calculator.expand(parent, n_children=plan.branching[parent.level])
            children[child_on_path.index_in_parent] = child_on_path
        return plan


# --------------------------------------------------------------------------- #
# Long-context cache management
# --------------------------------------------------------------------------- #

def trim_sliding_window(cache: List[dict], max_window: int, attention_sink_size: int = 0) -> None:
    """Bound attention blocks' KV-cache to the most recent `max_window`
    positions, in place. SSM blocks are untouched -- their state is already
    O(1) regardless of sequence length. This keeps memory flat over a
    generation run of arbitrary length instead of attention's KV-cache
    growing linearly with every token ever produced.

    attention_sink_size (StreamingLLM, Xiao et al. 2023): additionally keep
    the FIRST `attention_sink_size` positions permanently, never trimmed,
    alongside the sliding window. Plain sliding-window trimming alone
    causes a real, observed perplexity spike once the initial tokens fall
    out of the window -- attention sinks (the earliest few tokens, which
    empirically absorb a disproportionate share of attention mass
    regardless of their content) are cheap to keep and remove that spike,
    letting generation run indefinitely at roughly constant quality
    instead of just constant memory. 0 (default) reproduces the original
    trim-only behavior exactly."""
    for block_cache in cache:
        if "k" not in block_cache:
            continue
        total = block_cache["k"].shape[2]
        sink = min(attention_sink_size, max_window, total)
        if total <= max_window:
            continue
        recent = max_window - sink
        if sink <= 0:
            block_cache["k"] = block_cache["k"][:, :, -max_window:, :]
            block_cache["v"] = block_cache["v"][:, :, -max_window:, :]
        elif recent <= 0:
            # Degenerate configuration (attention_sink_size >= max_window):
            # no room left for any "recent" tokens at all -- fall back to
            # keeping the first max_window positions rather than a `[-0:]`
            # slice, which Python/PyTorch would silently treat as "from
            # index 0 to the end" (the whole tensor) instead of empty.
            block_cache["k"] = block_cache["k"][:, :, :max_window, :]
            block_cache["v"] = block_cache["v"][:, :, :max_window, :]
        else:
            block_cache["k"] = torch.cat([block_cache["k"][:, :, :sink, :],
                                           block_cache["k"][:, :, -recent:, :]], dim=2)
            block_cache["v"] = torch.cat([block_cache["v"][:, :, :sink, :],
                                           block_cache["v"][:, :, -recent:, :]], dim=2)
        # `offset` (used for RoPE phase) is intentionally left as the
        # true absolute position -- only the K/V *tensors* are trimmed,
        # so retained and future tokens keep correct relative phase.


# --------------------------------------------------------------------------- #
# Editable checkpoint system
# --------------------------------------------------------------------------- #

@dataclass
class GenerationCheckpoint:
    token_ids: torch.Tensor
    cache: List[dict]
    plan: Optional[HierarchicalPlan]
    world_state_json: str
    tracker_snapshot: Dict[str, List[float]]
    step_index: int
    primed: bool

    def save(self, path: str) -> None:
        """`path` is treated as a base name -- e.g. 'sessions/mystory.ckpt'
        produces 'sessions/mystory.safetensors' + 'sessions/mystory.json'
        (registry.checkpoint's atomic+checksummed format) rather than one
        pickle blob. Any tensors (token ids, per-block cache, the plan's
        current path's beat embeddings) go through safetensors; everything
        else (world state JSON, tracker snapshot, step index, the plan's
        branching/path metadata) is plain JSON -- no pickle anywhere in
        this path, so a shared/untrusted checkpoint file cannot execute
        code on load (see registry.checkpoint's module docstring).

        Only the CURRENT root-to-chapter path is persisted here -- see
        HierarchicalPlan.serialize_state's docstring for why that's
        sufficient (and is what makes checkpointing tractable at all at
        saga scale: this stays a handful of small tensors regardless of
        whether the plan is 6 chapters or 60,000)."""
        directory = os.path.dirname(path) or "."
        name = os.path.splitext(os.path.basename(path))[0]
        to_flatten = {
            "token_ids": self.token_ids,
            "cache": self.cache,
            "path_beats": [node.beat_embedding for node in self.plan.stack] if self.plan else None,
        }
        tensors, skeleton = flatten_tensors(to_flatten, prefix="gen")
        metadata = {
            "skeleton": skeleton,
            "plan_state": self.plan.serialize_state() if self.plan else None,
            "world_state_json": self.world_state_json,
            "tracker_snapshot": self.tracker_snapshot,
            "step_index": self.step_index,
            "primed": self.primed,
        }
        save_checkpoint(directory, name, tensors, metadata, make_latest=False)
        logger.info(f"generation checkpoint saved -> {directory}/{name}.* (step {self.step_index}, "
                    f"{self.token_ids.shape[1]} tokens"
                    + (f", at {self.plan.current_label()} of {self.plan.total_chapters():,} planned chapters)"
                       if self.plan else ")"))

    @classmethod
    def load(cls, path: str, plot_calculator: Optional[PlotCalculator] = None,
             map_location: str = "cpu") -> "GenerationCheckpoint":
        """plot_calculator is required if this checkpoint has a plan (it
        always will unless the orchestrator was used without begin() --
        not a normal case): reconstructing HierarchicalPlan's traversal
        state needs the SAME weights to re-expand ancestors' sibling
        groups deterministically (see HierarchicalPlan.restore).

        map_location: device the restored token ids / cache / plan beats
        are placed on -- pass the model's device (resume_from_disk does),
        otherwise they come back on CPU and the very next generate_next()
        fails with a cross-device error on any GPU run."""
        directory = os.path.dirname(path) or "."
        name = os.path.splitext(os.path.basename(path))[0]
        tensors, metadata = load_checkpoint(directory, name=name, map_location=map_location)
        restored = unflatten_tensors(metadata["skeleton"], tensors)
        # Note: world_state_json here is still a raw string -- it is
        # validated when the caller (typically
        # GenerationOrchestrator.resume_from_disk) turns it into a
        # WorldState via WorldState.from_json, which runs full
        # cross-reference validation before returning (see
        # universe_tracker.WorldState.validate).
        plan = None
        if restored["path_beats"] is not None:
            if plot_calculator is None:
                raise AsterWhiteError(
                    "this checkpoint has a hierarchical plan -- pass plot_calculator to "
                    "GenerationCheckpoint.load (or use GenerationOrchestrator.resume_from_disk, which already "
                    "requires one) so its traversal state can be reconstructed."
                )
            plan = HierarchicalPlan.restore(plot_calculator, metadata["plan_state"], restored["path_beats"])
        return cls(token_ids=restored["token_ids"], cache=restored["cache"], plan=plan,
                    world_state_json=metadata["world_state_json"], tracker_snapshot=metadata["tracker_snapshot"],
                    step_index=metadata["step_index"], primed=metadata["primed"])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class GenerationOrchestrator:
    """Ties PlotCalculator + prose synthesis (the shared AsterWhiteCore) +
    universe_tracker + a sliding-window long-context cache into one
    resumable generation session -- across a saga of arbitrarily many
    chapters, not just a single novel-length work (see HierarchicalPlan)."""

    def __init__(self, model: AsterWhiteCore, plot_calculator: PlotCalculator, world_state: WorldState,
                 tracker: DynamicStateTracker, max_window: int = 2048, attention_sink_size: int = 4,
                 conditioning_scale: float = 1.0):
        # The planner and tracker are nn.Modules just like the backbone, but
        # callers (train.py, notebooks) typically only move the backbone to
        # the GPU -- leaving these two on CPU while their inputs are built
        # on the backbone's device ("index is on cuda:0, different from
        # other tensors on cpu"). nn.Module.to() is in-place and returns the
        # same object, so the caller's own references stay valid.
        self.device = next(model.parameters()).device
        self.model = model
        self.plot_calculator = plot_calculator.to(self.device)
        self.world_state = world_state
        self.tracker = tracker.to(self.device)
        self.max_window = max_window
        self.attention_sink_size = attention_sink_size
        # How strongly the plan-beat / world-state vectors are injected into
        # the backbone, as a multiple of the model's own typical token-
        # embedding magnitude (see _conditioning_prefix). 1.0 = "as loud as an
        # ordinary token"; 0.0 = no conditioning prefix at all (plain
        # language-model continuation of the prompt).
        if conditioning_scale < 0:
            raise ValueError(f"conditioning_scale must be >= 0, got {conditioning_scale}")
        self.conditioning_scale = conditioning_scale
        self.cache: List[dict] = model.new_cache()
        self.token_ids: Optional[torch.Tensor] = None
        self.plan: Optional[HierarchicalPlan] = None
        self.step_index = 0
        self._primed = False

    def begin(self, premise_embedding: torch.Tensor, prompt_ids: torch.Tensor,
              branching: Sequence[int] = (6,), premise_label: str = "Saga") -> None:
        """branching: e.g. [6] plans a single flat run of 6 chapters (a
        normal-length short work); [20, 50] plans 20 volumes of 50
        chapters each (1,000 chapters total); add more levels for more
        scale -- see HierarchicalPlan's docstring. Only the FIRST chapter's
        beat is actually computed here; every other chapter (and every
        volume/part beyond the first) is expanded lazily the first time
        advance_chapter() reaches it, not upfront."""
        self.plan = HierarchicalPlan(self.plot_calculator, premise_embedding.to(self.device).unsqueeze(0),
                                      branching, premise_label=premise_label)
        self.token_ids = prompt_ids.to(self.device)

    @staticmethod
    def _match_dim(vec: torch.Tensor, d_model: int) -> torch.Tensor:
        if vec.shape[-1] == d_model:
            return vec
        if vec.shape[-1] < d_model:
            return F.pad(vec, (0, d_model - vec.shape[-1]))
        return vec[..., :d_model]

    def _conditioning_prefix(self, scene_character_ids: Optional[Sequence[str]]) -> Optional[torch.Tensor]:
        """(1, 2, d_model) soft-prompt [plan beat, world state] for the
        backbone, or None when conditioning_scale == 0.

        Each vector is rescaled to RMS = conditioning_scale x (RMS of the
        model's token-embedding table). The planner's and tracker's raw
        outputs live on a different scale from the backbone's learned
        embeddings (a LayerNorm'd planner output has RMS ~1; the bundled
        model's embeddings have RMS ~0.05) and enter the residual stream
        directly. Measured on the bundled model, feeding them unscaled did
        NOT change held-out loss (4.208 vs 4.211 with no prefix), so this is
        hardening rather than the fix for an observed failure -- but
        vectors ~20x outside anything the backbone ever saw are not
        something a larger model, or one fine-tuned with a prefix, should be
        left to cope with. An all-zero vector (an empty world state) stays
        all-zero."""
        if self.conditioning_scale == 0.0:
            return None
        beat_vec = self.plan.current_beat().squeeze(0)                          # (plan_dim,)
        world_vec = self.tracker.pooled_context(scene_character_ids or [])      # (world_dim,)
        d_model = self.model.cfg.d_model
        vecs = torch.stack([self._match_dim(beat_vec, d_model),
                             self._match_dim(world_vec, d_model)], dim=0)        # (2, d_model)
        target_rms = self.model.tok_emb.weight.detach().float().pow(2).mean().sqrt() * self.conditioning_scale
        vec_rms = vecs.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        scaled = vecs * (target_rms / vec_rms).to(vecs.dtype)
        # keep exact zeros exactly zero (clamp_min above would otherwise let
        # numerical noise through as an arbitrarily-scaled direction)
        scaled = torch.where(vecs.detach().abs().amax(dim=-1, keepdim=True) == 0, torch.zeros_like(scaled), scaled)
        return scaled.unsqueeze(0)

    @trace_errors
    def generate_next(self, max_new_tokens: int = 128, temperature: float = 0.9,
                       top_p: float = 0.9, top_k: Optional[int] = None, repetition_penalty: float = 1.0,
                       scene_character_ids: Optional[Sequence[str]] = None) -> torch.Tensor:
        """Prose Synthesis: generate the next span conditioned on the
        CURRENT CHAPTER's plan beat + pooled DynamicStateTracker context
        for whoever is in the scene, using the running cache so this is a
        true continuation (only the prompt is fully re-encoded, once, on
        the very first call) -- then apply the sliding-window trim.

        Call this as many times as the current chapter needs (a real
        chapter takes many spans of prose, not one) -- it does NOT
        advance to a new chapter on its own; call advance_chapter()
        explicitly once you (or your own chapter-length/end-of-chapter
        heuristic) decide the current one is finished."""
        if self.token_ids is None or self.plan is None:
            raise RuntimeError("call begin() before generate_next()")

        prefix = self._conditioning_prefix(scene_character_ids)

        # First call: prime the cache with the WHOLE prompt. Every call
        # after that: feed only the single last token -- everything before
        # it already lives in self.cache, so re-feeding the full history
        # would both waste compute and duplicate cache entries.
        feed_ids = self.token_ids if not self._primed else self.token_ids[:, -1:]
        new_ids, self.cache = self.model.generate(
            feed_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, repetition_penalty=repetition_penalty,
            hidden_prefix=prefix, cache=self.cache,
        )
        self._primed = True
        newly_generated = new_ids[:, feed_ids.shape[1]:]
        self.token_ids = torch.cat([self.token_ids, newly_generated], dim=1)

        trim_sliding_window(self.cache, self.max_window, attention_sink_size=self.attention_sink_size)
        self.step_index += 1
        return newly_generated

    def advance_chapter(self) -> bool:
        """The explicit signal that the current chapter is finished --
        moves the plan to the next chapter (transparently expanding new
        volumes/parts the first time they're reached). Returns False once
        every level's configured branching is exhausted (the whole planned
        saga is complete); True otherwise. Does NOT touch token_ids/cache
        -- the caller decides whether/how to mark a chapter break in the
        actual generated text (e.g. appending a title, or simply
        continuing the same running document)."""
        if self.plan is None:
            raise RuntimeError("call begin() before advance_chapter()")
        return self.plan.advance_chapter()

    def checkpoint(self) -> GenerationCheckpoint:
        return GenerationCheckpoint(
            token_ids=self.token_ids, cache=self.cache, plan=self.plan,
            world_state_json=self.world_state.to_json(), tracker_snapshot=self.tracker.state_dict_snapshot(),
            step_index=self.step_index, primed=self._primed,
        )

    def pause_to_disk(self, path: str) -> None:
        self.checkpoint().save(path)

    @classmethod
    def resume_from_disk(cls, path: str, model: AsterWhiteCore, plot_calculator: PlotCalculator,
                          tracker: DynamicStateTracker, max_window: int = 2048,
                          attention_sink_size: int = 4, conditioning_scale: float = 1.0) -> "GenerationOrchestrator":
        """The manual-intervention hook: an author pauses generation,
        hand-edits the saved world_state_json (or swaps it entirely), and
        resumes -- generation continues from the exact next token (and the
        exact chapter/volume position) with the edited universe/character
        state now in effect."""
        device = next(model.parameters()).device
        ckpt = GenerationCheckpoint.load(path, plot_calculator=plot_calculator.to(device), map_location=str(device))
        world_state = WorldState.from_json(ckpt.world_state_json)
        orch = cls(model, plot_calculator, world_state, tracker, max_window, attention_sink_size=attention_sink_size,
                   conditioning_scale=conditioning_scale)
        orch.cache = ckpt.cache
        orch.token_ids = ckpt.token_ids
        orch.plan = ckpt.plan
        orch.step_index = ckpt.step_index
        orch._primed = ckpt.primed
        orch.tracker.load_snapshot(ckpt.tracker_snapshot)
        logger.info(f"resumed generation from {path} at step {orch.step_index}, "
                    f"{orch.token_ids.shape[1]} tokens so far"
                    + (f", at {orch.plan.current_label()}" if orch.plan else ""))
        return orch

    def edit_world_state_json(self, edited_json: str) -> None:
        """Direct manual-intervention entry point: replace the current
        WorldState with author-edited plain JSON without a disk round-trip."""
        self.world_state = WorldState.from_json(edited_json)
        logger.info("world state manually edited in place")

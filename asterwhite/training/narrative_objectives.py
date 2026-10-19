"""
AsterWhite :: training.narrative_objectives

Explicit training objectives for the narrative-specific modules the spec
(section 7) calls out as needing them: the planner (PlotCalculator),
tracker (DynamicStateTracker), character encoder, revision suite
(QualityScorer), and style checker. Before this module, these components
had no loss function defined anywhere in the repo -- they were
architecture with nothing pulling their parameters toward a target.

Honesty about scope: these are real, correct, unit-tested loss functions
(see tests/test_narrative_objectives.py) -- what they are NOT is a
turn-key `compute_all(batch)` that trains on data you don't have yet.
Wiring them into an actual training loop needs matched narrative examples
(a premise + its beats + the prose actually written for each beat +
character profiles + known relationship affinities + a chronology of
events) -- assembling that dataset is a data-preparation task for you, not
something this module can manufacture. Each function's docstring says
exactly what tensors it needs and which existing class produces them.

NarrativeObjectiveHeads bundles the small extra parameters some of these
losses need (a bilinear relationship predictor, a beat-order classifier, a
temporal-order classifier, a safety probe) into one nn.Module so they
train and checkpoint alongside the rest of the model, instead of being
untracked ad-hoc tensors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NarrativeLossWeights:
    """Spec 7: 'Loss weights MUST be configurable, logged and included in
    checkpoints.' Pass an instance of this alongside the model config into
    registry.checkpoint metadata (e.g. metadata={"narrative_loss_weights":
    dataclasses.asdict(weights), ...}) so a resumed run can see exactly
    what weighting produced a given checkpoint."""
    plan_consistency: float = 1.0
    beat_order: float = 0.5
    character_identity: float = 1.0
    relationship_consistency: float = 0.5
    timeline_consistency: float = 0.5
    episodic_retrieval: float = 1.0
    revision_ranking: float = 1.0
    style_control: float = 0.3
    safety_policy: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NarrativeLossWeights":
        return cls(**d)


class NarrativeObjectiveHeads(nn.Module):
    """Small auxiliary heads needed only for training-time losses (never
    used at inference). Construct once, checkpoint its state_dict()
    alongside the backbone."""

    def __init__(self, d_model: int, plan_dim: int, character_dim: int,
                 world_state_dim: int, max_plan_beats: int):
        super().__init__()
        self.prose_to_plan = nn.Linear(d_model, plan_dim)
        self.prose_to_character = nn.Linear(d_model, character_dim)
        self.beat_order_head = nn.Linear(d_model, max_plan_beats)
        self.relationship_predictor = nn.Bilinear(world_state_dim, world_state_dim, 1)
        self.temporal_order_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.safety_probe = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                           nn.Linear(d_model // 2, 1))


def pooled_hidden_states(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """The same mean-pool-over-the-backbone pattern
    narrative.revision.HierarchicalRevisionSuite._pooled_hidden uses,
    exposed WITHOUT @torch.no_grad() -- these losses need gradients to
    flow back into the backbone; revision._pooled_hidden deliberately does
    not (it's an inference-time convenience)."""
    x = model.tok_emb(token_ids)
    for block in model.blocks:
        x = block(x)
    return model.norm_f(x).mean(dim=1)


def _info_nce(anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Standard in-batch-negatives contrastive loss: anchors[i] should be
    closest to positives[i] and far from positives[j != i]. Used for every
    "X should align with Y" objective below (plan<->prose,
    character<->prose, retrieval query<->memory)."""
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = anchors @ positives.T / temperature
    targets = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(logits, targets)


def plan_consistency_loss(heads: NarrativeObjectiveHeads, beat_embeddings: torch.Tensor,
                           prose_pooled: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Prose actually written for beat i should be the closest match to
    beat i's latent plan embedding, among all beats in the batch.
    beat_embeddings: (n_beats, plan_dim) -- collected from
        generation_orchestrator.PlanNode.beat_embedding for whichever
        nodes were actually visited while writing this batch (e.g. one row
        per chapter written, from HierarchicalPlan.current_beat() at the
        time -- these can be beats from different levels/branches of the
        same hierarchical plan, not necessarily one flat manuscript-wide
        list).
    prose_pooled: (n_beats, d_model) -- pooled_hidden_states() over the
        prose actually generated/written while "on" each beat, one row
        per beat, same order as beat_embeddings."""
    return _info_nce(beat_embeddings, heads.prose_to_plan(prose_pooled), temperature)


def beat_order_loss(heads: NarrativeObjectiveHeads, prose_pooled: torch.Tensor,
                     beat_indices: torch.Tensor) -> torch.Tensor:
    """Auxiliary classification: within its own immediate sibling group,
    which position (PlanNode.index_in_parent) is this prose chunk's
    content actually at? E.g. "this is chapter 12 of the current volume,"
    not a position in one flat manuscript-wide list -- the latter would
    need a classification head sized for the ENTIRE saga's chapter count
    (unworkable at the tens-of-thousands scale HierarchicalPlan is built
    for), whereas index_in_parent stays bounded by max_children (the same
    small ceiling PlotCalculator itself uses) at every level regardless of
    total saga depth. Still exactly what makes plan_consistency_loss's
    alignment meaningful across a long document rather than one scene: the
    backbone has to actually represent "where am I in the current
    chapter/volume/whatever level," not just "which beat is this."
    prose_pooled: (B, d_model). beat_indices: (B,) long, ground-truth
    0-based index_in_parent for each row (from the same PlanNode you used
    to order beat_embeddings above) -- max value must be < max_children."""
    logits = heads.beat_order_head(prose_pooled)
    return F.cross_entropy(logits, beat_indices)


def character_identity_loss(heads: NarrativeObjectiveHeads, character_vecs: torch.Tensor,
                             prose_pooled: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Dialogue/PoV prose actually written for character i should align
    with character i's CharacterEncoder vector, over other characters in
    the batch.
    character_vecs: (n_chars, character_dim) --
        narrative.character.CharacterEncoder.batch_forward(profiles)
    prose_pooled: (n_chars, d_model) -- pooled_hidden_states() over prose
        attributed to each of those same characters, same row order."""
    return _info_nce(character_vecs, heads.prose_to_character(prose_pooled), temperature)


def relationship_consistency_loss(heads: NarrativeObjectiveHeads, tracker_state_a: torch.Tensor,
                                   tracker_state_b: torch.Tensor, target_affinity: torch.Tensor) -> torch.Tensor:
    """The tracker's current entity states for two characters should be
    enough to recover their KNOWN relationship affinity -- grounds
    DynamicStateTracker's free-running GRU updates in the one piece of
    ground truth WorldState already has (WorldState.set_relationship's
    stored value), instead of letting it drift unconstrained.
    tracker_state_a, tracker_state_b: (B, world_state_dim) --
        DynamicStateTracker.get_state(char_a), .get_state(char_b), stacked
        over a batch of (char_a, char_b) pairs.
    target_affinity: (B,) float in [-1, 1] -- WorldState.characters[a].
        relationships[b] for each pair."""
    predicted = torch.tanh(heads.relationship_predictor(tracker_state_a, tracker_state_b).squeeze(-1))
    return F.mse_loss(predicted, target_affinity)


def timeline_consistency_loss(heads: NarrativeObjectiveHeads, pooled_event_a: torch.Tensor,
                               pooled_event_b: torch.Tensor, a_before_b: torch.Tensor) -> torch.Tensor:
    """Given pooled prose for two chronology events, predict whether A
    happened before B according to WorldState.chronology's own order --
    a direct penalty on timeline contradictions in generated prose.
    pooled_event_a, pooled_event_b: (B, d_model) -- pooled_hidden_states()
        over the prose describing each of two events.
    a_before_b: (B,) float in {0., 1.} -- ground truth from comparing the
        two events' positions in WorldState.chronology."""
    logits = heads.temporal_order_predictor(torch.cat([pooled_event_a, pooled_event_b], dim=-1)).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, a_before_b.float())


def episodic_retrieval_loss(query_vecs: torch.Tensor, memory_bank: torch.Tensor,
                             correct_indices: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """A query should retrieve its own correct memory over distractors --
    trains agent_simulation.EpisodicMemoryStore's embeddings to be
    retrieval-useful rather than merely present.
    query_vecs: (B, dim) -- embeddings of queries known to correspond to
        memory_bank[correct_indices[i]].
    memory_bank: (M, dim) -- EpisodicMemoryStore's stored memory vectors.
    correct_indices: (B,) long, indices into memory_bank."""
    query_vecs = F.normalize(query_vecs, dim=-1)
    memory_bank = F.normalize(memory_bank, dim=-1)
    logits = query_vecs @ memory_bank.T / temperature
    return F.cross_entropy(logits, correct_indices)


def revision_ranking_loss(draft_scores: torch.Tensor, revised_scores: torch.Tensor,
                           margin: float = 0.05) -> torch.Tensor:
    """A revised draft should score higher than what it revised, by at
    least `margin` -- a standard pairwise margin ranking loss. Reuses
    narrative.revision.QualityScorer directly (no extra head needed):
      draft_scores = scorer(pooled_hidden_states(model, draft_ids))
      revised_scores = scorer(pooled_hidden_states(model, revised_ids))
    Both (B,), same pairing order."""
    return F.relu(margin - (revised_scores - draft_scores)).mean()


def style_control_loss(drift_scores: torch.Tensor) -> torch.Tensor:
    """Pushes generated prose toward the established voice centroid.
    drift_scores: (B,) -- narrative.style_consistency.
        StyleConsistencyChecker.drift_score(generated_token_ids), called
        WITHOUT its own no_grad (drift_score itself has none; only
        update_centroid does) so this backpropagates into the checker's
        transformer, not into the fixed centroid buffer.
    Caveat verified by this module's tests: drift_score returns a constant
    1.0 tensor (no gradient) until StyleConsistencyChecker.update_centroid
    has been called at least once -- call it on a reference/accepted
    sample BEFORE training against this loss, or it is a silent no-op."""
    return (1.0 - drift_scores).mean()


def safety_policy_loss(heads: NarrativeObjectiveHeads, pooled_hidden: torch.Tensor,
                        heuristic_labels: torch.Tensor) -> torch.Tensor:
    """Supervises a lightweight safety probe against HEURISTIC labels
    (asterwhite.safety's keyword/regex policy filter run over the same
    text) -- a real, trainable signal, but explicitly not a substitute for
    a red-teamed safety classifier (see asterwhite/safety's module notes).
    pooled_hidden: (B, d_model). heuristic_labels: (B,) float in {0., 1.},
    1.0 = flagged by the heuristic filter."""
    logits = heads.safety_probe(pooled_hidden).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, heuristic_labels.float())

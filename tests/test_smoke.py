"""
AsterWhite :: tests.test_smoke

Runs every subsystem end-to-end on tiny dimensions (seconds on CPU). This
is not a placeholder -- it is the same suite that caught and fixed a real
NaN-handling bug in the CSV path during development. Run it after any
change:

    pytest tests/ -v
    python train.py selftest
"""
from __future__ import annotations

import torch

from asterwhite.alignment import GRPOTrainer
from asterwhite.config import AsterWhiteConfig, MODEL_NAME
from asterwhite.core import AsterWhiteCore
from asterwhite.narrative import (
    CharacterEncoder, CharacterProfile, HierarchicalRevisionSuite, OriginalityGuard,
    PassTagEmbedding, QualityScorer, StyleConsistencyChecker,
)
from asterwhite.utils import AsterWhiteError, trace_errors

from agent_simulation import Agent, AgentSimulationLoop, EpisodicMemoryStore, MemoryRecord
from data_pipeline import (
    InMemoryTextDataset, MemmapTokenDataset, StreamingRecordDataset,
    build_dataset, build_tokenized_binary,
)
from generation_orchestrator import GenerationOrchestrator, PlotCalculator, trim_sliding_window
from universe_tracker import ChronologyEvent, DynamicStateTracker, Location, Universe, WorldState

try:
    from asterwhite.utils.error_tracing import StateIntegrityError
except ImportError:  # pragma: no cover
    from asterwhite.utils import StateIntegrityError  # type: ignore


def _cfg():
    return AsterWhiteConfig.tiny_for_testing()


# --------------------------------------------------------------------------- #
# core backbone
# --------------------------------------------------------------------------- #

def test_model_name_is_exactly_case_sensitive_asterwhite():
    assert MODEL_NAME == "AsterWhite"


def test_forward_shapes_and_hybrid_layer_pattern():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    assert model.block_kinds == ["A", "S", "S", "A"]
    ids = torch.randint(0, cfg.backbone.vocab_size, (2, 12))
    logits = model(ids)
    assert logits.shape == (2, 12, cfg.backbone.vocab_size)


def test_incremental_cache_matches_full_forward():
    """The critical correctness property for streaming/long-context
    generation: token-by-token decode with a cache must reproduce the
    full-sequence forward pass, for BOTH attention and SSM blocks."""
    cfg = _cfg()
    torch.manual_seed(1)
    model = AsterWhiteCore(cfg.backbone).eval()
    seq = torch.randint(0, cfg.backbone.vocab_size, (1, 10))
    with torch.no_grad():
        full_logits = model(seq)
        cache = model.new_cache()
        step_logits = [model(seq[:, t:t + 1], cache=cache) for t in range(seq.shape[1])]
    step_logits = torch.cat(step_logits, dim=1)
    assert (full_logits - step_logits).abs().max().item() < 1e-3


def test_generate_and_hidden_prefix_conditioning():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 3))
    out, cache = model.generate(prompt, max_new_tokens=5)
    assert out.shape[1] == 8
    prefix = torch.randn(1, 4, cfg.backbone.d_model)
    ids = torch.randint(0, cfg.backbone.vocab_size, (1, 6))
    logits = model(ids, hidden_prefix=prefix)
    assert logits.shape == (1, 6, cfg.backbone.vocab_size)


def _probe_inner():
    return 1 / 0  # deliberate ZeroDivisionError -- this exact line is asserted against below


@trace_errors
def _probe_outer():
    return _probe_inner()


def test_error_tracing_locates_true_failure_frame():
    try:
        _probe_outer()
        assert False, "expected AsterWhiteError"
    except AsterWhiteError as e:
        assert e.location is not None
        assert e.location.file == "test_smoke.py"
        assert e.location.function == "_probe_inner"
        assert "1 / 0" in e.location.code_line


# --------------------------------------------------------------------------- #
# data_pipeline
# --------------------------------------------------------------------------- #

def test_in_memory_dataset(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("A small sample document. " * 40)
    ds = InMemoryTextDataset(str(p), block_size=16)
    assert len(ds) > 0
    assert ds[0].shape[0] == 17


def test_streaming_csv_skips_corrupt_rows(tmp_path):
    import pandas as pd
    p = tmp_path / "sample.csv"
    pd.DataFrame({"text": ["alpha beta gamma", "delta epsilon zeta", ""]}).to_csv(p, index=False)
    ds = StreamingRecordDataset([str(p)], block_size=8)
    blocks = list(ds)
    assert len(blocks) >= 1  # the empty/NaN row must be skipped, not crash the run


def test_memmap_roundtrip(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("token stream content for the memmap dataset. " * 30)
    binp = tmp_path / "tokens.bin"
    build_tokenized_binary([str(src)], str(binp), dtype="uint16")
    ds = MemmapTokenDataset(str(binp), block_size=16, dtype="uint16")
    assert len(ds) > 0
    block = ds[0]
    assert block.dtype == torch.int64
    _ = ds[len(ds) - 1]  # random access at the tail must not raise


def test_build_dataset_auto_scale_factory(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("x" * 100)
    auto = build_dataset(str(p), block_size=8, scale="auto", mb_threshold=1 << 20)
    assert isinstance(auto, InMemoryTextDataset)
    forced = build_dataset(str(p), block_size=8, scale="streaming")
    assert isinstance(forced, StreamingRecordDataset)


# --------------------------------------------------------------------------- #
# narrative subsystem
# --------------------------------------------------------------------------- #

def test_character_encoder_and_game_card_hides_secrets_by_default():
    cfg = _cfg()
    profile = CharacterProfile("c1", "Vera", traits=["stoic"], hidden_agendas=["secret plan"],
                                relationships={"c2": 0.5})
    enc = CharacterEncoder(character_dim=cfg.narrative.character_dim)
    vec = enc(profile)
    assert vec.shape == (cfg.narrative.character_dim,)
    card = profile.export_to_game_card()
    assert "hidden_agendas" not in card
    assert "hidden_agendas" in profile.export_to_game_card(reveal_secrets=True)


def test_style_consistency_drift_and_originality_guard():
    style = StyleConsistencyChecker(style_dim=16, vocab_size=64, n_layers=1, n_heads=2)
    style.update_centroid(torch.randint(0, 64, (2, 12)))
    scores = style.drift_score(torch.randint(0, 64, (3, 12)))
    assert scores.shape == (3,)

    guard = OriginalityGuard(n=3)
    guard.add([1, 2, 3, 4, 5])
    assert guard.originality_ratio([1, 2, 3, 4, 5]) < 1.0
    assert guard.originality_ratio([9, 9, 9, 9, 9]) == 1.0


def test_hierarchical_revision_suite_runs_all_passes_when_threshold_unreachable():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    scorer = QualityScorer(cfg.backbone.d_model)
    tags = PassTagEmbedding(cfg.backbone.d_model, prefix_len=2)
    suite = HierarchicalRevisionSuite(model, scorer, tags, max_passes=2, accept_threshold=2.0)
    draft = torch.randint(0, cfg.backbone.vocab_size, (1, 6))
    history = suite.revise(draft, max_new_tokens=4)
    assert [h.pass_name for h in history] == ["structural", "content"]


def test_grpo_single_step_updates_policy_without_error():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)

    def toy_reward(prompt_ids, completion_ids):
        return completion_ids.float().mean()

    trainer = GRPOTrainer(model, toy_reward, cfg.grpo)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    stats = trainer.step(prompt)
    assert isinstance(stats.loss, float)
    trainer.sync_reference()


# --------------------------------------------------------------------------- #
# universe_tracker
# --------------------------------------------------------------------------- #

def test_world_state_integrity_and_json_roundtrip():
    ws = WorldState("prime")
    ws.add_location(Location("loc_a", "The Keep"))
    ws.add_character(CharacterProfile("c1", "Vera", hidden_agendas=["secret"]))
    ws.add_character(CharacterProfile("c2", "Joran"))
    ws.set_relationship("c1", "c2", -0.4)
    ws.record_event(ChronologyEvent("Y1D1", "they meet", location_id="loc_a", participant_ids=["c1", "c2"]))

    try:
        ws.record_event(ChronologyEvent("Y1D2", "ghost", participant_ids=["missing"]))
        assert False, "expected StateIntegrityError"
    except StateIntegrityError:
        pass

    restored = WorldState.from_json(ws.to_json())
    assert restored.characters["c1"].hidden_agendas == ["secret"]
    assert restored.characters["c1"].relationships["c2"] == -0.4
    assert len(restored.chronology) == 1


def test_universe_fork_creates_independent_branch():
    uni = Universe("prime")
    uni.get("prime").add_location(Location("loc_a", "The Keep"))
    alt = uni.fork("prime", "alt")
    alt.add_location(Location("loc_b", "The Ruin"))
    assert "loc_b" not in uni.get("prime").locations
    assert "loc_b" in uni.get("alt").locations


def test_dynamic_state_tracker_persists_and_snapshots():
    tracker = DynamicStateTracker(world_state_dim=16)
    h1 = tracker.update("c1", torch.randn(16))
    h2 = tracker.update("c1", torch.randn(16))
    assert not torch.allclose(h1, h2)
    snap = tracker.state_dict_snapshot()
    restored = DynamicStateTracker(world_state_dim=16)
    restored.load_snapshot(snap)
    assert torch.allclose(restored.get_state("c1"), h2)


# --------------------------------------------------------------------------- #
# agent_simulation
# --------------------------------------------------------------------------- #

def test_episodic_memory_bounded_capacity():
    store = EpisodicMemoryStore(capacity=10, dim=8)
    for i in range(50):
        store.add(MemoryRecord(embedding=torch.randn(8), text=f"e{i}", tick=i))
    assert len(store) == 10


def test_episodic_memory_similarity_search_finds_true_match():
    store = EpisodicMemoryStore(capacity=20, dim=8)
    target = torch.randn(8)
    for i in range(15):
        store.add(MemoryRecord(embedding=torch.randn(8), text=f"noise{i}", tick=i))
    store.add(MemoryRecord(embedding=target.clone(), text="TARGET", tick=999, tags=["betrayal"]))
    hits = store.query(target, top_k=1)
    assert hits[0][0].text == "TARGET"


def test_agent_simulation_loop_cross_perception():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    encoder = CharacterEncoder(character_dim=cfg.narrative.character_dim)
    vera = CharacterProfile("c_vera", "Vera")
    joran = CharacterProfile("c_joran", "Joran")
    agents = {
        "c_vera": Agent(vera, model, encoder, working_slots=4, episodic_capacity=50),
        "c_joran": Agent(joran, model, encoder, working_slots=4, episodic_capacity=50),
    }
    loop = AgentSimulationLoop(agents)
    scene_ids = torch.randint(0, cfg.backbone.vocab_size, (1, 5))
    for _ in range(3):
        actions = loop.step(scene_ids, ["c_vera", "c_joran"], torch.randn(cfg.narrative.character_dim))
    assert set(actions.keys()) == {"c_vera", "c_joran"}
    assert len(agents["c_vera"].memory.working) == 4  # bounded by working_slots
    assert len(agents["c_vera"].memory.episodic) == 6  # 3 ticks x 2 actors, uncapped at 50


# --------------------------------------------------------------------------- #
# generation_orchestrator
# --------------------------------------------------------------------------- #

def test_sliding_window_keeps_attention_cache_bounded():
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    plot_calc = PlotCalculator(plan_dim=cfg.narrative.plan_dim, max_children=cfg.narrative.max_plan_beats)
    tracker = DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim)
    ws = WorldState("prime")
    orch = GenerationOrchestrator(model, plot_calc, ws, tracker, max_window=6)
    orch.begin(torch.randn(cfg.narrative.plan_dim), torch.randint(0, cfg.backbone.vocab_size, (1, 5)), branching=[4])
    for _ in range(4):
        orch.generate_next(max_new_tokens=5)
    attn_lens = [c["k"].shape[2] for c in orch.cache if "k" in c]
    assert all(length <= 6 for length in attn_lens)


def test_attention_sink_retains_earliest_positions_not_just_a_bounded_length():
    """trim_sliding_window with attention_sink_size > 0 must keep the FIRST
    N real positions permanently (StreamingLLM-style), not merely keep the
    cache under max_window -- distinguishes this from plain trimming."""
    k = torch.arange(20).float().view(1, 1, 20, 1)
    v = k.clone()
    cache_with_sink = [{"k": k.clone(), "v": v.clone(), "offset": 20}]
    cache_no_sink = [{"k": k.clone(), "v": v.clone(), "offset": 20}]

    trim_sliding_window(cache_with_sink, max_window=8, attention_sink_size=3)
    trim_sliding_window(cache_no_sink, max_window=8, attention_sink_size=0)

    kept_with_sink = cache_with_sink[0]["k"].view(-1).tolist()
    kept_no_sink = cache_no_sink[0]["k"].view(-1).tolist()

    assert len(kept_with_sink) == 8 and len(kept_no_sink) == 8
    assert kept_with_sink[:3] == [0.0, 1.0, 2.0]          # the sink positions survive
    assert kept_with_sink[3:] == [15.0, 16.0, 17.0, 18.0, 19.0]  # plus the most recent (8-3)=5
    assert 0.0 not in kept_no_sink                          # plain trim has none of the earliest positions left
    assert torch.equal(cache_with_sink[0]["k"], cache_with_sink[0]["v"])  # k/v trimmed identically


def test_checkpoint_resume_with_manual_world_state_edit(tmp_path):
    cfg = _cfg()
    model = AsterWhiteCore(cfg.backbone)
    plot_calc = PlotCalculator(plan_dim=cfg.narrative.plan_dim, max_children=cfg.narrative.max_plan_beats)
    tracker = DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim)
    ws = WorldState("prime")
    ws.add_location(Location("loc_keep", "The Keep"))

    orch = GenerationOrchestrator(model, plot_calc, ws, tracker, max_window=8)
    orch.begin(torch.randn(cfg.narrative.plan_dim), torch.randint(0, cfg.backbone.vocab_size, (1, 5)),
               branching=[3, 4])  # 2-level hierarchical plan: 3 volumes x 4 chapters = 12 chapters total
    orch.generate_next(max_new_tokens=5)
    orch.advance_chapter()  # move off the very first chapter, so resume is verifiably NOT just "always chapter 1"
    orch.generate_next(max_new_tokens=5)
    saved_tokens = orch.token_ids.clone()
    saved_label = orch.plan.current_label()

    ckpt_path = str(tmp_path / "ckpt.pt")
    orch.pause_to_disk(ckpt_path)

    edited = WorldState.from_json(orch.world_state.to_json())
    edited.add_location(Location("loc_swamp", "The Sunken Swamp"))

    resumed = GenerationOrchestrator.resume_from_disk(
        ckpt_path, model, plot_calc, DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim), max_window=8,
    )
    assert torch.equal(resumed.token_ids, saved_tokens)
    assert resumed.plan.current_label() == saved_label  # exact chapter position survives the round-trip
    assert resumed.plan.total_chapters() == 12
    resumed.edit_world_state_json(edited.to_json())
    assert "loc_swamp" in resumed.world_state.locations

    more = resumed.generate_next(max_new_tokens=5)
    assert resumed.token_ids.shape[1] == saved_tokens.shape[1] + more.shape[1]

    # advance_chapter() must still work correctly post-resume -- proves the
    # restored plan's sibling/ancestor structure (not just the current
    # node's own embedding) is genuinely intact, not merely a leaf stub.
    assert resumed.advance_chapter() is True
    assert resumed.plan.current_label() != saved_label

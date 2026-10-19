"""AsterWhite :: tests.test_device_placement -- regression tests for the
"Expected all tensors to be on the same device ... index is on cuda:0,
different from other tensors on cpu" crash in `train.py generate`.

Root cause: only the backbone was moved to the GPU; PlotCalculator and
DynamicStateTracker (also nn.Modules) stayed on CPU while their inputs were
created on the backbone's device. The "meta" device lets CPU-only CI prove
the placement logic without a GPU; the CUDA test is the true end-to-end
check and skips itself when no GPU is present.
"""
from __future__ import annotations

import pytest
import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core import AsterWhiteCore
from generation_orchestrator import GenerationOrchestrator, PlotCalculator
from universe_tracker import DynamicStateTracker, WorldState


def _parts(cfg):
    plot = PlotCalculator(plan_dim=cfg.narrative.plan_dim, max_children=cfg.narrative.max_plan_beats)
    tracker = DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim)
    return plot, tracker


def test_orchestrator_moves_planner_and_tracker_to_the_model_device():
    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone).to("meta")
    plot, tracker = _parts(cfg)
    orch = GenerationOrchestrator(model, plot, WorldState("u0"), tracker, max_window=8)

    assert orch.device.type == "meta"
    assert all(p.device.type == "meta" for p in plot.parameters())
    assert all(p.device.type == "meta" for p in tracker.parameters())
    # nn.Module.to() is in-place: the caller's own references stay valid
    assert orch.plot_calculator is plot and orch.tracker is tracker


def test_tracker_entity_states_follow_the_module_device():
    cfg = AsterWhiteConfig.tiny_for_testing()
    _, tracker = _parts(cfg)
    tracker.get_state("hero")                 # materialized on CPU
    tracker.to("meta")                        # _states are plain tensors: .to() alone does not move them
    assert tracker.get_state("hero").device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_generate_pause_resume_end_to_end_on_cuda(tmp_path):
    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone).cuda().eval()
    plot, tracker = _parts(cfg)               # deliberately left on CPU, as train.py used to
    orch = GenerationOrchestrator(model, plot, WorldState("u0"), tracker, max_window=8)
    orch.begin(torch.randn(cfg.narrative.plan_dim),   # CPU inputs: begin() must relocate them
               torch.randint(0, cfg.backbone.vocab_size, (1, 5)), branching=[3, 4])
    orch.generate_next(max_new_tokens=5)
    orch.advance_chapter()

    ckpt = str(tmp_path / "gpu.ckpt")
    orch.pause_to_disk(ckpt)
    plot2, tracker2 = _parts(cfg)
    resumed = GenerationOrchestrator.resume_from_disk(ckpt, model, plot2, tracker2, max_window=8)

    assert resumed.token_ids.device.type == "cuda"
    assert resumed.plan.current_beat().device.type == "cuda"
    resumed.generate_next(max_new_tokens=5)   # would raise a cross-device error if resume left tensors on CPU
    assert resumed.advance_chapter() is True

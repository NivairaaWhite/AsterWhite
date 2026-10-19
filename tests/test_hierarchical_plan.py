"""AsterWhite :: tests.test_hierarchical_plan -- generation_orchestrator's
recursive, lazily-materialized story planner (PlotCalculator.expand,
PlanNode, HierarchicalPlan). This is what replaced the old flat
PlotCalculator (hard-capped at max_beats=64 planned units total, baked
into a learned parameter tensor's shape) so a plan can reach tens of
thousands of chapters: only the current root-to-chapter path is ever held
in memory, not the whole tree.
"""
from __future__ import annotations

import pytest
import torch

from generation_orchestrator import HierarchicalPlan, PlanNode, PlotCalculator


def _tiny_calculator(plan_dim=16, max_children=8):
    return PlotCalculator(plan_dim=plan_dim, max_children=max_children, n_layers=1, n_heads=2, max_levels=6)


def test_expand_produces_correct_count_and_shape():
    calc = _tiny_calculator()
    root = PlanNode(beat_embedding=torch.randn(1, 16), level=0, index_in_parent=0, label="Saga")
    children = calc.expand(root, n_children=5)
    assert len(children) == 5
    assert all(c.beat_embedding.shape == (1, 16) for c in children)
    assert [c.index_in_parent for c in children] == [0, 1, 2, 3, 4]
    assert all(c.level == 1 for c in children)
    assert all(c.children is None for c in children)  # lazy -- nothing below this level exists yet


def test_expand_is_idempotent_not_recomputed():
    calc = _tiny_calculator()
    root = PlanNode(beat_embedding=torch.randn(1, 16), level=0, index_in_parent=0, label="Saga")
    first = calc.expand(root, n_children=4)
    second = calc.expand(root, n_children=4)
    assert first is second  # same objects, not recomputed -- calling expand again is a no-op


def test_hierarchical_plan_flat_single_level():
    calc = _tiny_calculator()
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[5])
    assert plan.total_chapters() == 5
    assert plan.current_label() == "Saga.1"

    labels_visited = [plan.current_label()]
    while plan.advance_chapter():
        labels_visited.append(plan.current_label())
    assert labels_visited == [f"Saga.{i}" for i in range(1, 6)]


def test_hierarchical_plan_two_level_document_order():
    """3 volumes x 4 chapters = 12 chapters, visited in strict document
    order (volume 1's 4 chapters, then volume 2's 4, then volume 3's 4)
    -- not just "12 distinct labels in any order"."""
    calc = _tiny_calculator()
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[3, 4])
    assert plan.total_chapters() == 12

    labels = [plan.current_label()]
    while plan.advance_chapter():
        labels.append(plan.current_label())

    expected = [f"Saga.{v}.{c}" for v in range(1, 4) for c in range(1, 5)]
    assert labels == expected
    assert plan.advance_chapter() is False  # exhausted -- calling again still returns False, doesn't error
    assert plan.advance_chapter() is False


def test_hierarchical_plan_three_level_document_order():
    calc = _tiny_calculator()
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[2, 2, 3])  # 2 parts x 2 volumes x 3 chapters = 12
    assert plan.total_chapters() == 12
    labels = [plan.current_label()]
    while plan.advance_chapter():
        labels.append(plan.current_label())
    expected = [f"Saga.{p}.{v}.{c}" for p in range(1, 3) for v in range(1, 3) for c in range(1, 4)]
    assert labels == expected


def test_memory_stays_bounded_at_real_saga_scale():
    """The core scalability claim, tested directly: advancing through
    hundreds of chapters across a plan configured for tens of thousands
    total must NOT materialize anywhere close to that many PlanNode
    objects -- only the current path (a handful of nodes) plus each
    already-visited level's sibling group (bounded by branching, not by
    total chapter count) should ever exist."""
    calc = _tiny_calculator(max_children=50)
    # 25,000 "chapters" at the 3rd level -- squarely in "tens of thousands"
    # territory, configured with a plan_dim=16 toy calculator so this runs
    # in well under a second.
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[50, 50, 10])
    assert plan.total_chapters() == 25_000

    def count_materialized_nodes(node: PlanNode) -> int:
        if node.children is None:
            return 1
        return 1 + sum(count_materialized_nodes(c) for c in node.children)

    for _ in range(300):  # a real sample, crossing multiple volume boundaries, without running all 25,000
        plan.advance_chapter()

    total_materialized = count_materialized_nodes(plan.stack[0])
    # Bound: root + (volumes expanded so far, <= a handful) * 50 chapters
    # each + at most 50*10 scenes for the couple of chapters actually
    # descended into -- nowhere near 25,000. A generous but real ceiling:
    assert total_materialized < 2_000, f"materialized {total_materialized} nodes -- lazy expansion regressed"


def test_serialize_and_restore_round_trip():
    calc = _tiny_calculator()
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[3, 4])
    plan.advance_chapter()
    plan.advance_chapter()
    label_before = plan.current_label()
    beat_before = plan.current_beat().clone()

    state = plan.serialize_state()
    path_beats = [n.beat_embedding for n in plan.stack]
    restored = HierarchicalPlan.restore(calc, state, path_beats)

    assert restored.current_label() == label_before
    assert torch.equal(restored.current_beat(), beat_before)
    assert restored.total_chapters() == plan.total_chapters()

    # advance_chapter() must keep working correctly after restore -- proves
    # the restored ancestors' sibling groups are genuinely intact, not
    # just the current node's own embedding.
    assert restored.advance_chapter() is True
    assert restored.current_label() == "Saga.1.4"


def test_single_level_branching_matches_old_flat_behavior_for_normal_length_stories():
    """A plain, non-saga-scale story (e.g. branching=[8] for a normal
    short novel) still works exactly as a flat plan would -- the
    hierarchical redesign is a strict superset, not a regression for the
    common case."""
    calc = _tiny_calculator()
    plan = HierarchicalPlan(calc, torch.randn(1, 16), branching=[8])
    assert plan.total_chapters() == 8
    count = 1
    while plan.advance_chapter():
        count += 1
    assert count == 8


def test_branching_exceeding_max_children_raises_immediately_not_silently_truncates():
    """Regression test for a real bug this test suite caught: expand()
    silently CLAMPS n_children to max_children (its beat_query parameter
    is sized exactly max_children), which would otherwise make
    total_chapters() report a story size the plan could never actually
    reach, with zero warning. HierarchicalPlan must refuse construction
    instead."""
    calc = _tiny_calculator()  # max_children=8
    with pytest.raises(ValueError, match="max_children"):
        HierarchicalPlan(calc, torch.randn(1, 16), branching=[20])
    with pytest.raises(ValueError, match="max_children"):
        HierarchicalPlan(calc, torch.randn(1, 16), branching=[5, 20])  # the SECOND level violates it -- still caught

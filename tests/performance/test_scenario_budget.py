"""
Performance budget for scenario evaluation.

`docs/02_technical_architecture_specification/performance_budget.md:30` allocates
**5 ms** to the rule engine, and `PipelineMetrics.rule_eval_ms` already measures
it, so this test costs almost nothing and closes a gap the plan named.

Two paths are measured:

* the **worst case** — every scenario evaluated every frame, no index. If that
  fits the budget, the indexed path trivially does, so this is the stronger
  claim to assert.
* the **indexed path** — only scenarios whose trigger mentions a detected class,
  which is what the compiled artifact's inverted index exists for and what a
  Raspberry Pi will actually run.

Spatial predicates are O(n^2) over detections, so the fixtures deliberately
include a crowded frame rather than a single tidy detection.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from src.pipeline import BoundingBox, Detection
from src.scenario_engine import EvalContext
from src.scenario_engine.compile import build_inverted_index
from src.scenario_engine.schema import Scenario
from src.scenario_engine.trigger import evaluate_trigger

#: The rule-engine allocation from the documented per-frame budget.
BUDGET_MS = 5.0

#: Realistic upper bound on the authored set. The Phase-6 taxonomy has 9
#: runnable scenarios; 300 is roughly a decade of growth.
SCENARIO_COUNT = 300

_CLASSES = [
    "person",
    "stove",
    "knife",
    "wet_floor",
    "walking_stick",
    "wire",
    "toilet",
    "sink",
    "support_handle",
    "medicine_strip",
    "bed",
    "chair",
]


class _Memory:
    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        return False

    def consecutive_frames(self, class_id: int) -> int:
        return 10_000

    def has_ever_seen(self, class_name: str) -> bool:
        return True


def _scenario(index: int) -> Scenario:
    """A scenario mixing every predicate kind, so the cost is representative."""
    a = _CLASSES[index % len(_CLASSES)]
    b = _CLASSES[(index + 3) % len(_CLASSES)]
    raw: dict[str, Any] = {
        "scenario_id": f"SC-KIT-{index % 1000:03d}",
        "schema_version": 1,
        "scenario_name": f"Synthetic scenario {index}",
        "category": "kitchen",
        "trigger": {
            "all": [
                {"op": "present_for", "args": {"class_name": a, "seconds": 3}},
                {"op": "count", "args": {"class_name": a, "at_least": 1}},
                {"op": "near", "args": {"class_a": a, "class_b": b, "max_dist": 0.3}},
                {"not": {"op": "absent_for", "args": {"class_name": b, "seconds": 30}}},
            ]
        },
        "risk_level": "LOW",
        "detectability": "direct",
        "claim_class": "observation",
        "status": "draft",
        "caregiver_channel": "digest",
        "next_best_action": "Continue monitoring",
        "messages": {},
        "patient_facing": False,
        "priority": index,
    }
    return Scenario.from_mapping(raw)


def _crowded_frame() -> EvalContext:
    """Several instances of several classes — spatial predicates are O(n^2)."""
    detections = [
        Detection(
            class_id=index % len(_CLASSES),
            class_name=_CLASSES[index % len(_CLASSES)],
            confidence=0.9,
            bbox=BoundingBox(cx=0.05 + 0.03 * index, cy=0.5, w=0.08, h=0.08),
            frame_id=1,
            timestamp_ms=0.0,
        )
        for index in range(24)
    ]
    return EvalContext.from_frame(detections, _Memory(), 15.0, room="kitchen")


def _median_ms(fn: Any, repeats: int = 20) -> float:
    """Median rather than mean — this box is noisy and CI is noisier."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return timings[len(timings) // 2]


@pytest.mark.performance
def test_worst_case_evaluation_fits_the_rule_engine_budget() -> None:
    """Every scenario evaluated every frame, with no index to help."""
    scenarios = [_scenario(i) for i in range(SCENARIO_COUNT)]
    ctx = _crowded_frame()

    def evaluate_all() -> None:
        for scenario in scenarios:
            evaluate_trigger(scenario.trigger, ctx)

    evaluate_all()  # warm caches before measuring
    elapsed = _median_ms(evaluate_all)

    assert elapsed < BUDGET_MS, (
        f"{SCENARIO_COUNT} scenarios took {elapsed:.2f} ms/frame against a "
        f"{BUDGET_MS} ms budget (performance_budget.md:30)"
    )


@pytest.mark.performance
def test_inverted_index_reduces_the_evaluated_set() -> None:
    """The index must actually prune, not merely exist.

    A sparse frame is the realistic case: most scenarios reference classes that
    are not present, and the artifact carries the index so an edge device can
    skip them entirely.
    """
    scenarios = [_scenario(i) for i in range(SCENARIO_COUNT)]
    index = build_inverted_index(scenarios)

    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1),
            frame_id=1,
            timestamp_ms=0.0,
        )
    ]
    ctx = EvalContext.from_frame(detections, _Memory(), 15.0, room="kitchen")

    candidate_ids = {sid for name in ctx.detected_names for sid in index.get(name, ())}
    candidates = [s for s in scenarios if s.scenario_id in candidate_ids]

    assert candidates, "the index pruned everything, which cannot be right"
    assert len(candidates) < len(scenarios), "the index pruned nothing"

    def evaluate_indexed() -> None:
        for scenario in candidates:
            evaluate_trigger(scenario.trigger, ctx)

    evaluate_indexed()
    assert _median_ms(evaluate_indexed) < BUDGET_MS


@pytest.mark.performance
def test_compilation_stays_under_ten_seconds() -> None:
    """The compile budget from the Phase-6 plan, measured on the real set."""
    from src.scenario_engine.compile import compile_scenarios

    start = time.perf_counter()
    result = compile_scenarios("configs/scenarios")
    elapsed = time.perf_counter() - start

    assert result.artifact["rule_count"] >= 1
    assert elapsed < 10.0, f"compilation took {elapsed:.2f}s against a 10s budget"

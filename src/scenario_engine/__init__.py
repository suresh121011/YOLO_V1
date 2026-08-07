"""
src.scenario_engine — Scenario & Rule Engine Knowledge Layer (Phase 6)
======================================================================

Turns detections over the frozen 23-class taxonomy into elderly-care scenarios:
risk level -> patient prompt -> next best action -> caregiver alert.

Layering (enforced by tests/unit/scenario_engine/test_layering.py):

    scripts/scenarios/*.py            build-time CLIs
            |
            v
    src/scenario_engine/              LEAF PACKAGE  <- you are here
            |
            v
    src.pipeline (contracts, event_memory) . src.utils

This package **may** import ``src.pipeline`` data contracts and ``EventMemory``,
and ``src.utils``. It **must never** import ``orchestrator``, ``detector``,
``scene_analyzer``, ``tts_engine``, ``src.dataset``, or ``src.training`` — and
``src.pipeline.rule_engine`` must never import this package. The orchestrator, a
higher layer, chooses which rule engine to inject.

See docs/08_scenario_engineering/ and its ADR-P6-01..09.
"""

from __future__ import annotations

from .context import EvalContext, MemoryView

__all__ = ["EvalContext", "MemoryView"]

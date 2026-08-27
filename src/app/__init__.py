"""
src.app — Composition Root
===========================

The only place in the codebase that knows about **both** ``src.pipeline`` and
``src.scenario_engine``.

Why it exists
-------------
The scenario engine imports the pipeline's data contracts, so the pipeline must
not import the scenario engine — that would be a package cycle, working today
only because ``src/pipeline/__init__.py`` re-exports no submodules and
detonating the first time one is added. ADR-P6-04 resolves this with constructor
injection, which leaves an obvious question: *who constructs it?*

This package. It sits above both, imports both, and is imported by neither.
``tests/unit/scenario_engine/test_layering.py`` enforces that.

Everything here is wiring. No policy, no evaluation logic — anything that
decides *behaviour* belongs in the layer that owns it, or it becomes invisible
to the validators and the gates.
"""

from __future__ import annotations

from .factory import build_pipeline, build_rule_engine

__all__ = ["build_pipeline", "build_rule_engine"]

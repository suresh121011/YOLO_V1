"""
src.dataset.annotation — Phase-5 Auto-Annotation & Verification Package
=======================================================================

Missing-annotation RESOLUTION (Phase-5), complementing the Phase-4 loss-level
mitigation: foundation-model candidate generation, CVAT human-verification
round-trip, the verification ledger, coverage estimation, and label overlay.

Core invariant (ADR-P5-01): auto-generated labels never write to any
``labels/`` directory — candidates live in their own artifact and reach
training labels only through the human verification loop.

No torch/ultralytics imports at module level anywhere in this package —
concrete backends import their runtimes lazily inside ``load()`` so the
QA/test surface stays importable in every environment (house pattern from
src/training/).
"""

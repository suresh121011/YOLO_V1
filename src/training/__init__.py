"""
src.training — Training-time extensions for the Elderly Assistant YOLO pipeline.
================================================================================

Phase-4 package: Missing Annotation Mitigation.

Public datasets label only a subset of the 23-class taxonomy, so YOLO's BCE
classification loss falsely pushes unlabeled-but-present classes toward
background. This package eliminates that supervision error at the loss level —
strictly opt-in via the ``missing_annotation_mitigation`` section of the
training config — without modifying Ultralytics source.

Modules:
    mitigation_config    — typed config surface (frozen dataclass + validation)
    completeness_lookup  — runtime reader of data/processed/completeness.json
    preflight            — pre-training gates (G1–G8) with fail-early diagnostics
    masked_loss          — masked BCE classification loss (v8DetectionLoss subclass)
    trainer              — DetectionTrainer/DetectionModel factory for injection
    evaluation           — baseline-vs-mitigated evaluation framework
    benchmark            — reproducible benchmark runner with performance budgets

Import policy: torch/ultralytics are imported lazily inside the modules that
need them, so importing this package (or the pure-Python modules) never pulls
heavyweight dependencies. When mitigation is disabled, ``train_yolo.py`` does
not import any torch-dependent module from this package.
"""

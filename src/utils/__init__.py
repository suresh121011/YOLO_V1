"""
src.utils — Shared Utility Library
====================================

Reusable helper modules for dataset preparation, QA, training, and inference
scripts. All script entry points in scripts/ import from this package.

No pipeline runtime components should import from src.utils — this package
is strictly for offline data engineering and tooling workflows.

Module inventory:
    config_helpers    — YAML loading, training config parsing, device resolution
    dataset_utils     — Image/label discovery, file hashing, pair matching
    annotation_utils  — YOLO annotation parsing and validation
    image_utils       — Image integrity validation and perceptual hashing
    report_utils      — CSV, JSON, and Markdown report generation
"""

__version__ = "0.1.0"

"""
src.dataset.capture — Custom Capture & Annotation Workflow (Phase-3)
====================================================================

Tooling for collecting, validating, annotating and tracking custom
Indian-home capture sessions for the 8 classes with no public-dataset
coverage (see configs/data.yaml and configs/capture_config.yaml).

Modules:
    config       — typed loader for configs/capture_config.yaml
    consent      — PII-free consent registry verification
    exif         — EXIF/GPS metadata inspection and stripping (privacy)
    ingest       — inbox → validated session under data/raw/custom_captures
    annotations  — CVAT (YOLO 1.1) export import, validation, finalize
    agreement    — dual-annotator agreement (IAA) computation
    progress     — collection progress vs governance targets

Operational guide: docs/04_dataset_engineering/capture_annotation_runbook.md
"""

from __future__ import annotations

from src.dataset.capture.config import (
    AnnotationSettings,
    CaptureConfig,
    CollectionTargets,
    ConsentSettings,
    IaaSettings,
    ImageRequirements,
    load_capture_config,
    parse_session_id,
)
from src.dataset.capture.consent import (
    ConsentRecord,
    find_withdrawn_consents,
    load_consent_registry,
    verify_consent,
)

__all__ = [
    "AnnotationSettings",
    "CaptureConfig",
    "CollectionTargets",
    "ConsentRecord",
    "ConsentSettings",
    "IaaSettings",
    "ImageRequirements",
    "find_withdrawn_consents",
    "load_capture_config",
    "load_consent_registry",
    "parse_session_id",
    "verify_consent",
]

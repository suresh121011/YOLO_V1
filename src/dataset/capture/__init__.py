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

from src.dataset.capture.annotations import (
    FinalizeResult,
    LabelValidation,
    YoloExport,
    finalize_annotations,
    read_yolo_export,
    stage_annotations,
    staged_annotators,
    update_annotation_status,
    validate_session_labels,
    verify_class_order,
)
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
from src.dataset.capture.exif import inspect_metadata, strip_metadata
from src.dataset.capture.ingest import (
    IngestResult,
    SessionMeta,
    ingest_session,
    init_captures_tree,
    is_eval_locked,
    load_session_manifests,
    lock_eval_set,
    rebuild_aggregate_manifest,
    verify_captures_tree,
)

__all__ = [
    "AnnotationSettings",
    "CaptureConfig",
    "CollectionTargets",
    "ConsentRecord",
    "ConsentSettings",
    "FinalizeResult",
    "IaaSettings",
    "ImageRequirements",
    "IngestResult",
    "LabelValidation",
    "SessionMeta",
    "YoloExport",
    "finalize_annotations",
    "find_withdrawn_consents",
    "ingest_session",
    "init_captures_tree",
    "inspect_metadata",
    "is_eval_locked",
    "load_capture_config",
    "load_consent_registry",
    "load_session_manifests",
    "lock_eval_set",
    "parse_session_id",
    "read_yolo_export",
    "rebuild_aggregate_manifest",
    "stage_annotations",
    "staged_annotators",
    "strip_metadata",
    "update_annotation_status",
    "validate_session_labels",
    "verify_captures_tree",
    "verify_class_order",
    "verify_consent",
]

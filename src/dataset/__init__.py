"""
src.dataset — Dataset Engineering Library
=========================================

Core, unit-testable logic for the Phase-2 dataset pipeline. The thin CLI
wrappers in ``scripts/dataset/`` and ``scripts/qa/`` import from this
package; no business logic lives in the scripts themselves.

Modules:
    manifest      — provenance manifests (source, capture-session, merged)
    split_config  — loader for configs/dataset_split_config.yaml
    sources_config— loader for configs/dataset_sources.yaml
    remap         — per-source class-ID remapping into the 23-class taxonomy
    filters       — indoor / image-quality heuristics
    dedup         — flip-robust perceptual duplicate detection
    merge         — merge remapped sources into data/merged with lineage
    negatives     — background (negative) image collection
    downloaders/  — per-source acquisition behind a common interface
    splitting/    — split strategies (group-aware, stratified-group, …)

Design rules (see docs/04_dataset_engineering/README.md):
    - Every acquisition writes a ``manifest.json`` next to its images.
    - Deduplication runs at merge time, BEFORE any split.
    - License and label-completeness metadata travel with the data.
"""

from src.dataset.manifest import (
    CaptureSessionManifest,
    MergedManifest,
    SourceManifest,
)

__all__ = [
    "CaptureSessionManifest",
    "MergedManifest",
    "SourceManifest",
]

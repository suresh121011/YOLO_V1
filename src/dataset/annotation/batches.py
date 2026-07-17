"""
src.dataset.annotation.batches — Verification Batch Manifests & Builder
========================================================================

Groups auto-annotation candidates into human-verification batches (M2, the
CVAT round-trip). A batch pins the candidate artifact it was built from
(``candidates_sha256``) so a later candidate regeneration never silently
changes an in-flight batch's pre-annotations — freshness is a sha256
cross-check (G9/RG2 at import time), not a DVC dependency edge: the
human-loop stages declare NO deps (dvc.yaml header note; breaks the
conceptual auto_annotate→…→ledger→auto_annotate cycle in the declared
graph, ADR-P5-01, plan §"DAG acyclicity").

Batch lifecycle mirrors capture sessions: created → exported → staged →
verified → imported. In-flight protection: an image already claimed by a
non-terminal batch (created/exported/staged — not yet imported) is never
double-batched, so two batches never race to verify the same cell.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError
from src.dataset.manifest import _JsonManifest, utc_now_iso

logger = logging.getLogger(__name__)

BATCHES_SCHEMA_VERSION = 1
BATCH_MANIFEST_FILENAME = "batch_manifest.json"
CVAT_LABELS_FILENAME = "cvat_labels.json"

#: Batch lifecycle (mirrors capture sessions' annotation_status).
VALID_BATCH_STATUSES = ("created", "exported", "staged", "verified", "imported")
#: Statuses that still hold an exclusive claim on their images (in-flight protection).
ACTIVE_BATCH_STATUSES = ("created", "exported", "staged")


@dataclass
class VerificationBatchManifest(_JsonManifest):
    """One CVAT verification batch's lifecycle + provenance record.

    Attributes:
        batch_id:              ``vb{NNN}_{backend}`` (sequential, stable).
        candidate_run:          ``{backend, run_id, candidates_sha256}`` —
                                pins the exact candidate artifact consumed.
        target_classes:         Class names this batch's verdicts may cover
                                (verified_absent/present_labeled scope, D4).
        images:                 Filenames in this batch (sorted).
        status:                 One of :data:`VALID_BATCH_STATUSES`.
        assignees:              Pseudonymous annotator handles.
        iaa_sample:             Filenames dual-annotated for the IAA check.
        iaa_agreement:          Measured agreement; -1.0 until measured.
        cvat_task_ref:          CVAT task id/URL once created.
        expected_gain:          Batch-ranking score (higher = prioritized).
        preannotations_sha256:  Digest of the written pre-annotation zip.
        notes:                  Free-text remarks.
    """

    batch_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    candidate_run: dict[str, Any] = field(default_factory=dict)
    target_classes: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    status: str = "created"
    assignees: list[str] = field(default_factory=list)
    iaa_sample: list[str] = field(default_factory=list)
    iaa_agreement: float = -1.0
    cvat_task_ref: str = ""
    expected_gain: float = 0.0
    preannotations_sha256: str = ""
    notes: str = ""
    schema_version: int = BATCHES_SCHEMA_VERSION


@dataclass(frozen=True)
class BatchDraft:
    """One planned batch before a batch_id/directory is assigned."""

    images: tuple[str, ...]
    target_classes: tuple[str, ...]
    expected_gain: float


def image_expected_gain(
    detections: list[Mapping[str, Any]],
    class_names_by_id: Mapping[int, str],
    priority_classes: frozenset[str],
) -> float:
    """Expected verification value of one image's candidate detections.

    Sum of detection confidences, with priority-class detections (the D1
    scope-honesty list) weighted 2x — a proxy for "how much a human
    correction here reduces missing-annotation risk". Used only to ORDER
    batches, never to filter which images get verified.
    """
    gain = 0.0
    for det in detections:
        name = class_names_by_id.get(det["class_id"])
        weight = 2.0 if name in priority_classes else 1.0
        gain += weight * float(det["conf"])
    return gain


def plan_batches(
    candidates: Mapping[str, Any],
    already_batched: frozenset[str],
    class_names_by_id: Mapping[int, str],
    priority_classes: frozenset[str],
    batch_size: int,
) -> list[BatchDraft]:
    """Rank untouched candidate images by expected gain and chunk into batches.

    Args:
        candidates:       Loaded candidate artifact (candidates.py schema).
        already_batched:  Filenames claimed by an active (non-terminal) batch
                          — never double-batched (in-flight protection).
        class_names_by_id: Taxonomy id -> name.
        priority_classes:  Names weighted higher in the gain proxy.
        batch_size:        Max images per batch (configs/annotation.yaml).

    Raises:
        AnnotationError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise AnnotationError(f"batch_size must be positive, got {batch_size}")

    images: Mapping[str, Any] = candidates.get("images", {})
    scored: list[tuple[float, str]] = []
    for filename, entry in images.items():
        if filename in already_batched:
            continue
        detections = entry.get("detections", [])
        if not detections:
            continue
        gain = image_expected_gain(detections, class_names_by_id, priority_classes)
        scored.append((gain, filename))
    # Descending gain; filename tiebreak keeps chunking deterministic.
    scored.sort(key=lambda t: (-t[0], t[1]))

    drafts: list[BatchDraft] = []
    for start in range(0, len(scored), batch_size):
        chunk = scored[start : start + batch_size]
        chunk_images = tuple(sorted(name for _, name in chunk))
        target_classes = sorted(
            {
                name
                for _, filename in chunk
                for det in images[filename]["detections"]
                if (name := class_names_by_id.get(det["class_id"])) is not None
            }
        )
        total_gain = sum(g for g, _ in chunk)
        drafts.append(
            BatchDraft(
                images=chunk_images,
                target_classes=tuple(target_classes),
                expected_gain=round(total_gain, 4),
            )
        )
    return drafts


def already_batched_images(batches_root: Path) -> frozenset[str]:
    """Filenames claimed by any non-terminal batch under ``batches_root``.

    "imported" batches release their claim — a later batch may legitimately
    re-target an image the ledger didn't fully settle (e.g. a different
    class), which the targeting.py ledger-cell check handles at the
    auto_annotate stage; this function only prevents two batches racing to
    verify the SAME cell concurrently.
    """
    claimed: set[str] = set()
    if not batches_root.exists():
        return frozenset()
    for manifest_path in sorted(batches_root.glob(f"vb*_*/{BATCH_MANIFEST_FILENAME}")):
        manifest = VerificationBatchManifest.load(manifest_path)
        if manifest.status in ACTIVE_BATCH_STATUSES:
            claimed.update(manifest.images)
    return frozenset(claimed)


def next_batch_id(batches_root: Path, backend: str) -> str:
    """Compute the next sequential batch id: ``vb{NNN}_{backend}``.

    Scans existing batch directories for the highest NNN so ids stay stable
    (never reused) even if an earlier batch directory was later removed.
    """
    max_n = 0
    if batches_root.exists():
        for path in batches_root.glob("vb*_*"):
            if not path.is_dir():
                continue
            digits = path.name[2:5]
            if digits.isdigit():
                max_n = max(max_n, int(digits))
    return f"vb{max_n + 1:03d}_{backend}"


def select_iaa_sample(images: tuple[str, ...], fraction: float) -> tuple[str, ...]:
    """Deterministically pick this batch's dual-annotated IAA sample.

    Evenly spaced picks over the SORTED image list (not random) so the
    sample is reproducible if a batch is ever rebuilt from the same inputs.
    At least one image once the batch is non-empty and ``fraction > 0`` —
    a batch small enough that 10% rounds to zero still gets a process check.

    Args:
        images:   This batch's filenames.
        fraction: ``verification.iaa_sample_fraction`` (e.g. 0.10).
    """
    if not images or fraction <= 0:
        return ()
    sorted_images = sorted(images)
    n = min(max(1, round(len(sorted_images) * fraction)), len(sorted_images))
    step = len(sorted_images) / n
    indices = sorted({int(i * step) for i in range(n)})
    return tuple(sorted_images[i] for i in indices)


def build_batch_manifests(
    candidates: Mapping[str, Any],
    backend: str,
    candidates_sha256: str,
    batches_root: Path,
    class_names_by_id: Mapping[int, str],
    priority_classes: frozenset[str],
    batch_size: int,
    iaa_sample_fraction: float = 0.0,
) -> list[VerificationBatchManifest]:
    """Plan batches from a candidate artifact and assign them sequential ids.

    Does not write anything to disk (the CLI writes the manifest + the
    cvat_package.py zip together so a manifest is never persisted without
    its pre-annotations, or vice versa).

    Args:
        candidates:          Loaded candidate artifact.
        backend:             Backend name (candidate_run provenance + batch id).
        candidates_sha256:   Digest of the on-disk candidates.json consumed.
        batches_root:        ``data/annotation/batches``.
        class_names_by_id:   Taxonomy id -> name.
        priority_classes:    Names weighted higher in the gain proxy.
        batch_size:          Max images per batch.
        iaa_sample_fraction: Fraction of each batch dual-annotated for the
                             IAA gate (``verification.iaa_sample_fraction``,
                             D4); 0 disables IAA sampling for these batches.
    """
    claimed = already_batched_images(batches_root)
    drafts = plan_batches(candidates, claimed, class_names_by_id, priority_classes, batch_size)

    manifests: list[VerificationBatchManifest] = []
    next_n = int(next_batch_id(batches_root, backend)[2:5])
    for offset, draft in enumerate(drafts):
        batch_id = f"vb{next_n + offset:03d}_{backend}"
        manifests.append(
            VerificationBatchManifest(
                batch_id=batch_id,
                candidate_run={
                    "backend": backend,
                    "run_id": candidates.get("run_id", ""),
                    "candidates_sha256": candidates_sha256,
                },
                target_classes=list(draft.target_classes),
                images=list(draft.images),
                status="created",
                iaa_sample=list(select_iaa_sample(draft.images, iaa_sample_fraction)),
                expected_gain=draft.expected_gain,
            )
        )
    return manifests

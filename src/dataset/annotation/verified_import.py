"""
src.dataset.annotation.verified_import — CVAT Export → Ledger Import
=======================================================================

Imports one verification batch's CVAT "YOLO 1.1" export (M2, D4). Reuses
``read_yolo_export`` + ``verify_class_order`` from
``src.dataset.capture.annotations`` — the same class-order-verified importer
Phase-3 capture sessions use, not a second implementation.

Per batch:
  (a) hard-fails if any NON-target-class line differs (byte-wise) from the
      base merged label — the one check that catches accidental edits to
      already-trusted labels a reviewer was never supposed to touch;
  (b) deltas = exported boxes whose class is in the batch's target_classes;
  (c) records one present_labeled/verified_absent verdict per (image,
      class) into the ledger via :func:`~src.dataset.annotation.ledger.record_verdict`
      (every conflict is a hard-fail unless ``supersedes`` is given).

Idempotent: re-running the exact same export twice records the exact same
verdicts (``record_verdict``'s no-op path) and (re)writes byte-identical
delta label files.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import VerificationBatchManifest
from src.dataset.annotation.ledger import record_verdict
from src.dataset.capture.agreement import AgreementReport, compare_annotators
from src.dataset.capture.annotations import YoloExport, verify_class_order
from src.utils.annotation_utils import Annotation, parse_label_file_raw, parse_yolo_line

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Outcome of importing one batch's export."""

    batch_id: str
    images_imported: int = 0
    verdicts_recorded: int = 0
    delta_files_written: int = 0
    problems: list[str] = field(default_factory=list)


def check_non_target_labels_unchanged(
    filename: str,
    export_lines: list[str],
    base_label_path: Path,
    target_class_ids: frozenset[int],
) -> list[str]:
    """Compare non-target-class lines between the export and the base label.

    Order-independent (CVAT's exporter does not promise line order), but
    byte-exact per line — catches a reviewer editing a box outside the
    batch's target classes, which must never happen (those boxes are
    already trusted; the reviewer's mandate is the target classes only).

    Args:
        filename:          Image filename (for the problem message).
        export_lines:      Raw stripped label lines from the export for
                           this image's stem.
        base_label_path:   ``data/merged/labels/<stem>.txt`` (absent is
                           valid — an image with no base labels at all).
        target_class_ids:  This batch's target class ids.

    Returns:
        Problem descriptions (empty = clean).
    """
    base_lines = parse_label_file_raw(base_label_path) if base_label_path.exists() else []

    def _non_target(lines: list[str]) -> list[str]:
        kept = []
        for line in lines:
            ann = parse_yolo_line(line)
            if ann is not None and ann.class_id not in target_class_ids:
                kept.append(line)
        return sorted(kept)

    base_non_target = _non_target(base_lines)
    export_non_target = _non_target(export_lines)
    if base_non_target != export_non_target:
        return [
            f"{filename}: non-target-class labels differ from the base merged label — a "
            f"reviewer edited a trusted box outside this batch's target classes "
            f"({len(base_non_target)} base non-target line(s) vs {len(export_non_target)} "
            f"in the export). Revert the accidental edit before re-importing."
        ]
    return []


def extract_deltas(export_lines: list[str], target_class_ids: frozenset[int]) -> list[Annotation]:
    """Exported boxes whose class is one of this batch's target classes."""
    deltas: list[Annotation] = []
    for line in export_lines:
        ann = parse_yolo_line(line)
        if ann is not None and ann.class_id in target_class_ids:
            deltas.append(ann)
    return deltas


def import_verified_batch(
    batch: VerificationBatchManifest,
    export: YoloExport,
    class_names_by_id: Mapping[int, str],
    ids_by_name: Mapping[str, int],
    merged_labels_dir: Path,
    verified_labels_dir: Path,
    ledger: dict[str, Any],
    source_by_image: Mapping[str, str],
    verifier: str,
    supersedes: str | None = None,
) -> ImportResult:
    """Import one batch's CVAT export: verdicts into ``ledger``, deltas to disk.

    Args:
        batch:                The batch being imported (target_classes,
                              images, candidate_run provenance).
        export:               Parsed CVAT export (``read_yolo_export``).
        class_names_by_id:    Full taxonomy, id -> name (class-order check).
        ids_by_name:          Full taxonomy, name -> id (target class lookup).
        merged_labels_dir:    ``data/merged/labels`` (base trusted labels).
        verified_labels_dir:  ``data/annotation/verified_labels`` — delta
                              files land here, one per image WITH deltas.
        ledger:               Ledger dict to mutate in-place (caller saves).
        source_by_image:      Filename -> provenance source (merged
                              manifest's ``image_provenance``).
        verifier:             Pseudonymous reviewer handle.
        supersedes:           Prior batch_id being intentionally overridden,
                              if this import conflicts with existing verdicts.

    Raises:
        AnnotationError: On a class-order mismatch, any non-target label
                         edit (checked for every image before any verdict is
                         recorded — all-or-nothing per batch), or a missing
                         provenance source.
    """
    class_order_problems = verify_class_order(export.names, dict(class_names_by_id))
    if class_order_problems:
        raise AnnotationError(
            f"Batch {batch.batch_id}: export class list does not match the taxonomy — "
            + "; ".join(class_order_problems)
        )

    target_ids = frozenset(ids_by_name[name] for name in batch.target_classes)

    exported_images = sorted(
        filename for filename in batch.images if Path(filename).stem in export.labels
    )

    non_target_problems: list[str] = []
    for filename in exported_images:
        stem = Path(filename).stem
        non_target_problems.extend(
            check_non_target_labels_unchanged(
                filename, export.labels[stem], merged_labels_dir / f"{stem}.txt", target_ids
            )
        )
    if non_target_problems:
        raise AnnotationError(
            f"Batch {batch.batch_id}: {len(non_target_problems)} image(s) have edited "
            f"non-target labels: " + " | ".join(non_target_problems)
        )

    result = ImportResult(batch_id=batch.batch_id)
    verified_labels_dir.mkdir(parents=True, exist_ok=True)

    for filename in exported_images:
        stem = Path(filename).stem
        source = source_by_image.get(filename)
        if source is None:
            raise AnnotationError(
                f"{filename}: no provenance source found in the merged manifest — "
                f"re-run the merge stage before importing."
            )

        deltas = extract_deltas(export.labels[stem], target_ids)
        for class_id in sorted(target_ids):
            class_name = class_names_by_id[class_id]
            class_boxes = [
                (ann.cx, ann.cy, ann.w, ann.h) for ann in deltas if ann.class_id == class_id
            ]
            status = "present_labeled" if class_boxes else "verified_absent"
            record_verdict(
                ledger,
                filename=filename,
                source=source,
                class_name=class_name,
                status=status,
                boxes=class_boxes,
                batch_id=batch.batch_id,
                verifier=verifier,
                method="cvat",
                cvat_task_ref=batch.cvat_task_ref,
                candidate_run=batch.candidate_run,
                supersedes=supersedes,
            )
            result.verdicts_recorded += 1

        if deltas:
            delta_text = (
                "\n".join(
                    f"{ann.class_id} {ann.cx:.6f} {ann.cy:.6f} {ann.w:.6f} {ann.h:.6f}"
                    for ann in sorted(deltas, key=lambda a: (a.class_id, a.cx, a.cy, a.w, a.h))
                )
                + "\n"
            )
            (verified_labels_dir / f"{stem}.txt").write_text(delta_text, encoding="utf-8")
            result.delta_files_written += 1
        result.images_imported += 1

    skipped = sorted(set(batch.images) - set(exported_images))
    if skipped:
        result.problems.append(
            f"{len(skipped)} batch image(s) not present in this export (partial review): "
            f"{skipped[:10]}"
        )
    return result


def compute_batch_iaa_agreement(
    batch: VerificationBatchManifest,
    primary_export: YoloExport,
    secondary_export: YoloExport,
    ids_by_name: Mapping[str, int],
    iou_threshold: float = 0.5,
) -> AgreementReport:
    """Dual-annotator agreement over this batch's IAA sample (D4 gate).

    Reuses ``compare_annotators`` — the same instrument Phase-3 capture
    sessions use for IAA — restricted to :attr:`batch.iaa_sample` images and
    :attr:`batch.target_classes` boxes only (verifying pre-labels is the
    task; agreement outside the target classes isn't this gate's concern).

    Args:
        batch:            The batch being verified (``iaa_sample`` must be
                          non-empty — callers gate on that before calling).
        primary_export:   The export used for the actual import.
        secondary_export: A second reviewer's independent export of the
                          SAME batch (dual-annotation).
        ids_by_name:      Full taxonomy, name -> id.
        iou_threshold:    Minimum IoU for two boxes to count as the same
                          object (capture's IAA default).
    """
    target_ids = frozenset(ids_by_name[name] for name in batch.target_classes)
    names_by_target_id = {i: name for name, i in ids_by_name.items() if i in target_ids}

    def _filtered(export: YoloExport) -> dict[str, list[Annotation]]:
        out: dict[str, list[Annotation]] = {}
        for filename in batch.iaa_sample:
            stem = Path(filename).stem
            out[stem] = [
                ann
                for ann in (parse_yolo_line(line) for line in export.labels.get(stem, []))
                if ann is not None and ann.class_id in target_ids
            ]
        return out

    return compare_annotators(
        _filtered(primary_export),
        _filtered(secondary_export),
        iou_threshold=iou_threshold,
        class_names=names_by_target_id,
        annotator_a="primary",
        annotator_b="secondary",
    )

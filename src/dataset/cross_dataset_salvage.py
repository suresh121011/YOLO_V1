"""
src.dataset.cross_dataset_salvage — L3 Label Salvage (ADR-P5-08)
====================================================================

D7 finding: dedup previously discarded a duplicate image's labels
unconditionally, which is dishonest whenever the DROPPED twin's source
trusts a class the KEPT source doesn't (the real overlap case at this
project's scale is Roboflow-derivative duplicates of COCO/Open Images).

Lives directly under ``src/dataset/`` (not ``src/dataset/annotation/``)
because ``merge.py`` calls it — ``src/dataset`` must never depend on
``src/dataset/annotation`` (the same layering rule that keeps
``completeness_policies.py``'s ``LedgerLike`` Protocol out of the
annotation package). The ``cross_dataset`` AutoAnnotator backend imports
FROM here instead, which is the allowed direction.

L3 v1, deliberately thin — no image-registration machinery:

    exact-sha256 duplicate — byte-identical images, so the dropped twin's
        boxes transfer directly onto the kept image's label set
        (:func:`transplant_trusted_boxes`), scoped to classes the DROPPED
        source trusts exhaustively (transplanting an untrusted-source box
        would itself be an unverified guess, defeating the point).
        Suppressed per-box when an existing same-class box on the kept
        image already overlaps at IoU >= 0.9 — both sources having already
        labeled the same object must not produce a double box.

    near-dup (perceptual, non-exact) — geometry is NOT guaranteed to
        align (crop/resize/re-encode could differ), so boxes are never
        transplanted directly. Recorded instead as a link
        (:func:`build_cross_dataset_link`) that the ``cross_dataset``
        backend turns into ordinary, human-verified candidates through the
        ADR-P5-01 pipeline — exactly as safe as any other auto-annotation
        candidate, never trusted outright.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.utils.annotation_utils import Annotation

#: D7: "both sources labeling the same object must not produce double boxes."
DEFAULT_IOU_SUPPRESS_THRESHOLD = 0.9


def _to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Normalized (x_center, y_center, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = box
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def _iou_xywhn(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two normalized xywhn boxes."""
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class SalvageResult:
    """Outcome of one exact-duplicate transplant decision.

    Attributes:
        transplanted: Dropped-image boxes to append to the kept image's
                      label set (eligible and not suppressed).
        suppressed:   Dropped-image boxes skipped as already-present
                      (IoU >= threshold against an existing same-class box).
    """

    transplanted: tuple[Annotation, ...]
    suppressed: tuple[Annotation, ...]


def transplant_trusted_boxes(
    dropped_annotations: list[Annotation],
    dropped_trusted_classes: frozenset[str],
    class_names_by_id: Mapping[int, str],
    kept_annotations: list[Annotation],
    iou_suppress_threshold: float = DEFAULT_IOU_SUPPRESS_THRESHOLD,
) -> SalvageResult:
    """Decide which of a byte-identical duplicate's boxes salvage onto the kept image.

    Pure function — no file I/O; the caller reads/writes label files.

    Args:
        dropped_annotations:     The dropped (duplicate) image's own boxes.
        dropped_trusted_classes: Class names the DROPPED image's source
                                 trusts exhaustively.
        class_names_by_id:       Taxonomy id -> name.
        kept_annotations:        The kept image's CURRENT boxes (already
                                 written by its own source).
        iou_suppress_threshold:  Suppression threshold (D7: 0.9).

    Returns:
        :class:`SalvageResult`.
    """
    kept_by_class: dict[int, list[Annotation]] = {}
    for ann in kept_annotations:
        kept_by_class.setdefault(ann.class_id, []).append(ann)

    transplanted: list[Annotation] = []
    suppressed: list[Annotation] = []
    for ann in dropped_annotations:
        class_name = class_names_by_id.get(ann.class_id)
        if class_name is None or class_name not in dropped_trusted_classes:
            continue
        existing = kept_by_class.get(ann.class_id, [])
        max_iou = max(
            (_iou_xywhn((ann.cx, ann.cy, ann.w, ann.h), (e.cx, e.cy, e.w, e.h)) for e in existing),
            default=0.0,
        )
        if max_iou >= iou_suppress_threshold:
            suppressed.append(ann)
        else:
            transplanted.append(ann)
    return SalvageResult(transplanted=tuple(transplanted), suppressed=tuple(suppressed))


def render_transplanted_lines(annotations: tuple[Annotation, ...]) -> list[str]:
    """YOLO-format lines for transplanted boxes, ready to append to a label file."""
    return [f"{ann.class_id} {ann.cx} {ann.cy} {ann.w} {ann.h}" for ann in annotations]


def build_cross_dataset_link(
    dropped_annotations: list[Annotation],
    dropped_trusted_classes: frozenset[str],
    class_names_by_id: Mapping[int, str],
    dropped_source: str,
) -> dict[str, object] | None:
    """Build one near-dup link entry for ``cross_dataset_links.json``.

    Same trusted-class eligibility rule as :func:`transplant_trusted_boxes`
    (only the dropped source's own exhaustively-labeled classes are worth
    proposing) — but geometry is passed through UNVERIFIED, since a near-dup
    (not byte-identical) twin's boxes are never assumed to align.

    Args:
        dropped_annotations:     The dropped (near-dup) image's own boxes.
        dropped_trusted_classes: Class names the DROPPED image's source
                                 trusts exhaustively.
        class_names_by_id:       Taxonomy id -> name.
        dropped_source:          The dropped image's source name (provenance).

    Returns:
        ``{"source": ..., "boxes": [[class_id, cx, cy, w, h], ...]}``, or
        ``None`` if no eligible box exists (nothing worth linking).
    """
    boxes = [
        [ann.class_id, ann.cx, ann.cy, ann.w, ann.h]
        for ann in dropped_annotations
        if class_names_by_id.get(ann.class_id) in dropped_trusted_classes
    ]
    if not boxes:
        return None
    return {"source": dropped_source, "boxes": boxes}

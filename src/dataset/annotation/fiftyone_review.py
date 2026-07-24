"""
src.dataset.annotation.fiftyone_review — FiftyOne Review Surface (P8)
=====================================================================

Replaces CVAT as the human review UI for a verification batch, **without**
touching the ledger/IAA machinery — for a safety-critical elderly dataset the
human verdict is never automated (ADR-P5-01). FiftyOne (Apache-2.0,
Python-native) embeds in the DVC/Ultralytics stack and needs no CVAT server.

This module is a *format bridge*: the pre-annotation labels a reviewer sees are
still ``build_preannotation_labels`` (base merged ∪ candidates) — identical to
the CVAT path — only the presentation differs. On the way out, reviewed labels
are written back as YOLO txt so the EXISTING import path
(``verified_import.py``: ``verify_class_order`` → ``check_non_target_labels_unchanged``
→ ``extract_deltas`` → ledger) consumes them unchanged. That reuse is what keeps
FiftyOne and CVAT at ledger parity (regression gate).

``fiftyone`` is imported lazily inside the functions that need it (house
pattern, ADR-P5-11), so this module imports everywhere; only the App-driving
functions require the optional dependency. The coordinate math and the
label↔detections conversions are pure/duck-typed and fully unit-tested.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: (class_id, cx, cy, w, h) — one YOLO detection row in normalized geometry.
LabelBox = tuple[int, float, float, float, float]


def xywhn_to_fo_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """YOLO center xywhn → FiftyOne ``[top_left_x, top_left_y, w, h]`` (normalized)."""
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def fo_bbox_to_xywhn(bbox: list[float]) -> tuple[float, float, float, float]:
    """FiftyOne ``[top_left_x, top_left_y, w, h]`` → YOLO center xywhn."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0, w, h)


def parse_label_text(text: str) -> list[LabelBox]:
    """Parse YOLO ``class cx cy w h`` label text into :data:`LabelBox` rows.

    Malformed / non-5-field / non-numeric lines are skipped (structural QA is
    where malformed labels fail, not the review bridge).
    """
    boxes: list[LabelBox] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


def boxes_to_label_text(boxes: list[LabelBox]) -> str:
    """:data:`LabelBox` rows → YOLO label text (6-decimal, sorted for determinism)."""
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in sorted(boxes)]
    return "\n".join(lines) + ("\n" if lines else "")


def to_fo_detections(boxes: list[LabelBox], class_names: Mapping[int, str]) -> Any:
    """Build a ``fiftyone.Detections`` from :data:`LabelBox` rows (lazy import).

    Class ids map to string labels via ``class_names``; ids absent from the map
    are skipped defensively (they cannot round-trip back to an id).
    """
    import fiftyone as fo

    dets = []
    for cid, cx, cy, w, h in boxes:
        name = class_names.get(cid)
        if name is None:
            continue
        dets.append(fo.Detection(label=name, bounding_box=xywhn_to_fo_bbox(cx, cy, w, h)))
    return fo.Detections(detections=dets)


def from_fo_detections(detections: Any, name_to_id: Mapping[str, int]) -> list[LabelBox]:
    """Reverse of :func:`to_fo_detections` — duck-typed on ``.detections`` /
    ``.label`` / ``.bounding_box`` so it is testable without ``fiftyone``.

    Detections whose label is not in ``name_to_id`` are dropped (a reviewer
    cannot introduce an out-of-taxonomy class).
    """
    if detections is None:
        return []
    boxes: list[LabelBox] = []
    for det in getattr(detections, "detections", []) or []:
        cid = name_to_id.get(det.label)
        if cid is None:
            continue
        cx, cy, w, h = fo_bbox_to_xywhn(list(det.bounding_box))
        boxes.append((cid, cx, cy, w, h))
    return boxes


def build_review_dataset(
    name: str,
    samples: list[tuple[Path, str]],
    class_names: Mapping[int, str],
    label_field: str = "prelabels",
) -> Any:
    """Create a FiftyOne dataset for one verification batch (lazy import).

    Args:
        name:        Dataset name (e.g. the batch id).
        samples:     ``(image_path, preannotation_label_text)`` per image —
                     the label text is ``build_preannotation_labels`` output
                     (base merged ∪ candidates), identical to the CVAT path.
        class_names: Taxonomy id → name.
        label_field: Sample field the pre-annotations land in for review.

    Returns:
        A ``fiftyone.Dataset`` ready for :func:`launch_app`.
    """
    import fiftyone as fo

    dataset = fo.Dataset(name=name, overwrite=True)
    fo_samples = []
    for image_path, label_text in samples:
        sample = fo.Sample(filepath=str(image_path))
        sample[label_field] = to_fo_detections(parse_label_text(label_text), class_names)
        fo_samples.append(sample)
    dataset.add_samples(fo_samples)
    logger.info(f"FiftyOne review dataset '{name}': {len(fo_samples)} samples")
    return dataset


def launch_app(dataset: Any, port: int = 5151, wait: bool = True) -> Any:
    """Launch the FiftyOne App on ``dataset`` for human review (lazy import)."""
    import fiftyone as fo

    session = fo.launch_app(dataset, port=port)
    if wait:
        session.wait()
    return session


def export_reviewed_labels(
    dataset: Any,
    out_labels_dir: Path,
    name_to_id: Mapping[str, int],
    label_field: str = "prelabels",
) -> int:
    """Write each reviewed sample's detections back to a YOLO ``.txt``.

    The output directory is exactly what ``verified_import.read_yolo_export``
    expects, so the CVAT import path (class-order + non-target-unchanged guards,
    delta extraction, ledger verdicts) consumes FiftyOne review output unchanged.

    Returns the number of label files written.
    """
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sample in dataset:
        stem = Path(sample.filepath).stem
        boxes = from_fo_detections(sample[label_field], name_to_id)
        (out_labels_dir / f"{stem}.txt").write_text(boxes_to_label_text(boxes), encoding="utf-8")
        written += 1
    logger.info(f"Exported {written} reviewed label files to {out_labels_dir}")
    return written

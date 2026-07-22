"""
src.dataset.annotation.cvat_package — CVAT YOLO 1.1 Pre-Annotation Packaging
=============================================================================

Builds the artifacts a human uploads into CVAT for one verification batch
(D4): a "YOLO 1.1" pre-annotation zip (``obj.names`` = the FULL ordered
23-class taxonomy, never a batch-scoped subset — a CVAT task must always be
able to display/correct any class) whose per-image label files are the base
merged labels UNION the batch's candidate detections, plus a
``cvat_labels.json`` label-constructor spec so the CVAT task's label list is
created in exact taxonomy order — this kills the manual label-order mistake
class before ``verify_class_order`` (reused from
``src.dataset.capture.annotations``) even runs at import time.

Zip entries use a fixed timestamp so the archive — and therefore its sha256
recorded in the batch manifest — is a deterministic function of its content,
not of wall-clock build time.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.dataset_utils import compute_file_hash

#: Fixed per-entry zip timestamp (minimum the zip format allows) — keeps
#: repeated builds of identical content byte-identical.
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def build_cvat_labels_spec(class_names_by_id: Mapping[int, str]) -> list[dict[str, Any]]:
    """CVAT label-constructor spec, taxonomy id order (D4).

    Paste as CVAT's "raw" label list when creating the verification task so
    label order can never drift from ``configs/data.yaml``.

    Each entry MUST carry an ``attributes`` array: CVAT's Raw label editor
    (``cvat-ui`` ``labels-editor/common.ts::validateParsedLabel``) rejects any
    label whose ``attributes`` is not an array with ``"Attributes must be an
    array"``. That client-side failure is what surfaces as the opaque
    ``labels: [object Object]`` task-creation error and the ``POST /api/tasks``
    ``400`` — a bare ``{"name": ...}`` object is NOT a valid Raw-editor label.
    ``name`` is the only server-required field, but the browser never lets a
    bare-name list reach the server. ``color``/``type`` stay unset so CVAT
    assigns its defaults (auto colour, ``type: "any"`` = all draw tools).
    """
    return [{"name": class_names_by_id[i], "attributes": []} for i in sorted(class_names_by_id)]


def build_preannotation_labels(
    detections: list[Mapping[str, Any]],
    base_label_path: Path,
) -> str:
    """YOLO-format label text: base merged labels UNION candidate detections.

    Args:
        detections:      This image's candidate detections (already scoped
                         to untrusted+unverified target classes upstream).
        base_label_path: ``data/merged/labels/<stem>.txt`` (absent for
                         images with no trusted labels at all).

    Returns:
        Label file text (empty string if there is nothing to write).
    """
    lines: list[str] = []
    if base_label_path.exists():
        lines.extend(
            line
            for line in base_label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    for det in sorted(detections, key=lambda d: (d["class_id"], tuple(d["bbox_xywhn"]))):
        x, y, w, h = det["bbox_xywhn"]
        lines.append(f"{det['class_id']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _write_deterministic(zf: zipfile.ZipFile, arcname: str, text: str) -> None:
    """Write one text entry with a fixed timestamp (reproducible zip bytes)."""
    info = zipfile.ZipInfo(arcname, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, text)


def build_preannotation_zip(
    batch_images: list[str],
    candidate_images: Mapping[str, Any],
    merged_labels_dir: Path,
    class_names_by_id: Mapping[int, str],
    out_zip: Path,
) -> str:
    """Write one batch's CVAT "YOLO 1.1" pre-annotation zip.

    Layout (standard Darknet/YOLO, what CVAT's importer expects):
    ``obj.names``, ``obj.data``, ``train.txt``,
    ``obj_train_data/<stem>.txt`` per image.

    Args:
        batch_images:      Filenames in this batch.
        candidate_images:  The candidate artifact's ``images`` mapping
                           (filename → ``{detections: [...], ...}``).
        merged_labels_dir: ``data/merged/labels`` (base trusted labels).
        class_names_by_id: Full taxonomy, id -> name (obj.names order).
        out_zip:           Destination zip path; parent dirs are created.

    Returns:
        The written zip's sha256 hex digest.
    """
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    sorted_images = sorted(batch_images)

    names_text = "\n".join(class_names_by_id[i] for i in sorted(class_names_by_id)) + "\n"
    obj_data_text = (
        f"classes = {len(class_names_by_id)}\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n"
    )
    train_text = "".join(f"data/obj_train_data/{name}\n" for name in sorted_images)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_deterministic(zf, "obj.names", names_text)
        _write_deterministic(zf, "obj.data", obj_data_text)
        _write_deterministic(zf, "train.txt", train_text)
        for filename in sorted_images:
            entry = candidate_images[filename]
            stem = Path(filename).stem
            label_text = build_preannotation_labels(
                entry["detections"], merged_labels_dir / f"{stem}.txt"
            )
            _write_deterministic(zf, f"obj_train_data/{stem}.txt", label_text)

    return compute_file_hash(out_zip)

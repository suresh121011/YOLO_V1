"""
src.dataset.downloaders.wider_face — WIDER FACE Downloader
==========================================================

Face-class acquisition from the WIDER FACE benchmark (HuggingFace-hosted
zips). LICENSE GATE: WIDER FACE is research-only / non-commercial; this
source only runs while ``allow_noncommercial: true`` — the gate, the
manifest license string, and the QA license report keep that visible
(governance doc §2).

Layout notes: the images zip stores ``WIDER_{split}/images/<event>/<file>``
and annotations come as ``wider_face_split.zip`` containing
``wider_face_{split}_bbx_gt.txt`` (filename line, box-count line, then
``x y w h blur expression illumination invalid occlusion pose`` lines,
absolute pixels). Only the selected members are extracted from the zip —
smoke mode never unpacks the whole archive.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

from src.dataset.downloaders.base import (
    BaseDownloader,
    DownloadSkippedError,
    coco_bbox_to_yolo,
    write_yolo_label,
)

logger = logging.getLogger(__name__)

_LOG_EVERY = 20


class WiderFaceDownloader(BaseDownloader):
    """WIDER FACE subset acquisition for the ``face`` class."""

    def source_classes(self) -> dict[str, str]:
        return {"0": "face"}

    def _query_extras(self) -> dict[str, Any]:
        return {"split": self._split(), "class_caps": self.source.options.get("class_caps", {})}

    def _split(self) -> str:
        return "val" if self.config.mode == "smoke" else "train"

    def fetch(self, limit: int | None) -> dict[str, int]:
        """Extract capped images + ground truth from the cached zips."""
        if self.source.noncommercial and not self.config.allow_noncommercial:
            raise DownloadSkippedError(
                "WIDER FACE is research-only/non-commercial and " "allow_noncommercial is false"
            )

        split = self._split()
        images_key = "smoke_images_url" if self.config.mode == "smoke" else "full_images_url"
        images_url = str(self.source.options[images_key])
        annotations_url = str(self.source.options["annotations_url"])

        annotations_zip = self.downloads_dir / Path(annotations_url).name
        images_zip = self.downloads_dir / Path(images_url).name
        if not self.fetch_url(annotations_url, annotations_zip):
            raise RuntimeError(f"Could not download WIDER FACE annotations: {annotations_url}")
        if not self.fetch_url(images_url, images_zip):
            raise RuntimeError(f"Could not download WIDER FACE images: {images_url}")

        ground_truth = self._parse_ground_truth(annotations_zip, split)
        logger.info(f"[wider_face] {len(ground_truth)} annotated images in {split}")

        # Full mode has no image `limit` — WIDER_train alone would otherwise
        # contribute ~160k face instances, dwarfing the rest of the ~30k-image
        # full-mode budget (M7 full-mode config review). Caps the single
        # "face" class the same way COCO caps per-class instances, just with
        # one class instead of many.
        face_cap = (self.source.options.get("class_caps") or {}).get("face")

        class_counts: dict[str, int] = {"face": 0}
        selected = 0
        with zipfile.ZipFile(images_zip) as zf:
            members = set(zf.namelist())
            for rel_name in sorted(ground_truth):  # deterministic
                if limit is not None and selected >= limit:
                    break
                if face_cap is not None and class_counts["face"] >= int(face_cap):
                    logger.info(f"[wider_face] face class_cap {face_cap} reached — stopping")
                    break
                member = f"WIDER_{split}/images/{rel_name}"
                if member not in members:
                    logger.debug(f"[wider_face] missing member {member}")
                    continue

                # Flatten the event-folder path into a unique flat filename.
                flat_name = rel_name.replace("/", "_")
                image_dest = self.images_dir / flat_name
                if not image_dest.exists():
                    with zf.open(member) as src:
                        image_dest.write_bytes(src.read())

                boxes = self._to_yolo_boxes(image_dest, ground_truth[rel_name])
                if not boxes:
                    image_dest.unlink(missing_ok=True)
                    continue

                write_yolo_label(self.labels_dir / f"{Path(flat_name).stem}.txt", boxes)
                class_counts["face"] += len(boxes)
                selected += 1
                if selected % _LOG_EVERY == 0:
                    logger.info(f"[wider_face] {selected} images extracted…")

        logger.info(f"[wider_face] done: {selected} images, {class_counts['face']} faces")
        return class_counts

    @staticmethod
    def _parse_ground_truth(
        annotations_zip: Path,
        split: str,
    ) -> dict[str, list[list[float]]]:
        """Relative image path → list of valid [x, y, w, h] pixel boxes."""
        member = f"wider_face_split/wider_face_{split}_bbx_gt.txt"
        with zipfile.ZipFile(annotations_zip) as zf, zf.open(member) as fh:
            lines = fh.read().decode("utf-8").splitlines()

        ground_truth: dict[str, list[list[float]]] = {}
        i = 0
        while i < len(lines):
            rel_name = lines[i].strip()
            i += 1
            if not rel_name:
                continue
            count = int(lines[i].strip())
            i += 1
            boxes: list[list[float]] = []
            # count == 0 still has one placeholder box line in the GT format.
            for j in range(max(count, 1)):
                parts = lines[i + j].split()
                if count == 0:
                    continue
                x, y, w, h = (float(v) for v in parts[:4])
                invalid = int(parts[7]) if len(parts) > 7 else 0
                if invalid or w < 8 or h < 8:  # drop invalid / tiny faces
                    continue
                boxes.append([x, y, w, h])
            i += max(count, 1)
            if boxes:
                ground_truth[rel_name] = boxes
        return ground_truth

    @staticmethod
    def _to_yolo_boxes(
        image_path: Path,
        pixel_boxes: list[list[float]],
    ) -> list[tuple[int, float, float, float, float]]:
        """Convert absolute pixel boxes to YOLO tuples for local class 0."""
        from src.utils.image_utils import get_image_dimensions

        dims = get_image_dimensions(image_path)
        if dims is None:
            return []
        width, height = dims
        boxes: list[tuple[int, float, float, float, float]] = []
        for pixel_box in pixel_boxes:
            yolo = coco_bbox_to_yolo(pixel_box, float(width), float(height))
            if yolo is not None:
                boxes.append((0, *yolo))
        return boxes

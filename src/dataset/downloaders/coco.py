"""
src.dataset.downloaders.coco — COCO 2017 Subset Downloader
==========================================================

Annotations-first acquisition (docs/03 dataset_templates.md §3.1):

    1. Download the official ``annotations_trainval2017.zip`` once
       (~241 MB, cached in ``_downloads/``) and extract only the instances
       JSON for the configured split (val2017 in smoke mode, train2017 in
       full mode).
    2. Select images containing our 10 mapped classes, honoring per-class
       caps (person 800, chair 500 by default) and the smoke limit.
    3. Fetch ONLY the selected images per-URL from images.cocodataset.org.

Memory note: parsing instances_train2017.json (full mode) loads a ~450 MB
JSON — needs ~3 GB RAM. Acceptable one-time cost; switch to ijson if it
becomes a problem (risk R-P2-1 in the plan).
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from src.dataset.downloaders.base import (
    BaseDownloader,
    coco_bbox_to_yolo,
    write_yolo_label,
)
from src.dataset.remap import REMAP_TABLES

logger = logging.getLogger(__name__)

_LOG_EVERY = 20


class CocoDownloader(BaseDownloader):
    """COCO 2017 subset acquisition for the 10 COCO-supplied classes."""

    def source_classes(self) -> dict[str, str]:
        """Local ids assigned alphabetically over the wanted COCO names."""
        wanted = sorted(REMAP_TABLES["coco"])
        return {str(i): name for i, name in enumerate(wanted)}

    def _query_extras(self) -> dict[str, Any]:
        return {
            "split": self._split(),
            "class_caps": self.source.options.get("class_caps", {}),
            "classes": sorted(REMAP_TABLES["coco"]),
        }

    def _split(self) -> str:
        key = "smoke_split" if self.config.mode == "smoke" else "full_split"
        return str(self.source.options.get(key, "val2017"))

    # ── Annotation index ────────────────────────────────────────────────

    def load_instances(self) -> dict[str, Any]:
        """Download/extract the instances JSON for the configured split.

        Returns:
            Parsed COCO instances dict (categories/images/annotations).

        Raises:
            RuntimeError: If the annotations archive cannot be fetched.
        """
        split = self._split()
        annotations_url = str(self.source.options["annotations_url"])
        archive = self.downloads_dir / Path(annotations_url).name
        member = f"annotations/instances_{split}.json"
        extracted = self.downloads_dir / f"instances_{split}.json"

        if not extracted.exists():
            if not self.fetch_url(annotations_url, archive):
                raise RuntimeError(f"Could not download COCO annotations: {annotations_url}")
            logger.info(f"[coco] extracting {member} from {archive.name}")
            with zipfile.ZipFile(archive) as zf, zf.open(member) as src:
                extracted.write_bytes(src.read())

        logger.info(f"[coco] parsing {extracted.name} (may take a while)")
        return json.loads(extracted.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def build_image_class_index(self) -> dict[str, set[str]]:
        """COCO file_name → set of COCO class names present.

        Used by the negatives collector to find images containing none of
        the taxonomy classes.
        """
        data = self.load_instances()
        cat_names = {c["id"]: c["name"] for c in data["categories"]}
        file_names = {img["id"]: img["file_name"] for img in data["images"]}
        index: dict[str, set[str]] = {name: set() for name in file_names.values()}
        for ann in data["annotations"]:
            file_name = file_names.get(ann["image_id"])
            if file_name is not None:
                index[file_name].add(cat_names.get(ann["category_id"], "?"))
        return index

    # ── Acquisition ─────────────────────────────────────────────────────

    def fetch(self, limit: int | None) -> dict[str, int]:
        """Select capped images per class and fetch them individually."""
        data = self.load_instances()
        split = self._split()

        wanted_names = set(REMAP_TABLES["coco"])
        local_ids = {name: int(i) for i, name in self.source_classes().items()}
        cat_id_to_name = {
            c["id"]: c["name"] for c in data["categories"] if c["name"] in wanted_names
        }

        images_by_id = {img["id"]: img for img in data["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            if ann.get("iscrowd") or ann["category_id"] not in cat_id_to_name:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        caps: dict[str, int] = {
            str(k): int(v) for k, v in (self.source.options.get("class_caps") or {}).items()
        }
        url_template = str(self.source.options["image_url_template"])

        class_counts: dict[str, int] = {}
        selected = 0
        skipped_for_caps = 0
        failed = 0

        for image_id in sorted(anns_by_image):  # deterministic order
            if limit is not None and selected >= limit:
                break

            image_info = images_by_id[image_id]
            annotations = anns_by_image[image_id]

            # Cap check: skip the image when EVERY wanted class in it is
            # already at its cap (an image is kept if it still contributes).
            contributing = [
                ann
                for ann in annotations
                if class_counts.get(cat_id_to_name[ann["category_id"]], 0)
                < caps.get(cat_id_to_name[ann["category_id"]], 10**9)
            ]
            if not contributing:
                skipped_for_caps += 1
                continue

            boxes: list[tuple[int, float, float, float, float]] = []
            for ann in annotations:
                yolo = coco_bbox_to_yolo(
                    ann["bbox"], float(image_info["width"]), float(image_info["height"])
                )
                if yolo is None:
                    continue
                name = cat_id_to_name[ann["category_id"]]
                boxes.append((local_ids[name], *yolo))
            if not boxes:
                continue

            file_name = str(image_info["file_name"])
            url = url_template.format(split=split, file_name=file_name)
            if not self.fetch_url(url, self.images_dir / file_name):
                failed += 1
                continue

            write_yolo_label(self.labels_dir / f"{Path(file_name).stem}.txt", boxes)
            for ann in annotations:
                name = cat_id_to_name[ann["category_id"]]
                class_counts[name] = class_counts.get(name, 0) + 1
            selected += 1

            if selected % _LOG_EVERY == 0:
                logger.info(f"[coco] {selected} images fetched…")

        logger.info(
            f"[coco] done: {selected} images, {failed} fetch failures, "
            f"{skipped_for_caps} skipped by caps; counts={class_counts}"
        )
        return class_counts

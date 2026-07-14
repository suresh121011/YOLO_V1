"""
src.dataset.downloaders.openimages — Open Images V7 Subset Downloader
=====================================================================

CSV-index-first acquisition for the three Open Images classes (Door,
Cupboard, Gas stove):

    1. Download the class-descriptions CSV (display name → MID label).
    2. Download the bbox annotations CSV for the configured split
       (validation ≈ 24 MB in smoke mode; train ≈ 2 GB in full mode) and
       STREAM it row-by-row — the train CSV is never loaded into memory.
    3. Fetch only the selected images from the public S3 mirror.

Open Images bbox coordinates are already normalized (XMin/XMax/YMin/YMax),
so YOLO conversion is direct.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from src.dataset.downloaders.base import BaseDownloader, write_yolo_label
from src.dataset.remap import REMAP_TABLES

logger = logging.getLogger(__name__)

_LOG_EVERY = 20


class OpenImagesDownloader(BaseDownloader):
    """Open Images V7 subset acquisition (Door / Cupboard / Gas stove)."""

    def source_classes(self) -> dict[str, str]:
        """Local ids assigned alphabetically over the wanted display names."""
        wanted = sorted(self.source.options.get("classes") or REMAP_TABLES["openimages"])
        return {str(i): name for i, name in enumerate(wanted)}

    def _query_extras(self) -> dict[str, Any]:
        return {"split": self._split(), "classes": sorted(self._wanted_names())}

    def _split(self) -> str:
        key = "smoke_split" if self.config.mode == "smoke" else "full_split"
        return str(self.source.options.get(key, "validation"))

    def _wanted_names(self) -> set[str]:
        return set(self.source.options.get("classes") or REMAP_TABLES["openimages"])

    def _bbox_url(self) -> str:
        key = "smoke_bbox_url" if self.config.mode == "smoke" else "full_bbox_url"
        return str(self.source.options[key])

    def _load_mid_map(self) -> dict[str, str]:
        """MID label (e.g. /m/02dgv) → display name, for wanted classes only."""
        url = str(self.source.options["class_descriptions_url"])
        dest = self.downloads_dir / Path(url).name
        if not self.fetch_url(url, dest):
            raise RuntimeError(f"Could not download class descriptions: {url}")

        wanted = self._wanted_names()
        mid_map: dict[str, str] = {}
        with open(dest, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[1] in wanted:
                    mid_map[row[0]] = row[1]

        missing = wanted - set(mid_map.values())
        if missing:
            raise RuntimeError(f"Classes not found in Open Images descriptions: {missing}")
        return mid_map

    def fetch(self, limit: int | None) -> dict[str, int]:
        """Stream the bbox CSV, select capped images, fetch them per-URL."""
        mid_map = self._load_mid_map()
        local_ids = {name: int(i) for i, name in self.source_classes().items()}
        split = self._split()
        url_template = str(self.source.options["image_url_template"])

        bbox_url = self._bbox_url()
        bbox_csv = self.downloads_dir / Path(bbox_url).name
        if not self.fetch_url(bbox_url, bbox_csv):
            raise RuntimeError(f"Could not download bbox annotations: {bbox_url}")

        # Stream: collect wanted boxes per image id (only wanted labels kept,
        # so memory stays proportional to the subset, not the CSV).
        boxes_by_image: dict[str, list[tuple[int, float, float, float, float]]] = {}
        with open(bbox_csv, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            col = {name: idx for idx, name in enumerate(header)}
            for row in reader:
                display = mid_map.get(row[col["LabelName"]])
                if display is None:
                    continue
                x_min = float(row[col["XMin"]])
                x_max = float(row[col["XMax"]])
                y_min = float(row[col["YMin"]])
                y_max = float(row[col["YMax"]])
                width = x_max - x_min
                height = y_max - y_min
                if width <= 0 or height <= 0:
                    continue
                boxes_by_image.setdefault(row[col["ImageID"]], []).append(
                    (
                        local_ids[display],
                        x_min + width / 2,
                        y_min + height / 2,
                        width,
                        height,
                    )
                )

        logger.info(f"[openimages] {len(boxes_by_image)} candidate images in {split}")

        id_to_name = {v: k for k, v in local_ids.items()}
        class_counts: dict[str, int] = {}
        selected = 0
        failed = 0

        for image_id in sorted(boxes_by_image):  # deterministic
            if limit is not None and selected >= limit:
                break
            url = url_template.format(split=split, image_id=image_id)
            if not self.fetch_url(url, self.images_dir / f"{image_id}.jpg"):
                failed += 1
                continue
            boxes = boxes_by_image[image_id]
            write_yolo_label(self.labels_dir / f"{image_id}.txt", boxes)
            for local_id, *_ in boxes:
                name = id_to_name[local_id]
                class_counts[name] = class_counts.get(name, 0) + 1
            selected += 1
            if selected % _LOG_EVERY == 0:
                logger.info(f"[openimages] {selected} images fetched…")

        logger.info(
            f"[openimages] done: {selected} images, {failed} fetch failures; "
            f"counts={class_counts}"
        )
        return class_counts

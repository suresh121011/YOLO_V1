"""
src.dataset.downloaders.negatives_dl — Negative (Background) Collector
======================================================================

Selects indoor COCO images verified to contain NONE of the taxonomy's
COCO-mapped classes and stores them with empty label files (background
examples for false-positive reduction). Reuses the COCO downloader's
cached annotation index — running the COCO stage first avoids a second
annotations download.
"""

from __future__ import annotations

import logging
from typing import Any

from src.dataset.downloaders.base import BaseDownloader
from src.dataset.downloaders.coco import CocoDownloader
from src.dataset.negatives import select_negative_candidates
from src.dataset.remap import REMAP_TABLES

logger = logging.getLogger(__name__)

# COCO categories that overlap or could be confused with taxonomy classes;
# a negative image must contain none of these.
_EXCLUDED_COCO_CLASSES: frozenset[str] = frozenset(REMAP_TABLES["coco"]) | frozenset(
    {"couch", "dining table", "potted plant", "cup", "wine glass", "cell phone", "remote"}
)


class NegativesDownloader(BaseDownloader):
    """Background-image acquisition from the COCO index."""

    def source_classes(self) -> dict[str, str]:
        return {}  # negatives carry no annotations by definition

    def _query_extras(self) -> dict[str, Any]:
        return {"excluded_classes": sorted(_EXCLUDED_COCO_CLASSES)}

    def fetch(self, limit: int | None) -> dict[str, int]:
        """Pick and fetch negative candidates; write empty labels."""
        coco_source = self.config.sources.get("coco")
        if coco_source is None:
            raise RuntimeError("negatives require the 'coco' source in dataset_sources.yaml")
        coco = CocoDownloader(coco_source, self.config)
        coco.downloads_dir.mkdir(parents=True, exist_ok=True)

        count_key = "smoke_count" if self.config.mode == "smoke" else "full_count"
        count = int(self.source.options.get(count_key, 20))
        if limit is not None:
            count = min(count, limit)

        index = coco.build_image_class_index()
        selected = select_negative_candidates(
            index, excluded_classes=set(_EXCLUDED_COCO_CLASSES), count=count
        )

        split = coco._split()  # noqa: SLF001 — deliberate reuse of the cached split
        url_template = str(coco_source.options["image_url_template"])

        fetched = 0
        for file_name in selected:
            url = url_template.format(split=split, file_name=file_name)
            dest = self.images_dir / file_name
            if not self.fetch_url(url, dest):
                continue
            (self.labels_dir / f"{dest.stem}.txt").write_text("", encoding="utf-8")
            fetched += 1

        logger.info(f"[negatives] done: {fetched} background images")
        return {}

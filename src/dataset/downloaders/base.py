"""
src.dataset.downloaders.base — Common Downloader Interface
==========================================================

Shared plumbing for all acquisition sources: retrying HTTP fetches,
resume-by-skip semantics, standard output layout, and manifest writing.

Output layout produced by every downloader::

    <output_dir>/
    ├── images/              YOLO-ready images
    ├── labels/              YOLO labels using LOCAL ids (pre-remap)
    ├── source_classes.json  local id → source class name
    ├── manifest.json        SourceManifest (provenance)
    └── _downloads/          cached archives/indexes (not images)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.dataset.manifest import MANIFEST_FILENAME, SourceManifest
from src.dataset.remap import SOURCE_CLASSES_FILENAME
from src.dataset.sources_config import SourceConfig, SourcesConfig
from src.utils.dataset_utils import compute_file_hash

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = 2.0


class DownloadSkippedError(Exception):
    """Raised when a source cannot run (missing credentials, license gate).

    CLIs treat this as a graceful skip (exit 0 with a warning), not an
    error — the QA stage reports the resulting class shortfalls.
    """


class BaseDownloader(ABC):
    """Abstract acquisition source.

    Subclasses implement :meth:`fetch` (do the actual work and return
    per-class annotation counts); the base class handles layout, the
    source-classes sidecar, hashing, and manifest writing.
    """

    def __init__(self, source: SourceConfig, config: SourcesConfig) -> None:
        """
        Args:
            source: This source's entry from configs/dataset_sources.yaml.
            config: The full acquisition config (mode, limits, gates).
        """
        self.source = source
        self.config = config
        self.output_dir = source.output_dir
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        # Archives/indexes cache OUTSIDE the DVC-managed output dir so that
        # `dvc repro` (which clears stage outs) never re-downloads them.
        self.downloads_dir = config.downloads_cache / source.name

    # ── Subclass contract ────────────────────────────────────────────────

    @abstractmethod
    def fetch(self, limit: int | None) -> dict[str, int]:
        """Download images+labels into the standard layout.

        Args:
            limit: Max images to acquire (smoke cap) or None for full.

        Returns:
            Per-source-class annotation counts (source class names).

        Raises:
            DownloadSkippedError: When the source cannot run in this environment.
        """

    @abstractmethod
    def source_classes(self) -> dict[str, str]:
        """Local label id (as string) → source class name."""

    # ── Template method ──────────────────────────────────────────────────

    def download(self, limit: int | None = None) -> SourceManifest:
        """Run the acquisition and write sidecar + manifest.

        Args:
            limit: Per-source image cap; defaults to the config's mode-based
                   limit (smoke cap or None).

        Returns:
            The written :class:`SourceManifest`.
        """
        effective_limit = limit if limit is not None else self.config.limit
        logger.info(
            f"[{self.source.name}] acquisition starting "
            f"(mode={self.config.mode}, limit={effective_limit})"
        )

        for directory in (self.images_dir, self.labels_dir, self.downloads_dir):
            directory.mkdir(parents=True, exist_ok=True)

        class_counts = self.fetch(effective_limit)

        (self.output_dir / SOURCE_CLASSES_FILENAME).write_text(
            json.dumps(self.source_classes(), indent=2) + "\n", encoding="utf-8"
        )

        images = sorted(self.images_dir.glob("*"))
        manifest = SourceManifest(
            source=self.source.name,
            license=self.source.license,
            url=str(self.source.options.get("annotations_url", ""))
            or str(self.source.options.get("class_descriptions_url", "")),
            query={
                "mode": self.config.mode,
                "limit": effective_limit,
                **self._query_extras(),
            },
            image_count=len(images),
            class_counts=class_counts,
            trusted_classes=list(self.source.trusted_classes),
            image_hashes={p.name: compute_file_hash(p) for p in images if p.is_file()},
            notes=f"{self.config.mode}-mode acquisition",
        )
        manifest.save(self.output_dir / MANIFEST_FILENAME)
        logger.info(f"[{self.source.name}] acquisition complete: {manifest.image_count} images")
        return manifest

    def _query_extras(self) -> dict[str, Any]:
        """Extra reproducibility parameters recorded in the manifest."""
        return {}

    # ── Shared HTTP helpers ──────────────────────────────────────────────

    def fetch_url(
        self,
        url: str,
        dest: Path,
        retries: int = DEFAULT_RETRIES,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> bool:
        """Download ``url`` to ``dest`` with retries; skip if already present.

        Args:
            url:     Source URL.
            dest:    Destination file path.
            retries: Attempts before giving up.
            timeout: Per-request timeout (seconds).

        Returns:
            True if the file is present after the call (downloaded or
            already cached), False if every attempt failed.
        """
        import requests

        if dest.exists() and dest.stat().st_size > 0:
            logger.debug(f"[{self.source.name}] cached: {dest.name}")
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            fh.write(chunk)
                tmp.replace(dest)
                return True
            except Exception as e:  # noqa: BLE001 — network errors are retried
                logger.warning(
                    f"[{self.source.name}] attempt {attempt}/{retries} failed " f"for {url}: {e}"
                )
                tmp.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(RETRY_BACKOFF_S * attempt)

        logger.error(f"[{self.source.name}] giving up on {url}")
        return False


def write_yolo_label(
    dest: Path,
    boxes: list[tuple[int, float, float, float, float]],
) -> None:
    """Write a YOLO label file from (class_id, cx, cy, w, h) tuples."""
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in boxes]
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def coco_bbox_to_yolo(
    bbox: list[float],
    img_w: float,
    img_h: float,
) -> tuple[float, float, float, float] | None:
    """Convert COCO [x, y, w, h] (absolute) to YOLO (cx, cy, w, h) normalized.

    Returns None for degenerate boxes (non-positive size after clamping).
    """
    if img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = bbox
    # Clamp to the image bounds before normalizing.
    x1 = min(max(x, 0.0), img_w)
    y1 = min(max(y, 0.0), img_h)
    x2 = min(max(x + w, 0.0), img_w)
    y2 = min(max(y + h, 0.0), img_h)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:  # sub-pixel boxes are annotation noise
        return None
    return (
        (x1 + bw / 2) / img_w,
        (y1 + bh / 2) / img_h,
        bw / img_w,
        bh / img_h,
    )

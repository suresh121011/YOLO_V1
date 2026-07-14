"""
src.dataset.downloaders.roboflow_dl — Roboflow Universe Downloader
==================================================================

Pulls configured Roboflow Universe datasets (medicine_bottle, charger,
wire, gas_cylinder classes) via the official ``roboflow`` SDK — the one
sanctioned exception to the bespoke-downloader rule (SDK exports are
painful to hand-roll).

Graceful-skip contract (never a crash):
    - ``datasets: []`` in config        → DownloadSkippedError
    - ROBOFLOW_API_KEY env var unset    → DownloadSkippedError
    - ``roboflow`` package not installed→ DownloadSkippedError

Each configured dataset is exported in YOLO format; its labels keep the
export's local ids, offset so ids stay unique across datasets, and the
combined mapping is written to ``source_classes.json`` for the remap
stage (aliases resolved via datasets[].classes in dataset_sources.yaml).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from src.dataset.downloaders.base import BaseDownloader, DownloadSkippedError
from src.utils.config_helpers import load_yaml
from src.utils.dataset_utils import find_image_files

logger = logging.getLogger(__name__)


class RoboflowDownloader(BaseDownloader):
    """Roboflow Universe acquisition for specialty classes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._combined_classes: dict[str, str] = {}

    def source_classes(self) -> dict[str, str]:
        return dict(self._combined_classes)

    def _query_extras(self) -> dict[str, Any]:
        return {
            "datasets": [
                f"{entry.get('slug')}:{entry.get('version')}"
                for entry in self.source.options.get("datasets") or []
            ]
        }

    def fetch(self, limit: int | None) -> dict[str, int]:
        """Export each configured Universe dataset and consolidate it."""
        datasets: list[dict[str, Any]] = self.source.options.get("datasets") or []
        if not datasets:
            raise DownloadSkippedError(
                "no Roboflow Universe datasets configured — add entries under "
                "sources.roboflow.datasets in configs/dataset_sources.yaml "
                "(each needs slug, version, license, classes)"
            )

        api_key_env = str(self.source.options.get("api_key_env", "ROBOFLOW_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise DownloadSkippedError(
                f"{api_key_env} is not set — create a free key at roboflow.com "
                f"and export it before running this stage"
            )

        try:
            from roboflow import Roboflow
        except ImportError as e:
            raise DownloadSkippedError(
                "the 'roboflow' package is not installed — pip install roboflow"
            ) from e

        client = Roboflow(api_key=api_key)
        class_counts: dict[str, int] = {}
        id_offset = 0
        remaining = limit

        for entry in datasets:
            slug = str(entry["slug"])  # "workspace/project"
            version = int(entry.get("version", 1))
            workspace, project_name = slug.split("/", 1)

            logger.info(f"[roboflow] exporting {slug} v{version} (YOLO format)")
            export_dir = self.downloads_dir / f"{slug.replace('/', '_')}_v{version}"
            if not export_dir.exists():
                project = client.workspace(workspace).project(project_name)
                project.version(version).download(
                    "yolov8", location=str(export_dir), overwrite=False
                )

            counts, n_classes, copied = self._consolidate_export(export_dir, id_offset, remaining)
            for name, count in counts.items():
                class_counts[name] = class_counts.get(name, 0) + count
            if remaining is not None:
                remaining = max(remaining - copied, 0)
            id_offset += n_classes

        return class_counts

    def _consolidate_export(
        self,
        export_dir: Path,
        id_offset: int,
        limit: int | None,
    ) -> tuple[dict[str, int], int, int]:
        """Copy one export's images/labels into the standard layout.

        Label ids are shifted by ``id_offset`` so ids from multiple
        datasets never collide; the export's data.yaml names extend the
        combined source_classes map.

        Returns:
            (per-class annotation counts, number of classes in the export,
            number of images copied — used to decrement the cross-dataset
            image budget)
        """
        data_yaml = export_dir / "data.yaml"
        names_raw = load_yaml(data_yaml).get("names", [])
        names = (
            {int(k): str(v) for k, v in names_raw.items()}
            if isinstance(names_raw, dict)
            else dict(enumerate(str(n) for n in names_raw))
        )
        for local_id, name in names.items():
            self._combined_classes[str(local_id + id_offset)] = name

        counts: dict[str, int] = {}
        copied = 0
        for subset in ("train", "valid", "test"):
            images = find_image_files(export_dir / subset / "images")
            for img_path in images:
                if limit is not None and copied >= limit:
                    return counts, len(names), copied
                label_path = export_dir / subset / "labels" / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                shifted: list[str] = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    local_id = int(parts[0])
                    shifted.append(" ".join([str(local_id + id_offset), *parts[1:]]))
                    name = names.get(local_id, f"class_{local_id}")
                    counts[name] = counts.get(name, 0) + 1
                if not shifted:
                    continue

                flat = f"{export_dir.name}_{img_path.name}"
                shutil.copy2(img_path, self.images_dir / flat)
                (self.labels_dir / f"{Path(flat).stem}.txt").write_text(
                    "\n".join(shifted) + "\n", encoding="utf-8"
                )
                copied += 1

        return counts, len(names), copied

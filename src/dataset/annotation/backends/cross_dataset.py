"""
src.dataset.annotation.backends.cross_dataset — L3 Near-Dup Candidates (ADR-P5-08)
====================================================================================

No ML: reads ``data/merged/cross_dataset_links.json`` (built by
``merge.py``'s exact-vs-near-dup salvage, ``src/dataset/cross_dataset_salvage.py``)
and passes the linked near-duplicate twin's boxes through as ordinary,
human-verified candidates — never trusted directly. Exact-sha256
duplicates never reach this backend; they were already transplanted
directly onto the kept image's label file at merge time (safe, since
byte-identical images share exact geometry).

Layering note: reads the SAME link-file schema ``merge.py`` writes, but
does not import ``src.dataset.cross_dataset_salvage`` (nothing there is
needed at read time — just the JSON shape) so this stays a plain
``src/dataset/annotation`` -> nothing-lower dependency.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import (
    AutoAnnotator,
    BackendConfig,
    Detection,
    ModelFingerprint,
    prompt_fingerprint,
)
from src.dataset.annotation.registry import register_annotator

logger = logging.getLogger(__name__)

#: Deterministic passthrough — no model uncertainty. The CVAT review step is
#: where a human actually judges these, exactly like any other candidate.
LINKED_CANDIDATE_CONFIDENCE = 1.0

DEFAULT_LINKS_PATH = "data/merged/cross_dataset_links.json"


@register_annotator("cross_dataset")
class CrossDatasetBackend(AutoAnnotator):
    """Surfaces near-dup L3 links as candidates — no model inference."""

    def __init__(self) -> None:
        self._config: BackendConfig | None = None
        self._links: dict[str, list[dict[str, Any]]] = {}

    def load(self, config: BackendConfig, device: str, ids_by_name: Mapping[str, int]) -> None:
        """Load the merge-time link file (JSON, not a model).

        Args:
            config:      Backend configuration (``extra["links_path"]``).
            device:      Unused (no ML) — accepted to satisfy the ABC.
            ids_by_name: Unused — link boxes already carry taxonomy class ids.
        """
        self._config = config
        links_path = Path(str(config.extra.get("links_path", DEFAULT_LINKS_PATH)))
        if links_path.exists():
            self._links = json.loads(links_path.read_text(encoding="utf-8"))
        else:
            logger.warning(
                f"cross_dataset: no links file at {links_path} — 0 candidates this run "
                f"(expected before the first `dvc repro merge_datasets`)."
            )
            self._links = {}

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        """Return the image's linked near-dup boxes, filtered to targeted classes.

        Raises:
            RuntimeError: If called before ``load``.
        """
        if self._config is None:
            raise RuntimeError("CrossDatasetBackend.load() must be called before annotate()")
        targets = set(target_class_ids)
        detections: list[Detection] = []
        for entry in self._links.get(image_path.name, []):
            source = str(entry.get("source", ""))
            for box in entry.get("boxes", []):
                class_id, cx, cy, w, h = box
                if int(class_id) not in targets:
                    continue
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        conf=LINKED_CANDIDATE_CONFIDENCE,
                        bbox_xywhn=(float(cx), float(cy), float(w), float(h)),
                        refined=False,
                        origin=f"cross_dataset:{source}",
                    )
                )
        return detections

    def fingerprint(self) -> ModelFingerprint:
        """Reproducibility record (post-``load`` only). No weights — model-free."""
        if self._config is None:
            raise RuntimeError("CrossDatasetBackend.load() must be called before fingerprint()")
        return ModelFingerprint(
            backend="cross_dataset",
            weights_path="",
            weights_sha256="",
            library_versions={},
            device="",
            prompt_fingerprint=prompt_fingerprint(self._config.prompts, self._config.thresholds),
        )

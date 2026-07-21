"""
src.dataset.merge — Multi-Source Dataset Merge with Lineage
===========================================================

Merges remapped raw sources into ``data/merged/{images,labels}`` applying,
in order: label presence check → quality/indoor filter → flip-robust
perceptual dedup (cross-source, BEFORE split — governance rule). Every
accepted image is renamed ``{source}_{original}`` (collision-safe, and the
prefix keeps capture-group extraction source-scoped) and recorded in
``merged_manifest.json`` together with per-source acceptance stats and the
label-completeness map.

Source order matters: dedup keeps the FIRST occurrence, so list
higher-priority sources (e.g. custom captures) first.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.dedup import DedupIndex
from src.dataset.filters import check_image_filter
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings
from src.utils.annotation_utils import parse_label_file
from src.utils.dataset_utils import get_image_label_pairs

logger = logging.getLogger(__name__)


@dataclass
class MergeSource:
    """One input to the merge stage.

    Attributes:
        name:                Source identifier (prefixes merged filenames).
        root:                Raw source dir containing images/ and labels/.
        trusted_classes:     Exhaustively labeled classes (propagated into
                             the merged manifest's label_completeness map).
        apply_indoor_filter: Whether the indoor/quality heuristics run for
                             this source (e.g. True for coco/openimages,
                             False for wider_face/custom captures).
        allow_empty_labels:  Accept images whose label file is empty
                             (negatives source); missing-label images are
                             always dropped.
        labels_dir:          Override for the labels location (the remap
                             stage writes taxonomy labels to
                             data/interim/<source>/labels while images stay
                             under root). Defaults to root/labels.
    """

    name: str
    root: Path
    trusted_classes: list[str] = field(default_factory=list)
    apply_indoor_filter: bool = False
    allow_empty_labels: bool = False
    labels_dir: Path | None = None


def merge_sources(
    sources: list[MergeSource],
    output_dir: Path,
    dedup_settings: DedupSettings,
    indoor_settings: IndoorFilterSettings,
    class_names: dict[int, str],
    notes: str = "",
) -> MergedManifest:
    """Merge raw sources into a single YOLO-format pool with lineage.

    Args:
        sources:         Inputs, in priority order (first wins dedup ties).
        output_dir:      Destination root (images/ and labels/ created).
        dedup_settings:  Perceptual-dedup thresholds.
        indoor_settings: Quality/indoor filter thresholds.
        class_names:     Taxonomy id → name (from configs/data.yaml).
        notes:           Free-text note stored in the merged manifest.

    Returns:
        The :class:`MergedManifest` (also written to
        ``output_dir/merged_manifest.json``).
    """
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    manifest = MergedManifest(notes=notes)
    dedup_index = DedupIndex(settings=dedup_settings)

    for source in sources:
        stats: dict[str, Any] = {
            "source": source.name,
            "total": 0,
            "accepted": 0,
            "duplicates": 0,
            "filtered_out": 0,
            "missing_labels": 0,
        }
        labels_dir = source.labels_dir if source.labels_dir is not None else source.root / "labels"
        pairs = get_image_label_pairs(source.root / "images", labels_dir)

        for img_path, lbl_path in pairs:
            stats["total"] += 1

            if lbl_path is None:
                stats["missing_labels"] += 1
                logger.warning(f"[{source.name}] no label for {img_path.name} — dropped")
                continue

            if not source.allow_empty_labels and not lbl_path.read_text(encoding="utf-8").strip():
                stats["missing_labels"] += 1
                logger.warning(f"[{source.name}] empty label for {img_path.name} — dropped")
                continue

            if source.apply_indoor_filter:
                keep, reason = check_image_filter(img_path, indoor_settings)
                if not keep:
                    stats["filtered_out"] += 1
                    manifest.filtered_out += 1
                    logger.debug(f"[{source.name}] {img_path.name} rejected: {reason}")
                    continue

            duplicate_of = dedup_index.check_and_add(img_path)
            if duplicate_of is not None:
                stats["duplicates"] += 1
                manifest.duplicates_removed += 1
                logger.debug(f"[{source.name}] {img_path.name} duplicate of {duplicate_of.name}")
                continue

            merged_name = f"{source.name}_{img_path.name}"
            shutil.copy2(img_path, images_out / merged_name)
            shutil.copy2(lbl_path, labels_out / f"{source.name}_{img_path.stem}.txt")

            manifest.image_provenance[merged_name] = source.name
            for ann in parse_label_file(lbl_path):
                name = class_names.get(ann.class_id, f"class_{ann.class_id}")
                manifest.class_counts[name] = manifest.class_counts.get(name, 0) + 1
            stats["accepted"] += 1

        manifest.sources.append(stats)
        manifest.label_completeness[source.name] = list(source.trusted_classes)
        logger.info(
            f"Merged '{source.name}': {stats['accepted']}/{stats['total']} accepted "
            f"({stats['duplicates']} dup, {stats['filtered_out']} filtered, "
            f"{stats['missing_labels']} label issues)"
        )

    manifest.save(output_dir / MERGED_MANIFEST_FILENAME)
    logger.info(
        f"Merge complete: {len(manifest.image_provenance)} images → {output_dir} "
        f"({manifest.duplicates_removed} duplicates removed)"
    )
    return manifest

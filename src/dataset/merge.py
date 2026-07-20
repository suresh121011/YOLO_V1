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

L3 label salvage (ADR-P5-08, D7): a duplicate is never simply discarded.
Byte-identical (exact-sha256) duplicates transplant the dropped twin's
trusted-class boxes onto the kept image (``cross_dataset_salvage``);
near-dup (perceptual-only) duplicates record a link consumed later by the
``cross_dataset`` auto-annotation backend, surfacing them as ordinary
human-verified candidates instead.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.cross_dataset_salvage import (
    build_cross_dataset_link,
    render_transplanted_lines,
    transplant_trusted_boxes,
)
from src.dataset.dedup import DedupIndex
from src.dataset.filters import check_image_filter
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings
from src.utils.annotation_utils import parse_label_file
from src.utils.dataset_utils import compute_file_hash, get_image_label_pairs

logger = logging.getLogger(__name__)

#: Written alongside merged_manifest.json — near-dup L3 links consumed by
#: the `cross_dataset` auto-annotation backend (M1 registry).
CROSS_DATASET_LINKS_FILENAME = "cross_dataset_links.json"


@dataclass
class _KeptImageInfo:
    """Bookkeeping for one already-accepted (kept) image, keyed by its
    ORIGINAL source path (the DedupIndex key) — needed because a later
    duplicate's `check_and_add` return value is that original path, not
    the renamed merged filename."""

    merged_name: str
    label_path: Path
    source_name: str
    trusted_classes: frozenset[str]


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
    kept_info: dict[Path, _KeptImageInfo] = {}
    cross_dataset_links: dict[str, list[dict[str, Any]]] = {}
    labels_salvaged = 0
    cross_dataset_candidates_linked = 0

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
                kept = kept_info.get(duplicate_of)
                if kept is not None:
                    dropped_annotations = parse_label_file(lbl_path)
                    if compute_file_hash(img_path) == compute_file_hash(duplicate_of):
                        # Exact-sha256 (D7): safe to transplant directly.
                        result = transplant_trusted_boxes(
                            dropped_annotations=dropped_annotations,
                            dropped_trusted_classes=frozenset(source.trusted_classes),
                            class_names_by_id=class_names,
                            kept_annotations=parse_label_file(kept.label_path),
                        )
                        if result.transplanted:
                            with open(kept.label_path, "a", encoding="utf-8") as f:
                                f.write("\n".join(render_transplanted_lines(result.transplanted)))
                                f.write("\n")
                            labels_salvaged += len(result.transplanted)
                            for ann in result.transplanted:
                                cname = class_names.get(ann.class_id, f"class_{ann.class_id}")
                                manifest.class_counts[cname] = (
                                    manifest.class_counts.get(cname, 0) + 1
                                )
                    else:
                        # Near-dup only: geometry unverified — link for the
                        # cross_dataset backend to surface as candidates.
                        link = build_cross_dataset_link(
                            dropped_annotations=dropped_annotations,
                            dropped_trusted_classes=frozenset(source.trusted_classes),
                            class_names_by_id=class_names,
                            dropped_source=source.name,
                        )
                        if link is not None:
                            cross_dataset_links.setdefault(kept.merged_name, []).append(link)
                            cross_dataset_candidates_linked += len(link["boxes"])  # type: ignore[arg-type]
                continue

            merged_name = f"{source.name}_{img_path.name}"
            merged_label_path = labels_out / f"{source.name}_{img_path.stem}.txt"
            shutil.copy2(img_path, images_out / merged_name)
            shutil.copy2(lbl_path, merged_label_path)
            kept_info[img_path] = _KeptImageInfo(
                merged_name=merged_name,
                label_path=merged_label_path,
                source_name=source.name,
                trusted_classes=frozenset(source.trusted_classes),
            )

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

    manifest.labels_salvaged = labels_salvaged
    manifest.cross_dataset_candidates_linked = cross_dataset_candidates_linked
    manifest.save(output_dir / MERGED_MANIFEST_FILENAME)
    (output_dir / CROSS_DATASET_LINKS_FILENAME).write_text(
        json.dumps(cross_dataset_links, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        f"Merge complete: {len(manifest.image_provenance)} images → {output_dir} "
        f"({manifest.duplicates_removed} duplicates removed, {labels_salvaged} labels "
        f"salvaged, {cross_dataset_candidates_linked} cross-dataset candidates linked)"
    )
    return manifest

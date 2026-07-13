"""
src.dataset.remap — Per-Source Class Remapping
==============================================

Rewrites YOLO label files from source-native class IDs into the locked
23-class taxonomy (configs/data.yaml). Tables follow
``docs/03_engineering_appendix/dataset_templates.md`` but are keyed by
source class *name* rather than numeric ID for readability (COCO name
"bottle" == COCO category 44, etc.).

Contract with downloaders:
    Each raw source directory contains ``labels/`` using LOCAL contiguous
    ids (0..k-1) plus a sidecar ``source_classes.json`` mapping local id →
    source class name, e.g. {"0": "person", "1": "bottle"}.

Remapping (``remap_label_dir``) then rewrites every label line from the
local id to the taxonomy id and drops annotations whose class has no
mapping. A ``.remap_done.json`` sentinel makes the operation idempotent —
re-running on already-remapped labels would silently corrupt IDs, so it is
refused unless ``force=True``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.dataset_utils import find_label_files

logger = logging.getLogger(__name__)

NUM_CLASSES = 23

SOURCE_CLASSES_FILENAME = "source_classes.json"
REMAP_SENTINEL_FILENAME = ".remap_done.json"

# Source class name → taxonomy class ID.
# COCO names correspond to docs' numeric table {1:0, 44:4, 49:5, 73:12,
# 72:13, 62:16, 65:17, 70:18, 81:19, 84:9}.
REMAP_TABLES: dict[str, dict[str, int]] = {
    "coco": {
        "person": 0,
        "bottle": 4,  # → water_bottle
        "knife": 5,
        "laptop": 12,
        "tv": 13,  # → monitor
        "chair": 16,
        "bed": 17,
        "toilet": 18,
        "sink": 19,
        "book": 9,
    },
    "openimages": {
        "Door": 15,
        "Cupboard": 14,
        "Gas stove": 6,  # → stove
    },
    "wider_face": {
        "face": 1,
    },
    # Roboflow Universe datasets vary; canonical names below, per-dataset
    # aliases can be added in configs/dataset_sources.yaml (datasets[].classes).
    "roboflow": {
        "medicine_bottle": 3,
        "charger": 10,
        "wire": 11,
        "gas_cylinder": 7,
    },
    # Custom captures and negatives are annotated directly in taxonomy IDs.
    "identity": {},
}


@dataclass
class RemapResult:
    """Outcome of remapping one source directory."""

    source_dir: Path
    files_processed: int = 0
    annotations_remapped: int = 0
    annotations_dropped: int = 0
    dropped_by_class: dict[str, int] = field(default_factory=dict)
    skipped: bool = False


def build_id_mapping(
    source_classes: dict[str, str],
    table: dict[str, int],
) -> dict[int, int | None]:
    """Build local-id → taxonomy-id mapping (None = drop annotation).

    Args:
        source_classes: Local id (as string) → source class name, from the
                        downloader's ``source_classes.json``.
        table:          Source class name → taxonomy id (REMAP_TABLES entry
                        plus any per-dataset aliases).

    Returns:
        Dict mapping local integer id to taxonomy id, or None when the
        source class has no taxonomy mapping.

    Raises:
        ValueError: If a mapped taxonomy id falls outside 0..NUM_CLASSES-1.
    """
    mapping: dict[int, int | None] = {}
    for local_id_str, class_name in source_classes.items():
        target = table.get(class_name)
        if target is not None and not 0 <= target < NUM_CLASSES:
            raise ValueError(
                f"Remap target {target} for '{class_name}' outside taxonomy "
                f"range 0..{NUM_CLASSES - 1}"
            )
        mapping[int(local_id_str)] = target
    return mapping


def remap_label_file(
    path: Path,
    mapping: dict[int, int | None],
    result: RemapResult,
    id_to_name: dict[int, str],
) -> None:
    """Rewrite one YOLO label file in place using the id mapping.

    Lines whose class maps to None (unmapped source class) are dropped and
    counted; malformed lines are dropped with a warning.
    """
    out_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            local_id = int(parts[0])
        except (ValueError, IndexError):
            logger.warning(f"Malformed label line dropped in {path.name}: '{line}'")
            result.annotations_dropped += 1
            continue

        if local_id not in mapping:
            logger.warning(f"Unknown local class id {local_id} dropped in {path.name}")
            result.annotations_dropped += 1
            continue

        target = mapping[local_id]
        if target is None:
            name = id_to_name.get(local_id, str(local_id))
            result.dropped_by_class[name] = result.dropped_by_class.get(name, 0) + 1
            result.annotations_dropped += 1
            continue

        out_lines.append(" ".join([str(target), *parts[1:]]))
        result.annotations_remapped += 1

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def remap_label_dir(
    source_dir: Path,
    table: dict[str, int],
    force: bool = False,
) -> RemapResult:
    """Remap all label files under ``source_dir/labels`` into taxonomy IDs.

    Args:
        source_dir: Raw source directory (contains labels/ and
                    source_classes.json).
        table:      Source class name → taxonomy id.
        force:      Re-run even if the remap sentinel exists (dangerous —
                    only for rebuilt label sets).

    Returns:
        :class:`RemapResult`. ``skipped=True`` when the sentinel exists.

    Raises:
        FileNotFoundError: If source_classes.json is missing.
        ValueError:        If the mapping targets invalid taxonomy ids.
    """
    result = RemapResult(source_dir=source_dir)
    sentinel = source_dir / REMAP_SENTINEL_FILENAME

    if sentinel.exists() and not force:
        logger.info(f"Remap skipped (already done): {source_dir}")
        result.skipped = True
        return result

    classes_path = source_dir / SOURCE_CLASSES_FILENAME
    if not classes_path.exists():
        raise FileNotFoundError(
            f"{SOURCE_CLASSES_FILENAME} not found in {source_dir} — "
            f"was this source downloaded by the pipeline?"
        )

    source_classes: dict[str, str] = json.loads(classes_path.read_text(encoding="utf-8"))
    mapping = build_id_mapping(source_classes, table)
    id_to_name = {int(k): v for k, v in source_classes.items()}

    label_files = find_label_files(source_dir / "labels")
    for label_path in label_files:
        remap_label_file(label_path, mapping, result, id_to_name)
        result.files_processed += 1

    sentinel.write_text(
        json.dumps(
            {
                "mapping": {str(k): v for k, v in mapping.items()},
                "files_processed": result.files_processed,
                "annotations_remapped": result.annotations_remapped,
                "annotations_dropped": result.annotations_dropped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        f"Remapped {source_dir.name}: {result.files_processed} files, "
        f"{result.annotations_remapped} kept, {result.annotations_dropped} dropped"
    )
    return result

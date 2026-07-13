"""
src.dataset.negatives — Background (Negative) Image Selection
=============================================================

Negative examples are indoor images verified to contain NONE of the
23 taxonomy classes. They are stored with intentionally empty label files
and teach the detector what background looks like (false-positive
reduction). Selection logic is source-agnostic; the COCO downloader feeds
it the per-image category index it already parsed.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def select_negative_candidates(
    image_class_index: dict[str, set[str]],
    excluded_classes: set[str],
    count: int,
    seed: int = 42,
) -> list[str]:
    """Pick images that contain none of the excluded (taxonomy) classes.

    Args:
        image_class_index: Image key → set of source class names present.
        excluded_classes:  Source class names that map to (or resemble)
                           taxonomy classes; any overlap disqualifies.
        count:             Number of negatives to select.
        seed:              Deterministic shuffle seed.

    Returns:
        Sorted-then-shuffled deterministic selection of image keys,
        at most ``count`` entries.
    """
    candidates = sorted(
        key for key, classes in image_class_index.items() if not classes & excluded_classes
    )
    # Deterministic sample for reproducible negative sets — not security.
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(candidates)
    selected = candidates[:count]
    logger.info(
        f"Selected {len(selected)}/{len(candidates)} negative candidates " f"(requested {count})"
    )
    return selected


def write_empty_labels(images_dir: Path, labels_dir: Path) -> int:
    """Create an empty YOLO label file for every image in ``images_dir``.

    Args:
        images_dir: Directory of negative images.
        labels_dir: Destination for the empty .txt files.

    Returns:
        Number of label files written.
    """
    from src.utils.dataset_utils import find_image_files

    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img in find_image_files(images_dir):
        (labels_dir / f"{img.stem}.txt").write_text("", encoding="utf-8")
        written += 1
    logger.info(f"Wrote {written} empty negative labels to {labels_dir}")
    return written

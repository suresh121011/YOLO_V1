"""
src.utils.annotation_utils — YOLO Annotation Parsing and Validation
====================================================================

Stateless helpers for reading, parsing, and validating YOLO-format annotation
files (.txt). Used by the QA pipeline and statistics generator.

YOLO format (per line):
    <class_id> <cx> <cy> <w> <h>

Where:
    class_id — integer class index (0-based)
    cx, cy   — bounding box center as fraction of image width/height [0, 1]
    w, h     — bounding box dimensions as fractions of image size [0, 1]

Usage:
    from src.utils.annotation_utils import parse_label_file, validate_yolo_line

    annotations = parse_label_file(Path("labels/train/image_001.txt"))
    for ann in annotations:
        errors = validate_yolo_line(ann, num_classes=23)
        if errors:
            print(errors)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ─── Data structures ──────────────────────────────────────────────────────────


class Annotation(NamedTuple):
    """A single parsed YOLO bounding box annotation.

    Args:
        class_id: Integer class index.
        cx:       Center x coordinate [0, 1].
        cy:       Center y coordinate [0, 1].
        w:        Bounding box width as fraction of image width [0, 1].
        h:        Bounding box height as fraction of image height [0, 1].
        line_num: 1-based line number in the source label file.
        raw:      Original raw line string (for error reporting).
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    line_num: int
    raw: str


# ─── Parsing ──────────────────────────────────────────────────────────────────


def parse_yolo_line(line: str, line_num: int = 0) -> Annotation | None:
    """Parse a single YOLO annotation line into an Annotation.

    Args:
        line:     Raw line string from a YOLO label file.
        line_num: 1-based line number for error context.

    Returns:
        Annotation if parsing succeeds, None if the line is empty or malformed.
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) != 5:
        logger.debug(f"Malformed line {line_num}: expected 5 fields, got {len(parts)}")
        return None

    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return Annotation(
            class_id=class_id, cx=cx, cy=cy, w=w, h=h, line_num=line_num, raw=stripped
        )
    except ValueError as e:
        logger.debug(f"Could not parse line {line_num}: {e}")
        return None


def parse_label_file(path: Path) -> list[Annotation]:
    """Parse all annotations from a YOLO label file.

    Silently skips empty lines and comment lines (starting with #).
    Logs a warning for malformed lines but continues parsing the rest.

    Args:
        path: Path to a YOLO .txt label file.

    Returns:
        List of Annotation objects. Returns empty list for empty files.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    annotations: list[Annotation] = []

    for line_num, line in enumerate(content.splitlines(), start=1):
        if line.strip().startswith("#"):
            continue
        ann = parse_yolo_line(line, line_num)
        if ann is not None:
            annotations.append(ann)

    return annotations


def parse_label_file_raw(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from a label file.

    Useful for detecting duplicate annotation lines.

    Args:
        path: Path to a YOLO .txt label file.

    Returns:
        List of stripped non-empty, non-comment lines.
    """
    if not path.exists():
        return []

    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


# ─── Validation ───────────────────────────────────────────────────────────────


def validate_yolo_line(ann: Annotation, num_classes: int) -> list[str]:
    """Validate a parsed annotation and return a list of error messages.

    Args:
        ann:         Parsed Annotation to validate.
        num_classes: Total number of valid classes (0 to num_classes-1).

    Returns:
        List of human-readable error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Class ID range check
    if ann.class_id < 0 or ann.class_id >= num_classes:
        errors.append(
            f"Line {ann.line_num}: invalid class_id={ann.class_id} "
            f"(valid range 0–{num_classes - 1})"
        )

    # Coordinate range checks [0, 1]
    for name, val in [("cx", ann.cx), ("cy", ann.cy), ("w", ann.w), ("h", ann.h)]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"Line {ann.line_num}: {name}={val:.6f} out of [0, 1] range")

    # Zero/negative area
    if ann.w <= 0.0 or ann.h <= 0.0:
        errors.append(f"Line {ann.line_num}: zero or negative box dimension (w={ann.w}, h={ann.h})")

    # Box extends beyond image boundaries
    x1 = ann.cx - ann.w / 2
    y1 = ann.cy - ann.h / 2
    x2 = ann.cx + ann.w / 2
    y2 = ann.cy + ann.h / 2

    if x1 < -1e-6 or y1 < -1e-6 or x2 > 1.0 + 1e-6 or y2 > 1.0 + 1e-6:
        errors.append(
            f"Line {ann.line_num}: bounding box extends outside image "
            f"(x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f})"
        )

    return errors


def check_bbox_bounds(cx: float, cy: float, w: float, h: float) -> list[str]:
    """Check YOLO bounding box coordinate validity.

    Convenience function for validating individual coordinates without
    constructing an Annotation object.

    Args:
        cx: Center x [0, 1].
        cy: Center y [0, 1].
        w:  Width [0, 1].
        h:  Height [0, 1].

    Returns:
        List of error strings. Empty list means all coordinates are valid.
    """
    errors: list[str] = []

    for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"{name}={val:.6f} out of valid [0, 1] range")

    if w <= 0.0 or h <= 0.0:
        errors.append(f"Zero or negative dimension: w={w}, h={h}")

    return errors


def check_zero_area(w: float, h: float) -> bool:
    """Return True if the bounding box has zero or negative area.

    Args:
        w: Bounding box width.
        h: Bounding box height.

    Returns:
        True if w <= 0 or h <= 0.
    """
    return w <= 0.0 or h <= 0.0


def check_duplicate_lines(lines: list[str]) -> list[tuple[int, int]]:
    """Find duplicate annotation lines within a single label file.

    Args:
        lines: Raw annotation lines from parse_label_file_raw().

    Returns:
        List of (first_occurrence_idx, duplicate_idx) pairs (0-based).
    """
    seen: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []

    for idx, line in enumerate(lines):
        if line in seen:
            duplicates.append((seen[line], idx))
        else:
            seen[line] = idx

    return duplicates


# ─── Aggregation ──────────────────────────────────────────────────────────────


def count_annotations_by_class(
    label_files: list[Path],
) -> dict[int, int]:
    """Count annotation instances per class across multiple label files.

    Args:
        label_files: List of YOLO .txt label file paths.

    Returns:
        Dict mapping class_id → instance count. Only includes classes
        that appear at least once.
    """
    counts: dict[int, int] = {}

    for label_path in label_files:
        try:
            for ann in parse_label_file(label_path):
                counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Could not read label file {label_path}: {e}")

    return counts


def count_annotations_by_class_and_split(
    split_label_dirs: dict[str, Path],
) -> dict[str, dict[int, int]]:
    """Count annotation instances per class for each dataset split.

    Args:
        split_label_dirs: Dict mapping split name to label directory Path
            (e.g., {"train": Path("data/processed/labels/train"), ...}).

    Returns:
        Nested dict: split_name → {class_id: count}.
    """
    from src.utils.dataset_utils import find_label_files

    result: dict[str, dict[int, int]] = {}
    for split_name, label_dir in split_label_dirs.items():
        files = find_label_files(label_dir)
        result[split_name] = count_annotations_by_class(files)
        logger.info(
            f"Split '{split_name}': {sum(result[split_name].values())} "
            f"annotations across {len(files)} label files"
        )
    return result

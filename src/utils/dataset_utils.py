"""
src.utils.dataset_utils — Dataset File Discovery and Grouping
=============================================================

Utilities for discovering image and label files, matching them into pairs,
computing file hashes for deduplication, and extracting group keys from
filenames to prevent data leakage during dataset splitting.

Design principles:
    - All functions are stateless (no class state)
    - All I/O errors are logged and propagated — callers decide recovery
    - Group-key extraction handles multiple naming conventions (see below)

Supported filename grouping conventions:
    video001_frame_00042.jpg    → group "video001"
    vid_001_042.jpg             → group "vid_001"
    IMG_20240601_143022_001.jpg → group "IMG_20240601_143022"
    custom_kitchen_001.jpg      → group "custom_kitchen"
    frame_00001.jpg             → no group (treated as standalone)

Usage:
    from src.utils.dataset_utils import find_image_files, get_image_label_pairs

    images = find_image_files(Path("data/processed/images"))
    pairs = get_image_label_pairs(
        Path("data/processed/images"),
        Path("data/processed/labels"),
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Supported file extensions ────────────────────────────────────────────────

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
LABEL_EXTENSION: str = ".txt"

# ─── Group key extraction patterns (tried in order) ──────────────────────────
# Each pattern must contain exactly one capture group that yields the group key.
# Patterns are tried top-to-bottom; first match wins.

_GROUP_PATTERNS: list[re.Pattern[str]] = [
    # video001_frame_00042  →  "video001"
    re.compile(r"^([a-zA-Z_]+\d{1,6})_frame_\d+$"),
    # vid_001_frame_042  →  "vid_001"
    re.compile(r"^((?:[a-zA-Z]+_\d+))_frame_\d+$"),
    # IMG_20240601_143022_001  →  "IMG_20240601_143022"
    re.compile(r"^(IMG_\d{8}_\d{6})_\d+$"),
    # custom_kitchen_001  →  "custom_kitchen"
    re.compile(r"^([a-zA-Z_]+)_\d{3,}$"),
    # sequence_name_0042  →  "sequence_name"
    re.compile(r"^([a-zA-Z][a-zA-Z0-9_]+)_\d+$"),
]


# ─── File Discovery ───────────────────────────────────────────────────────────


def find_image_files(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of image file Paths. Empty list if directory does not exist.
    """
    if not directory.exists():
        logger.warning(f"Image directory does not exist: {directory}")
        return []

    files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.debug(f"Found {len(files)} image files in {directory}")
    return files


def find_label_files(directory: Path) -> list[Path]:
    """Recursively find all YOLO label (.txt) files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of label file Paths. Empty list if directory does not exist.
    """
    if not directory.exists():
        logger.warning(f"Label directory does not exist: {directory}")
        return []

    files = sorted(p for p in directory.rglob(f"*{LABEL_EXTENSION}"))
    logger.debug(f"Found {len(files)} label files in {directory}")
    return files


def get_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path | None]]:
    """Match image files to their corresponding label files.

    Matching is done by stem (filename without extension). Labels are
    expected at the same relative path under labels_dir as images are
    under images_dir, but with a .txt extension.

    Args:
        images_dir: Root directory of images.
        labels_dir: Root directory of YOLO label files.

    Returns:
        List of (image_path, label_path_or_None) tuples. Label is None
        when no corresponding label file exists.
    """
    images = find_image_files(images_dir)
    pairs: list[tuple[Path, Path | None]] = []

    for img_path in images:
        # Preserve subdirectory structure
        try:
            rel = img_path.relative_to(images_dir)
        except ValueError:
            rel = Path(img_path.name)

        label_path = labels_dir / rel.with_suffix(LABEL_EXTENSION)
        pairs.append((img_path, label_path if label_path.exists() else None))

    missing = sum(1 for _, lbl in pairs if lbl is None)
    if missing:
        logger.warning(f"{missing}/{len(pairs)} images have no corresponding label file")

    return pairs


def build_label_index(labels_dir: Path) -> dict[str, Path]:
    """Build a stem → label path index for fast lookup.

    Args:
        labels_dir: Root directory of YOLO label files.

    Returns:
        Dict mapping file stem (e.g., "image_001") to its label Path.
    """
    return {p.stem: p for p in find_label_files(labels_dir)}


# ─── File Hashing ─────────────────────────────────────────────────────────────


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file.

    Reads in chunks to handle large image files without loading into memory.

    Args:
        path: Path to the file to hash.
        chunk_size: Read chunk size in bytes (default 64 KB).

    Returns:
        Lowercase hex SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_hash_index(
    files: list[Path],
    chunk_size: int = 65536,
) -> dict[str, list[Path]]:
    """Build a hash → [file paths] index for duplicate detection.

    Args:
        files: List of file paths to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        Dict mapping SHA-256 hash to list of files with that hash.
        Only hashes with 2+ files indicate duplicates.
    """
    index: dict[str, list[Path]] = {}
    for path in files:
        try:
            digest = compute_file_hash(path, chunk_size)
            index.setdefault(digest, []).append(path)
        except OSError as e:
            logger.warning(f"Could not hash {path}: {e}")

    duplicates = sum(1 for paths in index.values() if len(paths) > 1)
    if duplicates:
        logger.info(f"Hash index: {duplicates} duplicate groups found in {len(files)} files")

    return index


# ─── Group Key Extraction ─────────────────────────────────────────────────────


def extract_group_key(filename: str) -> str:
    """Extract a capture-session group key from an image filename.

    Used to keep frames from the same video or burst capture in the same
    dataset split, preventing temporal data leakage.

    Supported conventions (see module docstring for examples):
        - video001_frame_00042.jpg  → "video001"
        - IMG_20240601_143022_001.jpg → "IMG_20240601_143022"
        - custom_kitchen_001.jpg    → "custom_kitchen"
        - standalone_image.jpg      → "standalone_image" (no group found)

    Args:
        filename: Image filename (with or without extension).

    Returns:
        Group key string. Falls back to the full stem if no pattern matches.
    """
    stem = Path(filename).stem

    for pattern in _GROUP_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(1)

    # No pattern matched — treat each file as its own group
    return stem


def group_files_by_key(files: list[Path]) -> dict[str, list[Path]]:
    """Group files by their capture-session key.

    Args:
        files: List of file paths.

    Returns:
        Dict mapping group key to list of files in that group.
        Groups are sorted by key for deterministic iteration.
    """
    groups: dict[str, list[Path]] = {}
    for f in files:
        key = extract_group_key(f.name)
        groups.setdefault(key, []).append(f)

    logger.debug(f"Grouped {len(files)} files into {len(groups)} capture groups")
    return dict(sorted(groups.items()))

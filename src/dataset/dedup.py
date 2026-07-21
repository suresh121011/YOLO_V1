"""
src.dataset.dedup — Flip-Robust Perceptual Deduplication
========================================================

Duplicate detection across all sources at merge time, BEFORE splitting.
This ordering is a hard governance rule: duplicates that survive into
different splits silently inflate validation metrics.

Beyond the plain aHash used by QA (src/utils/image_utils), this module
also hashes the horizontally mirrored image when ``check_flips`` is
enabled. Roboflow Universe datasets are frequently pre-augmented copies of
COCO; a flipped twin defeats plain perceptual hashing but is caught here.

Complexity: near-duplicate search is O(n²) over 64-bit hashes with early
exact-match bucketing — fine up to a few tens of thousands of images
(one-time cost at merge). Swap in a BK-tree if the full build outgrows it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.sources_config import DedupSettings
from src.utils.dataset_utils import compute_file_hash
from src.utils.image_utils import compute_perceptual_hash

logger = logging.getLogger(__name__)

_HASH_SIZE = 8  # 64-bit aHash, matching src/utils/image_utils


@dataclass
class ImageHashes:
    """Perceptual hashes for one image (hex strings, or None if unhashable)."""

    ahash: str | None
    flip_ahash: str | None


@dataclass
class DedupIndex:
    """Incremental duplicate detector shared across all merged sources.

    Usage:
        index = DedupIndex(settings)
        duplicate_of = index.check_and_add(path)   # None → unique, kept
    """

    settings: DedupSettings
    # kept image path → (ahash int, flip int|None); insertion order preserved
    _kept: dict[Path, tuple[int, int | None]] = field(default_factory=dict)
    # exact-hash bucket for O(1) byte/near-exact hits
    _exact: dict[int, Path] = field(default_factory=dict)
    # SHA-256 fallback bucket for images PIL cannot hash
    _sha256: dict[str, Path] = field(default_factory=dict)

    def check_and_add(self, path: Path) -> Path | None:
        """Return the kept duplicate of ``path``, or None and register it.

        Duplicate rule: Hamming distance < ``hamming_threshold`` between
        aHashes, or between this image's flipped aHash and any kept aHash
        (when ``check_flips`` is on).
        """
        hashes = compute_image_hashes(path, check_flip=self.settings.check_flips)

        if hashes.ahash is None:
            # PIL unavailable/undecodable — fall back to byte-exact dedup.
            try:
                digest = compute_file_hash(path)
            except OSError:
                return None  # unreadable: let QA reject it later
            existing = self._sha256.get(digest)
            if existing is not None:
                return existing
            self._sha256[digest] = path
            return None

        a_int = int(hashes.ahash, 16)
        flip_int = int(hashes.flip_ahash, 16) if hashes.flip_ahash else None

        exact_hit = self._exact.get(a_int)
        if exact_hit is not None:
            return exact_hit

        threshold = self.settings.hamming_threshold
        for kept_path, (kept_a, _kept_flip) in self._kept.items():
            if _popcount(a_int ^ kept_a) < threshold:
                return kept_path
            if flip_int is not None and _popcount(flip_int ^ kept_a) < threshold:
                return kept_path

        self._kept[path] = (a_int, flip_int)
        self._exact[a_int] = path
        return None

    @property
    def kept_count(self) -> int:
        """Number of unique images registered so far."""
        return len(self._kept) + len(self._sha256)


def compute_image_hashes(path: Path, check_flip: bool = True) -> ImageHashes:
    """Compute aHash and (optionally) mirrored aHash for an image.

    Args:
        path:       Image file path.
        check_flip: Also hash the horizontally mirrored image.

    Returns:
        :class:`ImageHashes`; fields are None when PIL is unavailable or
        the image cannot be decoded.
    """
    ahash = compute_perceptual_hash(path, hash_size=_HASH_SIZE)
    if ahash is None or not check_flip:
        return ImageHashes(ahash=ahash, flip_ahash=None)

    return ImageHashes(ahash=ahash, flip_ahash=_compute_flipped_hash(path))


def _compute_flipped_hash(path: Path) -> str | None:
    """aHash of the horizontally mirrored image (same algorithm as aHash)."""
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as img:
            small = (
                img.convert("L")
                .transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                .resize((_HASH_SIZE, _HASH_SIZE), Image.Resampling.LANCZOS)
            )
            pixels = list(small.tobytes())  # mode "L" → one byte per pixel

        mean = sum(pixels) / len(pixels)
        bits = "".join("1" if p > mean else "0" for p in pixels)
        padded = bits.ljust((len(bits) + 3) // 4 * 4, "0")
        return format(int(padded, 2), f"0{len(padded) // 4}x")
    except Exception as e:  # noqa: BLE001 — undecodable image → no hash
        logger.debug(f"Flipped hash failed for {path.name}: {e}")
        return None


def _popcount(value: int) -> int:
    """Number of set bits (Hamming weight)."""
    return value.bit_count()

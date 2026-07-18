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

Scale (ADR-P5-12): above ``VECTORIZE_THRESHOLD`` kept images, the linear
Hamming scan switches to a numpy-vectorized popcount over all kept hashes
at once — same O(n) work per check, but done in C instead of a Python loop
calling ``int.bit_count()`` per comparison. This is a constant-factor
speedup, not a better asymptotic (a BK-tree was rejected for the same
reason a Phase-5 council review raised — no better worst case at n≈30k);
a property test pins that the two paths make IDENTICAL keep/drop decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.dataset.sources_config import DedupSettings
from src.utils.dataset_utils import compute_file_hash
from src.utils.image_utils import compute_perceptual_hash

logger = logging.getLogger(__name__)

_HASH_SIZE = 8  # 64-bit aHash, matching src/utils/image_utils

#: Kept-image count above which check_and_add switches to the vectorized
#: Hamming scan (ADR-P5-12; "activated above 2,000 kept images" per plan).
VECTORIZE_THRESHOLD = 2000


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
    # Parallel to _kept's iteration order — lets the vectorized path map a
    # matched array index back to its Path without rebuilding a list each call.
    _kept_paths: list[Path] = field(default_factory=list, init=False, repr=False)
    # Lazily (re)built numpy view of _kept's aHashes; invalidated on insert.
    _kept_ahash_arr: np.ndarray | None = field(default=None, init=False, repr=False)

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
        if len(self._kept) >= VECTORIZE_THRESHOLD:
            duplicate = self._check_vectorized(a_int, flip_int, threshold)
        else:
            duplicate = self._check_naive(a_int, flip_int, threshold)
        if duplicate is not None:
            return duplicate

        self._kept[path] = (a_int, flip_int)
        self._kept_paths.append(path)
        self._exact[a_int] = path
        self._kept_ahash_arr = None  # stale — rebuilt lazily on next vectorized check
        return None

    def _check_naive(self, a_int: int, flip_int: int | None, threshold: int) -> Path | None:
        """Linear Python scan — used below :data:`VECTORIZE_THRESHOLD`."""
        for kept_path, (kept_a, _kept_flip) in self._kept.items():
            if _popcount(a_int ^ kept_a) < threshold:
                return kept_path
            if flip_int is not None and _popcount(flip_int ^ kept_a) < threshold:
                return kept_path
        return None

    def _check_vectorized(self, a_int: int, flip_int: int | None, threshold: int) -> Path | None:
        """Numpy-vectorized scan over all kept aHashes at once.

        Must make the IDENTICAL decision as :meth:`_check_naive` for every
        (a_int, flip_int, threshold, kept set) — pinned by
        ``tests/unit/test_dedup_vectorized.py``.
        """
        if self._kept_ahash_arr is None:
            self._kept_ahash_arr = np.array(
                [kept_a for kept_a, _ in self._kept.values()], dtype=np.uint64
            )
        arr = self._kept_ahash_arr

        distances = _vectorized_popcount(arr ^ np.uint64(a_int))
        hit = _first_below(distances, threshold)
        if hit is not None:
            return self._kept_paths[hit]

        if flip_int is not None:
            flip_distances = _vectorized_popcount(arr ^ np.uint64(flip_int))
            hit = _first_below(flip_distances, threshold)
            if hit is not None:
                return self._kept_paths[hit]
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


def _vectorized_popcount(x: np.ndarray) -> np.ndarray:
    """Elementwise Hamming weight of a uint64 array (SWAR algorithm).

    Works on numpy<2.0 (no dependency on the 2.0+ ``np.bitwise_count``
    ufunc — this project pins ``numpy<2.0``). Unsigned wraparound is
    intentional and exact for this bit-trick.
    """
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def _first_below(distances: np.ndarray, threshold: int) -> int | None:
    """Index of the first element ``< threshold``, or ``None``."""
    hits = np.flatnonzero(distances < threshold)
    return int(hits[0]) if hits.size else None

"""
src.utils.image_utils — Image Integrity Validation and Hashing
==============================================================

Utilities for validating image file integrity and computing perceptual hashes
for duplicate detection. Used by the QA pipeline (check_annotations.py).

Validation strategy:
    1. Try PIL (Pillow) header-only read — fast, catches truncated/corrupt JPEG
    2. Fall back to OpenCV for formats PIL does not handle well

Perceptual hashing uses the average hash (aHash) algorithm:
    - Resize to 8×8 grayscale → compute mean → binary threshold → 64-bit hash
    - Identical or near-identical images produce the same hash
    - Exact duplicate detection: Hamming distance = 0
    - Similar image detection: Hamming distance ≤ 10

Usage:
    from src.utils.image_utils import validate_image, compute_perceptual_hash

    ok, msg = validate_image(Path("image.jpg"))
    phash = compute_perceptual_hash(Path("image.jpg"))  # None if PIL unavailable
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum valid image dimensions (below this is likely metadata corruption)
MIN_IMAGE_DIM: int = 32


# ─── Image Validation ─────────────────────────────────────────────────────────


def validate_image(path: Path) -> tuple[bool, str]:
    """Validate that an image file is readable and not corrupted.

    Performs a lightweight header read using PIL when available, with an
    OpenCV fallback. Does not decode the full image to keep QA fast.

    Args:
        path: Path to the image file.

    Returns:
        Tuple (is_valid: bool, message: str). Message is empty on success,
        or a human-readable error description on failure.
    """
    if not path.exists():
        return False, f"File does not exist: {path}"

    if path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {path.name}"

    # Try PIL first (fast header read)
    try:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as img:
                img.verify()  # Validates headers without full decode
            # Re-open after verify (PIL closes after verify)
            with Image.open(path) as img:
                w, h = img.size
                if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                    return False, f"Image too small ({w}×{h}) — minimum {MIN_IMAGE_DIM}px"
            return True, ""
        except UnidentifiedImageError:
            return False, f"PIL cannot identify image format: {path.name}"
        except Exception as e:
            # PIL failed — try OpenCV fallback before declaring corrupt
            logger.debug(f"PIL validation failed for {path.name}, trying OpenCV: {e}")

    except ImportError:
        logger.debug("PIL not available — falling back to OpenCV for image validation")

    # OpenCV fallback
    return _validate_with_opencv(path)


def _validate_with_opencv(path: Path) -> tuple[bool, str]:
    """Validate image using OpenCV imread.

    Args:
        path: Path to the image file.

    Returns:
        Tuple (is_valid, message).
    """
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return False, f"OpenCV could not read image: {path.name}"

        h, w = img.shape[:2]
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return False, f"Image too small ({w}×{h}) — minimum {MIN_IMAGE_DIM}px"

        return True, ""
    except ImportError:
        # Neither PIL nor OpenCV available — trust file existence
        logger.warning(
            "Neither PIL nor OpenCV is available. "
            "Image validation will only check file existence and size."
        )
        return True, ""
    except Exception as e:
        return False, f"OpenCV error reading {path.name}: {e}"


def get_image_dimensions(path: Path) -> tuple[int, int] | None:
    """Get image width and height without full decoding.

    Args:
        path: Path to the image file.

    Returns:
        (width, height) tuple, or None if the image cannot be read.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size  # (width, height)
            return width, height
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PIL could not get dimensions for {path.name}: {e}")

    try:
        import cv2

        arr = cv2.imread(str(path))
        if arr is not None:
            h, w = arr.shape[:2]
            return w, h
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"OpenCV could not get dimensions for {path.name}: {e}")

    return None


# ─── Perceptual Hashing ───────────────────────────────────────────────────────


def compute_perceptual_hash(path: Path, hash_size: int = 8) -> str | None:
    """Compute an average perceptual hash (aHash) for an image.

    Perceptual hashes are similar for visually similar images, unlike
    SHA-256 which differs for any byte-level change.

    Algorithm:
        1. Open image as grayscale
        2. Resize to hash_size × hash_size (default 8×8)
        3. Compute pixel mean
        4. Threshold: 1 if pixel ≥ mean, else 0
        5. Pack bits into a hex string

    Args:
        path:      Path to the image file.
        hash_size: Hash grid size (hash_size² bits in result). Default 8.

    Returns:
        Hex string of length hash_size² / 4, or None if PIL is not available
        or the image cannot be opened.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.debug("PIL not available — perceptual hashing disabled")
        return None

    try:
        with Image.open(path) as img:
            # Convert to grayscale and resize to hash_size × hash_size
            small = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = list(small.tobytes())  # mode "L" → one byte per pixel

        mean = sum(pixels) / len(pixels)
        # Use strict greater-than: pixels equal to mean are treated as "low"
        # This correctly handles edge cases like all-zero (black) images where
        # mean=0 and >= would incorrectly mark every pixel as high.
        bits = "".join("1" if p > mean else "0" for p in pixels)

        # Pack bits into a hex string (pad to multiple of 4)
        padded = bits.ljust((len(bits) + 3) // 4 * 4, "0")
        hex_hash = format(int(padded, 2), f"0{len(padded) // 4}x")
        return hex_hash

    except Exception as e:
        logger.debug(f"Perceptual hash failed for {path.name}: {e}")
        return None


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute the Hamming distance between two hex perceptual hash strings.

    Args:
        hash_a: First perceptual hash hex string.
        hash_b: Second perceptual hash hex string.

    Returns:
        Number of differing bits. 0 = identical, higher = more different.
        Returns max int if hashes have different lengths.
    """
    if len(hash_a) != len(hash_b):
        return 2**32  # Incomparable

    a_int = int(hash_a, 16)
    b_int = int(hash_b, 16)
    xor = a_int ^ b_int

    # Count set bits (popcount)
    return bin(xor).count("1")


def build_perceptual_hash_index(
    files: list[Path],
    hash_size: int = 8,
) -> dict[str, list[Path]]:
    """Build a perceptual hash → [file paths] index.

    Falls back to SHA-256 hashing if PIL is not available.

    Args:
        files:     List of image file paths.
        hash_size: Hash grid size for perceptual hashing.

    Returns:
        Dict mapping perceptual hash to list of files with that hash.
        Files that could not be hashed are excluded.
    """
    from src.utils.dataset_utils import compute_file_hash

    index: dict[str, list[Path]] = {}
    pil_available = _check_pil_available()

    for path in files:
        if pil_available:
            h = compute_perceptual_hash(path, hash_size)
        else:
            # SHA-256 fallback: only catches byte-exact duplicates
            try:
                h = compute_file_hash(path)
            except OSError:
                h = None

        if h is not None:
            index.setdefault(h, []).append(path)

    return index


def _check_pil_available() -> bool:
    """Return True if PIL (Pillow) is importable."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False

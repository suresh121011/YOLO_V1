"""
src.dataset.capture.exif — Image Metadata Inspection & Stripping
================================================================

Privacy tooling for custom capture ingest (Phase-3). Phone photos embed
EXIF metadata — GPS coordinates, device serials, timestamps — that must
never reach the repo or the DVC remote (docs/04 §4, DPDP). Every image is
stripped at ingest and verified clean afterwards.

Implementation notes:
    - Pure PIL, no new dependencies. PIL is REQUIRED here (unlike the
      QA validators, privacy stripping has no OpenCV fallback) — a
      missing PIL raises rather than silently passing PII through.
    - JPEG: re-saved with ``quality="keep"`` (preserves the original
      quantization tables — near-lossless). The re-encode is not
      bit-identical; ingest computes provenance hashes AFTER stripping.
    - EXIF Orientation is baked into the pixels before stripping so
      photos do not lose their rotation.
    - PNG: re-saved without text chunks / eXIf.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: EXIF IFD pointer tag for the GPS sub-directory.
GPS_IFD_TAG = 0x8825
#: EXIF Orientation tag.
ORIENTATION_TAG = 0x0112


def _require_pil() -> Any:
    """Import and return the PIL.Image module, or raise RuntimeError."""
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Pillow is required for EXIF privacy stripping (pip install Pillow) — "
            "refusing to ingest un-stripped images"
        ) from e
    return Image


def inspect_metadata(path: Path) -> dict[str, Any]:
    """Report what embedded metadata an image carries.

    Args:
        path: Image file path.

    Returns:
        Dict with keys:
            has_exif   — any EXIF tags present
            has_gps    — GPS IFD present/non-empty (location PII)
            has_text   — PNG text chunks present
            tag_count  — number of top-level EXIF tags
            clean      — True when none of the above metadata exists

    Raises:
        RuntimeError: If Pillow is not installed.
    """
    image_mod = _require_pil()

    with image_mod.open(path) as img:
        exif = img.getexif()
        gps_ifd = exif.get_ifd(GPS_IFD_TAG)
        has_gps = bool(gps_ifd) or GPS_IFD_TAG in exif
        text_chunks = getattr(img, "text", {}) or {}
        report = {
            "has_exif": len(exif) > 0,
            "has_gps": has_gps,
            "has_text": len(text_chunks) > 0,
            "tag_count": len(exif),
        }
    report["clean"] = not (report["has_exif"] or report["has_gps"] or report["has_text"])
    return report


def strip_metadata(src: Path, dst: Path) -> None:
    """Write a metadata-free copy of ``src`` to ``dst``.

    EXIF (including GPS), PNG text chunks and other ancillary metadata are
    dropped by re-encoding without them. The EXIF Orientation flag, if set,
    is applied to the pixel data first so the visual rotation survives.
    The ICC color profile is preserved (not PII).

    Args:
        src: Source image path.
        dst: Destination path (parent directories are created).

    Raises:
        RuntimeError: If Pillow is not installed.
        OSError:      If the image cannot be decoded or written.
    """
    image_mod = _require_pil()
    from PIL import ImageOps

    dst.parent.mkdir(parents=True, exist_ok=True)

    with image_mod.open(src) as img:
        img_format = img.format
        icc_profile = img.info.get("icc_profile")
        orientation = img.getexif().get(ORIENTATION_TAG, 1)

        if orientation != 1:
            # Bake rotation into pixels; the new image loses the JPEG
            # quantization tables, so fall back to a fixed high quality.
            transposed = ImageOps.exif_transpose(img)
            _save_clean(transposed, dst, img_format, icc_profile, quality=95)
            logger.debug(f"{src.name}: baked EXIF orientation {orientation} while stripping")
        else:
            _save_clean(img, dst, img_format, icc_profile, quality="keep")


def _save_clean(
    img: Any, dst: Path, img_format: str | None, icc_profile: Any, quality: Any
) -> None:
    """Re-save an image without metadata (no ``exif=``/``pnginfo=`` kwargs)."""
    save_kwargs: dict[str, Any] = {}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if img_format == "JPEG":
        # quality="keep" reuses the source quantization tables (JPEG-only).
        img.save(dst, format="JPEG", quality=quality, **save_kwargs)
    elif img_format == "PNG":
        img.save(dst, format="PNG", **save_kwargs)
    else:
        img.save(dst, format=img_format, **save_kwargs)

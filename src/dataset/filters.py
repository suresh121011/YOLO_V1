"""
src.dataset.filters — Image Quality & Indoor Heuristics
=======================================================

Heuristic filters applied at merge time (docs/03 dataset_templates.md
§3.3). Thresholds come from ``configs/dataset_sources.yaml``
(``indoor_filter:`` section):

    - min dimension:  images below ``min_image_dim`` px are rejected.
    - portrait keep:  aspect w/h < ``portrait_aspect_max`` → likely indoor.
    - brightness:     mean grayscale > ``brightness_outdoor_threshold`` →
                      likely outdoor (bright sky) → rejected.

These are deliberately cheap heuristics, not classifiers; QA reports what
was dropped so a human can audit the filter.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.dataset.sources_config import IndoorFilterSettings
from src.utils.image_utils import get_image_dimensions

logger = logging.getLogger(__name__)

# Downscale size used when computing mean brightness (speed over precision).
_BRIGHTNESS_SAMPLE_SIZE = 64


def compute_mean_brightness(path: Path) -> float | None:
    """Mean grayscale brightness (0–255) of a downscaled copy of the image.

    Returns:
        Mean brightness, or None if the image cannot be read (caller should
        treat unreadable images as failures elsewhere, not here).
    """
    try:
        from PIL import Image
    except ImportError:
        logger.debug("PIL not available — brightness filter disabled")
        return None

    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((_BRIGHTNESS_SAMPLE_SIZE, _BRIGHTNESS_SAMPLE_SIZE))
            pixels = list(small.tobytes())  # mode "L" → one byte per pixel
        return float(sum(pixels)) / len(pixels)
    except Exception as e:  # noqa: BLE001 — any decode failure means "unknown"
        logger.debug(f"Brightness computation failed for {path.name}: {e}")
        return None


def check_image_filter(
    path: Path,
    settings: IndoorFilterSettings,
) -> tuple[bool, str]:
    """Decide whether an image passes the quality/indoor filter.

    Args:
        path:     Image file path.
        settings: Thresholds from configs/dataset_sources.yaml.

    Returns:
        (keep, reason) — reason is empty when kept, otherwise names the
        rejecting rule ("too_small", "likely_outdoor").
    """
    if not settings.enabled:
        return True, ""

    dims = get_image_dimensions(path)
    if dims is not None:
        width, height = dims
        if min(width, height) < settings.min_image_dim:
            return False, "too_small"

        # Portrait images are overwhelmingly indoor captures — keep without
        # the brightness check (docs heuristic).
        if height > 0 and (width / height) < settings.portrait_aspect_max:
            return True, ""

    brightness = compute_mean_brightness(path)
    if brightness is not None and brightness > settings.brightness_outdoor_threshold:
        return False, "likely_outdoor"

    return True, ""

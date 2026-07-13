"""
scripts.dataset.01_download_coco_subset — COCO 2017 Acquisition CLI
===================================================================

Thin wrapper around :class:`src.dataset.downloaders.coco.CocoDownloader`.
All behavior (split, caps, smoke limit, license) comes from
``configs/dataset_sources.yaml``.

Usage:
    python scripts/dataset/01_download_coco_subset.py
    python scripts/dataset/01_download_coco_subset.py --limit 60

DVC integration:
    Invoked by the ``download_coco`` stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.downloaders import DownloadSkippedError
from src.dataset.downloaders.coco import CocoDownloader
from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download the COCO 2017 subset for the 23-class taxonomy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG_PATH,
        help="Path to dataset_sources.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override the per-source image cap (default: mode-based).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success/skip, 1 on error."""
    args = parse_args()

    try:
        config = load_sources_config(args.sources_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    if not config.is_source_allowed("coco"):
        logger.warning("COCO source disabled or gated — nothing to do")
        return 0

    try:
        manifest = CocoDownloader(config.sources["coco"], config).download(args.limit)
    except DownloadSkippedError as e:
        logger.warning(f"COCO acquisition skipped: {e}")
        return 0
    except (RuntimeError, OSError, KeyError) as e:
        logger.error(f"COCO acquisition failed: {e}")
        return 1

    logger.info(f"✅ COCO subset ready: {manifest.image_count} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())

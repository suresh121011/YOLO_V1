"""
src.dataset.downloaders.runner — Shared Acquisition CLI Skeleton
================================================================

All ``scripts/dataset/0N_download_*.py`` CLIs are identical except for the
source name and downloader class; this module holds that shared skeleton so
the numbered scripts stay one-call thin.

Exit-code contract: 0 = success or graceful skip (disabled source, license
gate, missing credentials), 1 = real failure.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from src.dataset.downloaders.base import BaseDownloader, DownloadSkippedError
from src.dataset.sources_config import (
    DEFAULT_SOURCES_CONFIG_PATH,
    SourceConfig,
    SourcesConfig,
    load_sources_config,
)

logger = logging.getLogger(__name__)

DownloaderFactory = Callable[[SourceConfig, SourcesConfig], BaseDownloader]


def run_acquisition_cli(
    source_name: str,
    downloader_factory: DownloaderFactory,
    description: str,
) -> int:
    """Parse standard CLI args and run one source's acquisition.

    Args:
        source_name:        Key in configs/dataset_sources.yaml.
        downloader_factory: Downloader constructor for this source.
        description:        argparse description line.

    Returns:
        Process exit code (0 success/skip, 1 failure).
    """
    parser = argparse.ArgumentParser(
        description=description,
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
    args = parser.parse_args()

    try:
        config = load_sources_config(args.sources_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    if not config.is_source_allowed(source_name):
        logger.warning(f"Source '{source_name}' disabled or license-gated — nothing to do")
        return 0

    try:
        manifest = downloader_factory(config.sources[source_name], config).download(args.limit)
    except DownloadSkippedError as e:
        logger.warning(f"[{source_name}] acquisition skipped: {e}")
        return 0
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        logger.error(f"[{source_name}] acquisition failed: {e}")
        return 1

    logger.info(f"[{source_name}] ready: {manifest.image_count} images")
    return 0

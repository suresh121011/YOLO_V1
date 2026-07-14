"""
scripts.dataset.02_download_openimages_subset — Open Images CLI
===============================================================

Thin wrapper around OpenImagesDownloader (Door / Cupboard / Gas stove).
Behavior comes from configs/dataset_sources.yaml.

Usage:
    python scripts/dataset/02_download_openimages_subset.py [--limit N]

DVC integration:
    Invoked by the ``download_openimages`` stage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.downloaders.openimages import OpenImagesDownloader
from src.dataset.downloaders.runner import run_acquisition_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if __name__ == "__main__":
    sys.exit(
        run_acquisition_cli(
            "openimages",
            OpenImagesDownloader,
            "Download the Open Images V7 subset (Door/Cupboard/Gas stove).",
        )
    )

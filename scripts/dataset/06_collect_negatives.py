"""
scripts.dataset.06_collect_negatives — Negative Image CLI
=========================================================

Thin wrapper around NegativesDownloader: indoor COCO images containing
none of the taxonomy-adjacent classes, stored with empty labels. Run the
COCO stage first so its annotation cache is reused.

Usage:
    python scripts/dataset/06_collect_negatives.py [--limit N]

DVC integration:
    Invoked by the ``collect_negatives`` stage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.downloaders.negatives_dl import NegativesDownloader
from src.dataset.downloaders.runner import run_acquisition_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if __name__ == "__main__":
    sys.exit(
        run_acquisition_cli(
            "negatives",
            NegativesDownloader,
            "Collect background (negative) images with empty labels from COCO.",
        )
    )

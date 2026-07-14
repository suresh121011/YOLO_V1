"""
scripts.dataset.04_download_wider_face — WIDER FACE CLI
=======================================================

Thin wrapper around WiderFaceDownloader (face class). LICENSE-GATED:
research-only/non-commercial — runs only while ``allow_noncommercial:
true`` in configs/dataset_sources.yaml (see docs/04_dataset_engineering).

Usage:
    python scripts/dataset/04_download_wider_face.py [--limit N]

DVC integration:
    Invoked by the ``download_wider_face`` stage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.downloaders.runner import run_acquisition_cli
from src.dataset.downloaders.wider_face import WiderFaceDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if __name__ == "__main__":
    sys.exit(
        run_acquisition_cli(
            "wider_face",
            WiderFaceDownloader,
            "Download the WIDER FACE subset for the face class (license-gated).",
        )
    )

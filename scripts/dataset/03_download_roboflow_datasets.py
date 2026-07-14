"""
scripts.dataset.03_download_roboflow_datasets — Roboflow Universe CLI
=====================================================================

Thin wrapper around RoboflowDownloader (medicine_bottle/charger/wire/
gas_cylinder). Skips gracefully when no datasets are configured or
ROBOFLOW_API_KEY is unset — the QA stage reports the class shortfall.

Usage:
    ROBOFLOW_API_KEY=... python scripts/dataset/03_download_roboflow_datasets.py

DVC integration:
    Invoked by the ``download_roboflow`` stage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.downloaders.roboflow_dl import RoboflowDownloader
from src.dataset.downloaders.runner import run_acquisition_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if __name__ == "__main__":
    sys.exit(
        run_acquisition_cli(
            "roboflow",
            RoboflowDownloader,
            "Download configured Roboflow Universe datasets (YOLO export).",
        )
    )

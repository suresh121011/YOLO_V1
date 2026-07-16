"""
scripts.training.preflight_check — Standalone Mitigation Preflight CLI
======================================================================

On-demand run of the Phase-4 preflight gates (G1–G8) without starting
training. Useful in CI, before long runs, and while debugging completeness
artifacts. train_yolo.py runs the same gates automatically whenever
missing-annotation mitigation is enabled.

Usage:
    python scripts/training/preflight_check.py
    python scripts/training/preflight_check.py --config configs/training/yolo11n_config.yaml
    python scripts/training/preflight_check.py --json-out reports/preflight.json

Exit codes (mirrors scripts/qa/run_full_qa.py):
    0 — all gates pass
    1 — at least one gate FAILED
    2 — warnings only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.mitigation_config import MITIGATION_SECTION, MitigationConfig
from src.training.preflight import run_preflight
from src.utils.config_helpers import load_training_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    """Run the preflight gates and report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code 0 (pass) / 1 (fail) / 2 (warnings only).
    """
    try:
        train_cfg = load_training_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Training config not found: {e}")
        return 1

    try:
        mitigation = MitigationConfig.from_training_config(train_cfg)
    except ValueError as e:
        logger.error(f"Invalid {MITIGATION_SECTION} section: {e}")
        return 1

    if not mitigation.enabled:
        logger.info(
            f"Note: {MITIGATION_SECTION}.enabled is false in {args.config} — training "
            f"would not run these gates. Checking the artifact anyway."
        )

    report = run_preflight(
        mitigation,
        data_yaml_path=args.data,
        train_cfg=train_cfg,
        processed_root=args.processed_root,
    )

    for line in report.format_lines():
        logger.info(line)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info(f"Preflight report written: {args.json_out}")

    if report.verdict == "FAIL":
        return 1
    if report.verdict == "WARN":
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the missing-annotation-mitigation preflight gates (G1–G8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/yolo11n_config.yaml"),
        help="Training config YAML (its missing_annotation_mitigation section is used).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Dataset taxonomy YAML.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Dataset root containing images/{train,val}.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable gate report.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

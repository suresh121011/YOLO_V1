"""
scripts.dataset.10_capture_progress — Collection Progress CLI
===============================================================

Reports custom-capture collection progress against the Phase-3
governance targets via :mod:`src.dataset.capture.progress`: per-class
instance counts, house/room/lighting coverage, annotation status,
consent anomalies (withdrawn references) and eval-set status.

Usage:
    python scripts/dataset/10_capture_progress.py
    python scripts/dataset/10_capture_progress.py --fail-under-targets

Exit codes: 0 = report written (targets met, or --fail-under-targets not
set), 1 = report written but targets not yet met and --fail-under-targets
was passed, 1 = load/config failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.capture.config import load_capture_config
from src.dataset.capture.consent import load_consent_registry
from src.dataset.capture.progress import build_progress_report, write_progress_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Report custom-capture collection progress vs Phase-3 targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to capture_config.yaml (default: configs/capture_config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qa_reports"),
        help="Report output directory.",
    )
    parser.add_argument(
        "--fail-under-targets",
        action="store_true",
        help="Exit 1 when collection targets are not yet fully met.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on failure or unmet targets."""
    args = parse_args()

    try:
        config = load_capture_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    registry = load_consent_registry(config.consent.registry_path)
    report = build_progress_report(config.captures_root, config.eval_root, config.targets, registry)
    paths = write_progress_report(report, args.output)

    logger.info(
        f"Progress: {report.total_images}/{report.total_target} images, "
        f"{len(report.houses)}/{report.houses_target} houses, "
        f"{len(report.classes_met)}/{len(report.class_counts)} classes at target "
        f"→ {paths['markdown']}"
    )
    if report.classes_pending:
        logger.info(f"Classes still short of target: {report.classes_pending}")
    if report.withdrawn_sessions:
        logger.warning(f"Withdrawn-consent sessions need removal: {report.withdrawn_sessions}")

    if args.fail_under_targets and not report.targets_met:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

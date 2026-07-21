"""
scripts.dataset.17_dataset_quality_report — L5 Dataset Quality Report
=======================================================================

Phase-5 L5 entry point: aggregates completeness, coverage (L4), the
verification ledger, and the merged manifest into one machine-readable
dataset quality report — read by the v0.7+/v1.0 release gate (RG3) to check
per-class coverage thresholds. Writes
``data/qa_reports/dataset_quality_report.json`` (a DVC metric).

Usage:
    python scripts/dataset/17_dataset_quality_report.py

DVC integration: the dataset_quality_report stage, run after coverage_report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.quality import build_quality_report, validate_quality_report
from src.utils.report_utils import save_json_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    """Build, validate, and save the dataset quality report."""
    try:
        report = build_quality_report(
            completeness_path=args.completeness,
            coverage_report_path=args.coverage_report,
            merged_manifest_path=args.merged_manifest,
            ledger_path=args.ledger_path,
            batches_root=args.batches_root,
            data_yaml_path=args.data,
            capture_manifests_dir=args.capture_manifests,
        )
    except (AnnotationError, FileNotFoundError, ValueError) as e:
        logger.error(f"Dataset quality report generation failed: {e}")
        return 1

    errors = validate_quality_report(report)
    if errors:
        for err in errors:
            logger.error(f"Quality report validation: {err}")
        return 1

    save_json_report(report, args.output)
    logger.info(f"Dataset quality report written: {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the L5 dataset quality report (Phase-5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--completeness", type=Path, default=Path("data/processed/completeness.json")
    )
    parser.add_argument(
        "--coverage-report", type=Path, default=Path("data/qa_reports/coverage_report.json")
    )
    parser.add_argument(
        "--merged-manifest", type=Path, default=Path("data/merged/merged_manifest.json")
    )
    parser.add_argument(
        "--ledger-path", type=Path, default=Path("data/annotation/verification_ledger.json")
    )
    parser.add_argument("--batches-root", type=Path, default=Path("data/annotation/batches"))
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--capture-manifests",
        type=Path,
        default=Path("data/raw/custom_captures/manifests"),
        help="Per-session capture manifest directory (house/image scale counts).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/qa_reports/dataset_quality_report.json")
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

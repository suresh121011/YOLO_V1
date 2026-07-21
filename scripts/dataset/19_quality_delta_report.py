"""
scripts.dataset.19_quality_delta_report — M8 Quality-Report Delta CLI
========================================================================

Compares two ``dataset_quality_report.json`` snapshots (e.g. a released
build's copy vs. the current in-progress build) and writes a per-class
residual-risk / coverage-score delta table — the artifact behind M8's plan
acceptance criterion "coverage report shows per-priority-class residual-
risk drop vs v0.5 (delta table in changelog)". Pure aggregation over two
already-built reports; recomputes nothing.

Usage:
    python scripts/dataset/19_quality_delta_report.py \
        --baseline data/releases/dataset-v0.5.0/dataset_quality_report.json \
        --current data/qa_reports/dataset_quality_report.json

Not a DVC stage — a manual release-prep tool run once per version bump, not
part of the automated `dvc repro` chain (there is no single "current"
report independent of when it's invoked).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.quality_delta import build_quality_delta
from src.utils.config_helpers import load_yaml
from src.utils.report_utils import write_all_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_report_sections(delta: dict) -> list[dict]:
    """Assemble the priority-class delta table for the Markdown report."""
    return [
        {
            "heading": "Priority-class residual-risk delta",
            "content": (
                f"Baseline generated: {delta['baseline_generated_at']}. "
                f"Current generated: {delta['current_generated_at']}. "
                "Negative delta-residual and positive delta-coverage both mean "
                "improvement."
            ),
            "table": {
                "headers": [
                    "Class",
                    "Baseline residual",
                    "Current residual",
                    "Delta residual",
                    "Baseline coverage",
                    "Current coverage",
                    "Delta coverage",
                ],
                "rows": [
                    [
                        row["class"],
                        row["baseline_residual_missing_estimate"],
                        row["current_residual_missing_estimate"],
                        row["residual_missing_delta"],
                        row["baseline_coverage_score"],
                        row["current_coverage_score"],
                        row["coverage_score_delta"],
                    ]
                    for row in delta["priority_class_delta"]
                ],
            },
        }
    ]


def run(args: argparse.Namespace) -> int:
    """Build, save, and print the quality delta report."""
    if not args.baseline.exists():
        logger.error(f"Baseline quality report not found: {args.baseline}")
        return 1
    if not args.current.exists():
        logger.error(f"Current quality report not found: {args.current}")
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))

    priority_classes = frozenset(
        str(c)
        for c in load_yaml(args.annotation_config)
        .get("auto_annotation", {})
        .get("targeting", {})
        .get("priority_classes", [])
    )
    delta = build_quality_delta(baseline, current, priority_classes)

    paths = write_all_formats(
        report_data=delta,
        csv_rows=delta["per_class_delta"],
        md_title="Dataset Quality Delta (M8)",
        md_sections=build_report_sections(delta),
        output_dir=args.output_dir,
        base_name=args.base_name,
        md_metadata={"Baseline": args.baseline.as_posix(), "Current": args.current.as_posix()},
    )
    logger.info(f"Quality delta written: {paths['markdown']}")
    print(paths["markdown"].read_text(encoding="utf-8"))  # pasteable into DATASET_CHANGELOG.md
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two dataset_quality_report.json snapshots (Phase-5 M8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Earlier dataset_quality_report.json."
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/qa_reports/dataset_quality_report.json"),
        help="Later dataset_quality_report.json.",
    )
    parser.add_argument("--annotation-config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa_reports"))
    parser.add_argument("--base-name", type=str, default="quality_delta")
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

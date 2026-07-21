"""
scripts.dataset.16_coverage_report — L4 Coverage Estimation (DVC: coverage_report)
====================================================================================

Phase-5 L4 entry point: pure-arithmetic residual missing-annotation risk
estimate over the pinned auto-annotation candidates, the verification
ledger, and the trained labels (ADR-P5-06 — zero inference at report time).
Writes ``data/qa_reports/coverage_report.json`` (+csv/md).

Inputs (hard-fails on taxonomy drift against any of them):
    data/annotation/candidates/<backend>/candidates.json (one or more)
    data/annotation/verification_ledger.json
    data/processed/completeness.json
    data/processed/labels/{train,val,test}
    configs/data.yaml
    configs/annotation.yaml (coverage.* params)

Usage:
    python scripts/dataset/16_coverage_report.py

DVC integration: the coverage_report stage, run after generate_completeness.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.coverage import (
    build_coverage_report,
    grounding_dino_decision,
    validate_coverage_report,
)
from src.utils.config_helpers import load_yaml
from src.utils.report_utils import write_all_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_report_sections(report: dict[str, Any], csv_rows: list[dict[str, Any]]) -> list[dict]:
    """Assemble Markdown report sections from a built coverage report."""
    dataset = report["dataset"]
    summary = report["per_image_summary"]
    class_rows = [
        [
            row["class"],
            row["annotated_instances"],
            row["unverified_candidates"],
            row["verified_present"],
            row["verified_absent"],
            row["residual_missing_estimate"],
            row["coverage_score"],
        ]
        for row in csv_rows
    ]
    calibration_rows = [
        [name, c["verified_cells"], c["estimator_precision"], c["estimator_recall_proxy"]]
        for name, c in sorted(report["calibration"].items())
    ]
    sections: list[dict] = [
        {
            "heading": "Dataset residual risk",
            "content": (
                f"Estimated residual missing instances: **{dataset['residual_missing_total']}** "
                f"(from {dataset['unknown_objects_total']} unverified candidate detections). "
                f"Mean per-image completeness: {summary['mean_completeness']}, "
                f"p10: {summary['p10_completeness']}, "
                f"images below 0.5 completeness: {summary['images_below_0_5']}."
            ),
        },
        {
            "heading": "Per-class coverage",
            "content": (
                "`residual_missing_estimate` discounts unverified candidates by the "
                "calibrated (or configured prior) estimator precision for that class."
            ),
            "table": {
                "headers": [
                    "Class",
                    "Annotated",
                    "Unverified candidates",
                    "Verified present",
                    "Verified absent",
                    "Residual estimate",
                    "Coverage score",
                ],
                "rows": class_rows,
            },
        },
    ]
    if calibration_rows:
        sections.append(
            {
                "heading": "Estimator calibration",
                "content": "Derived from the ledger's own verified cells (free calibration data).",
                "table": {
                    "headers": ["Class", "Verified cells", "Precision", "Recall proxy"],
                    "rows": calibration_rows,
                },
            }
        )
    if "grounding_dino_recommendation" in report:
        rec = report["grounding_dino_recommendation"]
        sections.append(
            {
                "heading": "grounding_dino enablement decision (M8, ADR-P5-02)",
                "content": (f"**Recommend enable: {rec['recommend_enable']}** — {rec['reason']}"),
                "table": {
                    "headers": ["Priority class", "Calibrated precision"],
                    "rows": [
                        [name, precision if precision is not None else "uncalibrated"]
                        for name, precision in sorted(rec["priority_class_precisions"].items())
                    ],
                },
            }
        )
    return sections


def run(args: argparse.Namespace) -> int:
    """Build, validate, save, and report the coverage artifact."""
    annotation_cfg = load_yaml(args.annotation_config)
    coverage_cfg = annotation_cfg.get("coverage", {})
    iou_threshold = float(coverage_cfg.get("iou_match_threshold", 0.5))
    estimation_conf = {
        str(k): float(v) for k, v in (coverage_cfg.get("estimation_conf") or {}).items()
    }
    estimation_conf.setdefault("default", 0.35)

    try:
        report = build_coverage_report(
            candidates_root=args.candidates_root,
            ledger_path=args.ledger_path,
            completeness_path=args.completeness,
            processed_labels_root=args.labels_root,
            data_yaml_path=args.data,
            iou_match_threshold=iou_threshold,
            estimation_conf=estimation_conf,
        )
    except (AnnotationError, FileNotFoundError, ValueError) as e:
        logger.error(f"Coverage report generation failed: {e}")
        return 1

    auto_annotation_cfg = annotation_cfg.get("auto_annotation", {})
    priority_classes = frozenset(
        str(c) for c in auto_annotation_cfg.get("targeting", {}).get("priority_classes", [])
    )
    precision_threshold = float(
        auto_annotation_cfg.get("backends", {})
        .get("grounding_dino", {})
        .get("enable_below_precision", 0.4)
    )
    report["grounding_dino_recommendation"] = grounding_dino_decision(
        report["calibration"], priority_classes, precision_threshold
    )

    errors = validate_coverage_report(report)
    if errors:
        for err in errors:
            logger.error(f"Coverage report validation: {err}")
        return 1

    csv_rows = [{"class": name, **entry} for name, entry in sorted(report["per_class"].items())]
    paths = write_all_formats(
        report_data=report,
        csv_rows=csv_rows,
        md_title="Coverage Report (L4)",
        md_sections=build_report_sections(report, csv_rows),
        output_dir=args.output_dir,
        base_name="coverage_report",
        md_metadata={
            "Taxonomy fingerprint": report["taxonomy_fingerprint"][:23] + "…",
            "Classes": len(report["per_class"]),
            "Images": len(report["per_image"]),
        },
    )
    logger.info(f"Coverage report written: {paths['markdown']}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the L4 coverage estimation report (Phase-5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates-root", type=Path, default=Path("data/annotation/candidates"))
    parser.add_argument(
        "--ledger-path", type=Path, default=Path("data/annotation/verification_ledger.json")
    )
    parser.add_argument(
        "--completeness", type=Path, default=Path("data/processed/completeness.json")
    )
    parser.add_argument("--labels-root", type=Path, default=Path("data/processed/labels"))
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--annotation-config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa_reports"))
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

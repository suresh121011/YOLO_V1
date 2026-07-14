"""
scripts.dataset.09_import_annotations — Annotation Import CLI
=============================================================

Imports CVAT "YOLO 1.1" (or any YOLO-format) annotation exports for an
ingested capture session via :mod:`src.dataset.capture.annotations`:
class-order verification against configs/data.yaml (CRITICAL — a CVAT
task with a subset/reordered label list silently shifts every class ID),
session-scoped label validation, per-annotator staging and finalize.

Usage:
    # Stage annotator A's export (validates before staging)
    python scripts/dataset/09_import_annotations.py \\
        --session h01_kitchen_s001 --stage --export exports/asha.zip \\
        --annotator asha

    # Promote one annotator's staged labels to the session's final labels
    python scripts/dataset/09_import_annotations.py \\
        --session h01_kitchen_s001 --finalize --from asha

Exit codes: 0 = success, 1 = critical (class order, orphan labels,
under-coverage, format errors, missing staged labels), 2 = warnings
(declared class with zero boxes).

DVC integration:
    Runs between ingest (08) and ``dvc commit -f ingest_custom_captures``;
    see docs/04 capture_annotation_runbook.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.capture.annotations import (
    finalize_annotations,
    read_yolo_export,
    stage_annotations,
    staged_annotators,
    update_annotation_status,
    validate_session_labels,
    verify_class_order,
)
from src.dataset.capture.config import load_capture_config
from src.dataset.manifest import CaptureSessionManifest
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Import, validate, stage and finalize YOLO annotation exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session", required=True, help="Session ID, e.g. h01_kitchen_s001.")
    parser.add_argument(
        "--dataset",
        choices=("captures", "eval"),
        default="captures",
        help="Which capture tree the session lives in.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to capture_config.yaml (default: configs/capture_config.yaml).",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml (taxonomy class order).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage", action="store_true", help="Validate an export and stage it per annotator."
    )
    mode.add_argument(
        "--finalize",
        action="store_true",
        help="Promote one annotator's staged labels to final session labels.",
    )

    parser.add_argument(
        "--export", type=Path, help="Annotation export (.zip or directory) for --stage."
    )
    parser.add_argument("--annotator", help="Annotator handle (pseudonymous) for --stage.")
    parser.add_argument(
        "--from",
        dest="from_annotator",
        help="Annotator whose staged labels to finalize (--finalize).",
    )
    return parser.parse_args()


def _stage(args: argparse.Namespace) -> int:
    """--stage: read export, verify class order, validate, stage."""
    if args.export is None or not args.annotator:
        logger.error("--stage requires --export and --annotator")
        return 1

    config = load_capture_config(args.config)
    root = config.eval_root if args.dataset == "eval" else config.captures_root

    manifest_path = root / "manifests" / f"{args.session}.json"
    if not manifest_path.exists():
        logger.error(f"Session '{args.session}' not ingested (no manifest at {manifest_path})")
        return 1
    manifest = CaptureSessionManifest.load(manifest_path)

    try:
        class_names = get_class_names_from_data_yaml(load_data_config(args.data_config))
        export = read_yolo_export(args.export)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    order_problems = verify_class_order(export.names, class_names)
    if order_problems:
        for problem in order_problems:
            logger.error(f"CLASS ORDER: {problem}")
        logger.error("Aborting — every class ID in this export is suspect")
        return 1

    session_stems = {Path(name).stem for name in manifest.image_hashes}
    validation = validate_session_labels(
        export,
        session_stems,
        class_names,
        config.annotation.min_labeled_fraction,
        trusted_classes=tuple(manifest.trusted_classes),
    )
    for problem in validation.problems:
        logger.error(problem)
    for warning in validation.warnings:
        logger.warning(warning)
    if validation.problems:
        return 1

    stage_annotations(export, args.session, args.annotator, config.annotation.staging_dir)
    update_annotation_status(root, args.session, "staged", annotator=args.annotator)
    logger.info(
        f"Staged {args.session} by {args.annotator}: "
        f"{validation.labeled_images}/{validation.total_images} images labeled, "
        f"{sum(validation.class_counts.values())} boxes {validation.class_counts}"
    )
    others = [
        a
        for a in staged_annotators(config.annotation.staging_dir, args.session)
        if a != args.annotator
    ]
    if others:
        logger.info(f"Also staged by: {others} — compare before finalizing (see runbook)")
    return 2 if validation.warnings else 0


def _finalize(args: argparse.Namespace) -> int:
    """--finalize: promote staged labels to the session's final labels."""
    if not args.from_annotator:
        logger.error("--finalize requires --from <annotator>")
        return 1

    config = load_capture_config(args.config)
    root = config.eval_root if args.dataset == "eval" else config.captures_root

    try:
        class_names = get_class_names_from_data_yaml(load_data_config(args.data_config))
        result = finalize_annotations(
            config.annotation.staging_dir,
            args.session,
            args.from_annotator,
            root,
            class_names,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    logger.info(
        f"✅ Finalized {result.session_id} ({result.labels_written} label files, "
        f"{sum(result.class_counts.values())} boxes)"
    )
    if args.dataset == "captures":
        logger.info(
            "Reminder: once the first session is finalized, set "
            "sources.custom_captures.enabled: true in configs/dataset_sources.yaml, "
            "then `dvc commit -f ingest_custom_captures` and `dvc repro` (see runbook)."
        )
    return 0


def main() -> int:
    """Entry point. Returns 0 on success, 1 on critical, 2 on warnings."""
    args = parse_args()
    if args.stage:
        return _stage(args)
    return _finalize(args)


if __name__ == "__main__":
    sys.exit(main())

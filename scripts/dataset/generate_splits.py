"""
scripts.dataset.generate_splits — Dataset Split Orchestrator
============================================================

High-level orchestrator that runs the full dataset preparation workflow:
    1. Run split_dataset.py logic (train/val/test split)
    2. Run dataset_stats.py to generate statistics
    3. Validate split integrity (no leakage)
    4. Print a human-readable summary

This is the single entry point for the split_train_val_test DVC stage.

Usage:
    python scripts/dataset/generate_splits.py
    python scripts/dataset/generate_splits.py --config configs/data.yaml --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataset.dataset_stats import build_reports, compute_split_stats
from scripts.dataset.split_dataset import (
    compute_split_assignments,
    copy_split_files,
    generate_split_report,
    verify_no_leakage,
)
from src.dataset.split_config import DEFAULT_SPLIT_CONFIG_PATH, load_split_settings
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import find_image_files, group_files_by_key
from src.utils.report_utils import write_all_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Full dataset split and statistics pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source directory containing flat images/ and labels/ "
        "(default: from split config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for split images/ and labels/ " "(default: from split config).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/qa_reports"),
        help="Root directory for all generated reports.",
    )
    parser.add_argument("--train", type=float, default=None)
    parser.add_argument("--val", type=float, default=None)
    parser.add_argument("--test", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG_PATH,
        help="Path to the dataset split configuration YAML.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip the splitting step (use if splits already exist).",
    )
    return parser.parse_args()


def print_summary(
    split_stats: dict[str, dict],
    leakage: list[str],
    empty_classes: list[str],
) -> None:
    """Print a human-readable pipeline summary to stdout.

    Args:
        split_stats:   Per-split statistics dict.
        leakage:       List of leaking filenames.
        empty_classes: List of class names with zero instances.
    """
    print()
    print("=" * 60)
    print("  Dataset Preparation Summary")
    print("=" * 60)
    for split, data in split_stats.items():
        print(
            f"  {split.upper():8s}  {data['total_images']:5d} images  "
            f"{data['total_boxes']:6d} boxes"
        )
    print()

    if leakage:
        print(f"  🔴 DATA LEAKAGE: {len(leakage)} files found in multiple splits")
    else:
        print("  ✅ Data leakage check: PASS")

    if empty_classes:
        print(f"  ⚠️  Empty classes ({len(empty_classes)}): {', '.join(empty_classes)}")
    else:
        print("  ✅ All classes have annotations")

    print("=" * 60)
    print()


def main() -> int:
    """Main orchestrator entry point. Returns exit code."""
    # Windows cp1252 consoles cannot encode the emoji in print_summary.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()

    # Resolve settings: explicit CLI flags override the split config YAML.
    try:
        settings = load_split_settings(args.split_config).with_overrides(
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            source_dir=args.source,
            output_dir=args.output,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    args.source = settings.source_dir
    args.output = settings.output_dir
    args.train = settings.train_ratio
    args.val = settings.val_ratio
    args.test = settings.test_ratio
    args.seed = settings.seed

    logger.info("=" * 60)
    logger.info("Dataset Pipeline Orchestrator — Elderly Assistant System")
    logger.info("=" * 60)

    # ── Step 1: Load config ──────────────────────────────────────────────────
    try:
        data_cfg = load_data_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data config: {e}")
        return 1

    class_names = get_class_names_from_data_yaml(data_cfg)
    num_classes = data_cfg.get("nc", len(class_names))
    logger.info(f"Config loaded: {num_classes} classes")

    # ── Step 2: Split dataset ────────────────────────────────────────────────
    leakage: list[str] = []

    if not args.skip_split:
        logger.info("Step 1/3: Splitting dataset…")
        images_dir = args.source / "images"
        labels_dir = args.source / "labels"

        if not images_dir.exists():
            logger.error(f"Source images directory not found: {images_dir.absolute()}")
            return 1

        all_images = find_image_files(images_dir)
        if not all_images:
            logger.error("No images found in source directory.")
            return 1

        logger.info(f"Found {len(all_images)} images to split")

        groups = group_files_by_key(all_images)

        try:
            assignments = compute_split_assignments(
                groups,
                train_ratio=args.train,
                val_ratio=args.val,
                test_ratio=args.test,
                seed=args.seed,
            )
        except ValueError as e:
            logger.error(str(e))
            return 1

        stats = copy_split_files(
            groups=groups,
            assignments=assignments,
            images_source_dir=images_dir,
            labels_source_dir=labels_dir,
            output_dir=args.output,
        )

        leakage = verify_no_leakage(args.output)

        # Create a minimal Namespace for the split report
        import argparse as ap

        split_args = ap.Namespace(
            seed=args.seed,
            train=args.train,
            val=args.val,
            test=args.test,
            source=args.source,
            output=args.output,
        )
        generate_split_report(stats, assignments, groups, split_args, args.output, leakage)
    else:
        logger.info("Step 1/3: Split skipped (--skip-split)")

    # ── Step 3: Generate statistics ──────────────────────────────────────────
    logger.info("Step 2/3: Generating statistics…")

    split_stats = compute_split_stats(args.output, class_names, num_classes)

    if split_stats:
        json_report, csv_rows, md_sections = build_reports(
            split_stats, class_names, num_classes, args.output
        )

        write_all_formats(
            report_data=json_report,
            csv_rows=csv_rows,
            md_title="Dataset Statistics — Elderly Assistant System",
            md_sections=md_sections,
            output_dir=args.reports_dir / "dataset_statistics",
            base_name="dataset_statistics",
            csv_fieldnames=[
                "split",
                "class_id",
                "class_name",
                "images_with_class",
                "bounding_boxes",
                "is_safety_critical",
                "is_custom_required",
                "total_split_boxes",
                "pct_of_split",
            ],
        )
        empty_classes = json_report.get("empty_classes", [])
    else:
        logger.warning("No split statistics computed — split directories may be empty")
        empty_classes = []

    # ── Step 4: Final summary ────────────────────────────────────────────────
    logger.info("Step 3/3: Pipeline complete")
    print_summary(split_stats, leakage, empty_classes)

    if leakage:
        logger.error("Aborting: data leakage detected. Check split reports.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

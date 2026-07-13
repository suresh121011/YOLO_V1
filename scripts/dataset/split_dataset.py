"""
scripts.dataset.split_dataset — Dataset Train/Val/Test Splitter
===============================================================

Splits a YOLO-format dataset into train/val/test sets using group-aware
splitting to prevent data leakage. Images from the same video or burst
capture sequence are always assigned to the same split.

Split strategy:
    - 80% train / 10% val / 10% test (configurable)
    - Groups by capture session (filename prefix) before splitting
    - Deterministic: same seed always produces the same split
    - Copies files only (never moves or modifies originals)

Input directory structure (data/processed/):
    images/          ← flat or any structure
    labels/          ← matching .txt files

Output directory structure (data/processed/):
    images/train/    val/    test/
    labels/train/    val/    test/

Configuration:
    Defaults come from configs/dataset_split_config.yaml (ratios, seed,
    strategy, source/output dirs). Explicit CLI flags override the YAML.

Usage:
    python scripts/dataset/split_dataset.py
    python scripts/dataset/split_dataset.py --seed 42 --train 0.8 --val 0.1 --test 0.1
    python scripts/dataset/split_dataset.py --source data/merged --output data/processed

DVC integration:
    This script is invoked by the split_train_val_test DVC stage.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.split_config import DEFAULT_SPLIT_CONFIG_PATH, load_split_settings
from src.dataset.splitting import SplitContext, get_strategy
from src.utils.dataset_utils import (
    IMAGE_EXTENSIONS,
    find_image_files,
    group_files_by_key,
)
from src.utils.report_utils import save_json_report, save_markdown_report, timestamp_str

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SPLIT_NAMES: list[str] = ["train", "val", "test"]


# ─── Core Split Logic ─────────────────────────────────────────────────────────


def compute_split_assignments(
    groups: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Assign capture groups to splits while respecting ratio targets.

    Groups are shuffled with a fixed seed then assigned round-robin to
    bins until ratios are satisfied. This prevents data leakage by keeping
    all frames from the same video/burst in the same split.

    Args:
        groups:      Dict mapping group key → list of image Paths.
        train_ratio: Target fraction for training set (e.g., 0.80).
        val_ratio:   Target fraction for validation set (e.g., 0.10).
        test_ratio:  Target fraction for test set (e.g., 0.10).
        seed:        Random seed for reproducibility.

    Returns:
        Dict mapping split name ("train"/"val"/"test") to list of group keys.
    """
    # Delegates to the default strategy (kept for backward compatibility —
    # generate_splits.py and the unit tests call this directly).
    context = SplitContext(
        groups=groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return get_strategy("group_aware").assign(context)


def copy_split_files(
    groups: dict[str, list[Path]],
    assignments: dict[str, list[str]],
    images_source_dir: Path,
    labels_source_dir: Path,
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Copy image and label files into their assigned split directories.

    Args:
        groups:            Group key → image Paths mapping.
        assignments:       Split name → list of group keys.
        images_source_dir: Root directory of source images.
        labels_source_dir: Root directory of source labels.
        output_dir:        Root output directory (images/ and labels/ subdirs
                           will be created under this path).

    Returns:
        Nested dict: split → {"images": count, "labels": count}.
    """
    stats: dict[str, dict[str, int]] = {s: {"images": 0, "labels": 0} for s in SPLIT_NAMES}

    for split_name, group_keys in assignments.items():
        img_out_dir = output_dir / "images" / split_name
        lbl_out_dir = output_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for key in group_keys:
            for img_path in groups[key]:
                # Copy image
                dst_img = img_out_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                stats[split_name]["images"] += 1

                # Copy label if it exists
                try:
                    rel = img_path.relative_to(images_source_dir)
                except ValueError:
                    rel = Path(img_path.name)

                lbl_path = labels_source_dir / rel.with_suffix(".txt")
                if lbl_path.exists():
                    dst_lbl = lbl_out_dir / lbl_path.name
                    shutil.copy2(lbl_path, dst_lbl)
                    stats[split_name]["labels"] += 1

        logger.info(
            f"Split '{split_name}': "
            f"{stats[split_name]['images']} images, "
            f"{stats[split_name]['labels']} labels"
        )

    return stats


def verify_no_leakage(
    output_dir: Path,
) -> list[str]:
    """Verify that no image filename appears in more than one split.

    Args:
        output_dir: Root output directory (must contain images/train, val, test).

    Returns:
        List of filenames that appear in more than one split (leakage).
        Empty list means no leakage detected.
    """
    split_files: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        split_dir = output_dir / "images" / split
        if split_dir.exists():
            # Only images count — scaffolding files (.gitkeep) are not leakage.
            split_files[split] = {
                p.name
                for p in split_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            }
        else:
            split_files[split] = set()

    leakage: list[str] = []
    for i, s1 in enumerate(SPLIT_NAMES):
        for s2 in SPLIT_NAMES[i + 1 :]:
            overlap = split_files[s1] & split_files[s2]
            if overlap:
                leakage.extend(sorted(overlap))
                logger.error(
                    f"DATA LEAKAGE: {len(overlap)} files appear in both "
                    f"'{s1}' and '{s2}': {sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
                )

    if not leakage:
        logger.info("Leakage verification: PASS — no overlap between splits")

    return leakage


# ─── Report Generation ────────────────────────────────────────────────────────


def generate_split_report(
    stats: dict[str, dict[str, int]],
    assignments: dict[str, list[str]],
    groups: dict[str, list[Path]],
    args: argparse.Namespace,
    output_dir: Path,
    leakage: list[str],
) -> None:
    """Write split summary as JSON and Markdown reports.

    Args:
        stats:      Per-split file counts.
        assignments: Per-split group key assignments.
        groups:     Group key → image Paths mapping.
        args:       Parsed CLI arguments (for metadata).
        output_dir: Report output directory.
        leakage:    List of leaking filenames (should be empty).
    """
    total_images = sum(s["images"] for s in stats.values())
    total_labels = sum(s["labels"] for s in stats.values())

    report = {
        "timestamp": timestamp_str(),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "total_images": total_images,
        "total_labels": total_labels,
        "total_groups": len(groups),
        "leakage_count": len(leakage),
        "leakage_files": leakage[:20],
        "splits": {
            s: {
                "images": stats[s]["images"],
                "labels": stats[s]["labels"],
                "groups": len(assignments[s]),
                "pct_images": round(100.0 * stats[s]["images"] / max(total_images, 1), 1),
            }
            for s in SPLIT_NAMES
        },
    }

    # JSON report
    report_dir = output_dir / "split_report"
    save_json_report(report, report_dir / "split_summary.json")

    # Markdown report
    table_rows = [
        [
            s.upper(),
            str(stats[s]["images"]),
            str(stats[s]["labels"]),
            str(len(assignments[s])),
            f"{report['splits'][s]['pct_images']:.1f}%",
        ]
        for s in SPLIT_NAMES
    ]
    table_rows.append(["**TOTAL**", str(total_images), str(total_labels), str(len(groups)), "100%"])

    leakage_status = "✅ None detected" if not leakage else f"🔴 {len(leakage)} files"

    sections = [
        {
            "heading": "Split Summary",
            "table": {
                "headers": ["Split", "Images", "Labels", "Groups", "% of Total"],
                "rows": table_rows,
            },
        },
        {
            "heading": "Configuration",
            "content": (
                f"- **Seed:** `{args.seed}`\n"
                f"- **Train ratio:** {args.train}\n"
                f"- **Val ratio:** {args.val}\n"
                f"- **Test ratio:** {args.test}\n"
                f"- **Group-aware splitting:** Enabled\n"
                f"- **Data leakage:** {leakage_status}"
            ),
        },
    ]

    save_markdown_report(
        "Dataset Split Report",
        sections,
        report_dir / "split_summary.md",
        metadata={"Source": str(args.source), "Output": str(args.output)},
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Split a YOLO dataset into train/val/test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source directory containing images/ and labels/ subdirectories "
        "(default: from split config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for split images/ and labels/ subdirectories "
        "(default: from split config).",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=None,
        help="Fraction of data for the training set (default: from split config).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=None,
        help="Fraction of data for the validation set (default: from split config).",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=None,
        help="Fraction of data for the test set (default: from split config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits (default: from split config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml for class configuration.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG_PATH,
        help="Path to the dataset split configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute split assignments without copying any files.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point. Returns exit code (0 = success, 1 = error)."""
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
    logger.info("Dataset Splitter — Elderly Assistant System")
    logger.info("=" * 60)
    logger.info(f"Source:  {args.source.absolute()}")
    logger.info(f"Output:  {args.output.absolute()}")
    logger.info(f"Ratios:  train={args.train}, val={args.val}, test={args.test}")
    logger.info(f"Seed:    {args.seed}")
    logger.info(f"Strategy: {settings.strategy} (config: {args.split_config})")

    # Validate ratios
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        logger.error(f"Ratios must sum to 1.0, got {total:.6f}")
        return 1

    # Discover images
    images_dir = args.source / "images"
    labels_dir = args.source / "labels"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir.absolute()}")
        return 1

    all_images = find_image_files(images_dir)
    if not all_images:
        logger.error("No image files found. Ensure images/ directory is populated.")
        return 1

    logger.info(f"Found {len(all_images)} images")

    # Group by capture session
    groups = group_files_by_key(all_images)
    logger.info(f"Found {len(groups)} capture groups")

    # Compute assignments via the configured strategy
    try:
        strategy = get_strategy(settings.strategy)
        assignments = strategy.assign(
            SplitContext(
                groups=groups,
                train_ratio=args.train,
                val_ratio=args.val,
                test_ratio=args.test,
                seed=args.seed,
                labels_dir=labels_dir,
            )
        )
    except (ValueError, NotImplementedError) as e:
        logger.error(str(e))
        return 1

    if args.dry_run:
        logger.info("Dry run — no files copied")
        for split, keys in assignments.items():
            img_count = sum(len(groups[k]) for k in keys)
            logger.info(f"  {split}: {len(keys)} groups, {img_count} images")
        return 0

    # Copy files
    stats = copy_split_files(
        groups=groups,
        assignments=assignments,
        images_source_dir=images_dir,
        labels_source_dir=labels_dir,
        output_dir=args.output,
    )

    # Verify no leakage
    leakage = verify_no_leakage(args.output)

    # Generate report
    generate_split_report(stats, assignments, groups, args, args.output, leakage)

    # Summary
    total_images = sum(s["images"] for s in stats.values())
    logger.info("=" * 60)
    logger.info(f"Split complete: {total_images} images processed")
    if leakage:
        logger.error(f"DATA LEAKAGE DETECTED: {len(leakage)} files — check split_summary.json")
        return 1

    logger.info("✅ Split complete. No data leakage detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

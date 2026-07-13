"""
scripts.dataset.dataset_stats — Dataset Statistics Generator
=============================================================

Generates comprehensive statistics about a YOLO-format dataset, including
per-class annotation counts, class imbalance metrics, empty/missing class
detection, and safety-critical class summaries.

Output:
    data/qa_reports/dataset_statistics/stats.json
    data/qa_reports/dataset_statistics/stats.csv
    data/qa_reports/dataset_statistics/stats.md

Usage:
    python scripts/dataset/dataset_stats.py
    python scripts/dataset/dataset_stats.py --data-dir data/processed --output data/qa_reports/

DVC integration:
    This script is called by generate_splits.py after splitting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.annotation_utils import parse_label_file
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import find_image_files, find_label_files
from src.utils.report_utils import (
    timestamp_str,
    write_all_formats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Safety-critical class names (from data.yaml comments) ───────────────────

SAFETY_CRITICAL_CLASSES: frozenset[str] = frozenset(
    {
        "medicine_strip",
        "medicine_bottle",
        "knife",
        "stove",
        "gas_cylinder",
        "wire",
        "wet_floor",
    }
)

# ─── Custom data classes (require Indian-home custom captures) ────────────────

CUSTOM_REQUIRED_CLASSES: frozenset[str] = frozenset(
    {
        "gas_cylinder",
        "medicine_strip",
        "wet_floor",
        "walking_stick",
        "support_handle",
        "stove",
        "passport",
        "cupboard",
    }
)

SPLITS: list[str] = ["train", "val", "test"]


# ─── Statistics Computation ───────────────────────────────────────────────────


def compute_split_stats(
    data_dir: Path,
    class_names: dict[int, str],
    num_classes: int,
) -> dict[str, dict]:
    """Compute per-class annotation statistics for all splits.

    Args:
        data_dir:    Root data directory (containing images/ and labels/).
        class_names: class_id → class_name mapping from data.yaml.
        num_classes: Total number of classes.

    Returns:
        Nested dict: split_name → {class_id: {"name", "images", "boxes"}}.
    """
    split_stats: dict[str, dict] = {}

    for split in SPLITS:
        labels_dir = data_dir / "labels" / split
        images_dir = data_dir / "images" / split

        if not labels_dir.exists():
            logger.warning(f"Labels directory not found for split '{split}': {labels_dir}")
            continue

        class_image_counts: dict[int, set] = defaultdict(set)
        class_box_counts: dict[int, int] = defaultdict(int)

        label_files = find_label_files(labels_dir)
        total_images = len(find_image_files(images_dir)) if images_dir.exists() else 0

        for lbl_path in label_files:
            try:
                annotations = parse_label_file(lbl_path)
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Could not read {lbl_path}: {e}")
                continue

            for ann in annotations:
                if 0 <= ann.class_id < num_classes:
                    class_image_counts[ann.class_id].add(lbl_path.stem)
                    class_box_counts[ann.class_id] += 1

        split_stats[split] = {
            "total_images": total_images,
            "total_labels": len(label_files),
            "total_boxes": sum(class_box_counts.values()),
            "classes": {
                cid: {
                    "name": class_names.get(cid, f"class_{cid}"),
                    "images_with_class": len(class_image_counts.get(cid, set())),
                    "bounding_boxes": class_box_counts.get(cid, 0),
                }
                for cid in range(num_classes)
            },
        }

        logger.info(
            f"Split '{split}': {total_images} images, "
            f"{len(label_files)} labels, "
            f"{split_stats[split]['total_boxes']} boxes"
        )

    return split_stats


def compute_imbalance_metrics(
    class_counts: dict[int, int],
    num_classes: int,
) -> dict[str, float]:
    """Compute class imbalance metrics for a single split.

    Args:
        class_counts: class_id → bounding box count.
        num_classes:  Expected total number of classes.

    Returns:
        Dict with keys: max_ratio, min_ratio, imbalance_ratio, gini_coefficient.
    """
    counts = [class_counts.get(i, 0) for i in range(num_classes)]
    total = sum(counts)

    if total == 0:
        return {
            "max_ratio": 0.0,
            "min_ratio": 0.0,
            "imbalance_ratio": 0.0,
            "gini_coefficient": 0.0,
        }

    ratios = [c / total for c in counts]
    max_count = max(counts)
    min_nonzero = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 1

    # Gini coefficient (0 = perfectly balanced, 1 = maximally imbalanced)
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    gini = sum(abs(sorted_counts[i] - sorted_counts[j]) for i in range(n) for j in range(n)) / (
        2 * n * max(total, 1)
    )

    return {
        "max_ratio": max(ratios),
        "min_ratio": min(r for r in ratios if r > 0) if any(r > 0 for r in ratios) else 0.0,
        "imbalance_ratio": round(max_count / max(min_nonzero, 1), 1),
        "gini_coefficient": round(gini, 4),
    }


def find_empty_and_missing_classes(
    class_counts: dict[int, int],
    class_names: dict[int, str],
    num_classes: int,
) -> tuple[list[str], list[str]]:
    """Find classes with zero instances and classes not in the taxonomy.

    Args:
        class_counts: class_id → count dict.
        class_names:  class_id → name dict from data.yaml.
        num_classes:  Total expected number of classes.

    Returns:
        Tuple (empty_class_names, unknown_class_ids_as_strings).
    """
    empty = [
        class_names.get(cid, f"class_{cid}")
        for cid in range(num_classes)
        if class_counts.get(cid, 0) == 0
    ]
    unknown = [str(cid) for cid in class_counts if cid < 0 or cid >= num_classes]
    return empty, unknown


# ─── Report Assembly ──────────────────────────────────────────────────────────


def build_reports(
    split_stats: dict[str, dict],
    class_names: dict[int, str],
    num_classes: int,
    data_dir: Path,
) -> tuple[dict, list[dict], list[dict]]:
    """Build the JSON report dict, CSV rows, and Markdown sections.

    Returns:
        (json_report, csv_rows, md_sections)
    """
    # Aggregate across all splits
    all_splits_box_counts: dict[int, int] = defaultdict(int)
    for split_data in split_stats.values():
        for cid, cdata in split_data.get("classes", {}).items():
            all_splits_box_counts[int(cid)] += cdata["bounding_boxes"]

    empty_classes, unknown_ids = find_empty_and_missing_classes(
        all_splits_box_counts, class_names, num_classes
    )

    # Safety-critical summary
    safety_stats = {
        name: all_splits_box_counts.get(cid, 0)
        for cid, name in class_names.items()
        if name in SAFETY_CRITICAL_CLASSES
    }

    # Custom-required summary
    custom_stats = {
        name: all_splits_box_counts.get(cid, 0)
        for cid, name in class_names.items()
        if name in CUSTOM_REQUIRED_CLASSES
    }

    # Overall imbalance (across all splits combined)
    imbalance = compute_imbalance_metrics(all_splits_box_counts, num_classes)

    # JSON report
    json_report: dict = {
        "timestamp": timestamp_str(),
        "data_dir": str(data_dir.absolute()),
        "num_classes": num_classes,
        "empty_classes": empty_classes,
        "unknown_class_ids": unknown_ids,
        "safety_critical_counts": safety_stats,
        "custom_required_counts": custom_stats,
        "imbalance": imbalance,
        "splits": split_stats,
    }

    # CSV rows (one row per class per split)
    csv_rows: list[dict] = []
    for split, split_data in split_stats.items():
        for cid_str, cdata in split_data.get("classes", {}).items():
            cid = int(cid_str)
            name = cdata["name"]
            csv_rows.append(
                {
                    "split": split,
                    "class_id": cid,
                    "class_name": name,
                    "images_with_class": cdata["images_with_class"],
                    "bounding_boxes": cdata["bounding_boxes"],
                    "is_safety_critical": name in SAFETY_CRITICAL_CLASSES,
                    "is_custom_required": name in CUSTOM_REQUIRED_CLASSES,
                    "total_split_boxes": split_data["total_boxes"],
                    "pct_of_split": round(
                        100.0 * cdata["bounding_boxes"] / max(split_data["total_boxes"], 1), 2
                    ),
                }
            )

    # Markdown sections
    # Summary table
    total_rows = [
        [
            str(cid),
            class_names.get(cid, f"class_{cid}"),
            str(all_splits_box_counts.get(cid, 0)),
            "✅" if class_names.get(cid, "") in SAFETY_CRITICAL_CLASSES else "",
            "🇮🇳" if class_names.get(cid, "") in CUSTOM_REQUIRED_CLASSES else "",
            "⚠️ EMPTY" if all_splits_box_counts.get(cid, 0) == 0 else "",
        ]
        for cid in range(num_classes)
    ]

    split_summary_rows = [
        [
            split.upper(),
            str(split_stats.get(split, {}).get("total_images", 0)),
            str(split_stats.get(split, {}).get("total_labels", 0)),
            str(split_stats.get(split, {}).get("total_boxes", 0)),
        ]
        for split in SPLITS
        if split in split_stats
    ]

    md_sections: list[dict] = [
        {
            "heading": "Split Overview",
            "table": {
                "headers": ["Split", "Images", "Labels", "Bounding Boxes"],
                "rows": split_summary_rows,
            },
        },
        {
            "heading": "Class Distribution (All Splits Combined)",
            "table": {
                "headers": [
                    "ID",
                    "Class",
                    "Total Boxes",
                    "Safety-Critical",
                    "Custom Required",
                    "Status",
                ],
                "rows": total_rows,
            },
        },
        {
            "heading": "Class Imbalance",
            "content": (
                f"- **Imbalance ratio** (max/min instances): "
                f"`{imbalance['imbalance_ratio']:.1f}x`\n"
                f"- **Gini coefficient**: `{imbalance['gini_coefficient']:.4f}` "
                f"(0=balanced, 1=maximally imbalanced)\n"
                f"- **Empty classes** ({len(empty_classes)}): "
                + (", ".join(f"`{c}`" for c in empty_classes) if empty_classes else "None")
            ),
        },
        {
            "heading": "Safety-Critical Class Summary",
            "content": "\n".join(
                f"- **{name}**: {count} instances" + (" ⚠️ ZERO" if count == 0 else "")
                for name, count in sorted(safety_stats.items())
            ),
        },
    ]

    return json_report, csv_rows, md_sections


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics for a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root data directory (must contain images/ and labels/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml for class configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qa_reports/dataset_statistics"),
        help="Output directory for statistics reports.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Dataset Statistics Generator — Elderly Assistant System")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir.absolute()}")
    logger.info(f"Config:   {args.config}")
    logger.info(f"Output:   {args.output.absolute()}")

    # Load class config
    try:
        data_cfg = load_data_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data.yaml: {e}")
        return 1

    class_names = get_class_names_from_data_yaml(data_cfg)
    num_classes = data_cfg.get("nc", len(class_names))
    logger.info(f"Loaded {num_classes} classes from {args.config}")

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir.absolute()}")
        return 1

    # Compute statistics
    split_stats = compute_split_stats(args.data_dir, class_names, num_classes)

    if not split_stats:
        logger.error("No split data found. Run split_dataset.py first.")
        return 1

    # Build reports
    json_report, csv_rows, md_sections = build_reports(
        split_stats, class_names, num_classes, args.data_dir
    )

    # Write all formats
    paths = write_all_formats(
        report_data=json_report,
        csv_rows=csv_rows,
        md_title="Dataset Statistics — Elderly Assistant System",
        md_sections=md_sections,
        output_dir=args.output,
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
        md_metadata={
            "Data directory": str(args.data_dir.absolute()),
            "Config": str(args.config),
            "Total classes": num_classes,
        },
    )

    logger.info("=" * 60)
    logger.info("✅ Statistics generated:")
    for fmt, path in paths.items():
        logger.info(f"   {fmt.upper():10s}: {path}")

    # Warn about empty classes
    empty = json_report.get("empty_classes", [])
    if empty:
        logger.warning(f"⚠️  {len(empty)} empty classes: {', '.join(empty)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

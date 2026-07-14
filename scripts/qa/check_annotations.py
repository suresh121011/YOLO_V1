"""
scripts.qa.check_annotations — Comprehensive Annotation QA Pipeline
====================================================================

Runs all annotation quality checks on a YOLO-format dataset and generates
structured reports. This is the single QA entry point for the dataset
validation workflow.

Checks performed:
    CRITICAL (block training if found):
        - invalid_class_ids      — Class IDs outside [0, num_classes)
        - bbox_out_of_bounds     — Bounding box coordinates outside [0, 1]
        - zero_area_boxes        — Bounding boxes with w=0 or h=0
        - invalid_yolo_format    — Lines with ≠ 5 whitespace-separated fields
        - corrupted_images       — Images that cannot be opened/decoded
        - train_val_leakage      — Same image hash in both train and val
        - train_test_leakage     — Same image hash in both train and test

    WARNING (log, review before training):
        - empty_label_files      — Label files with no annotations
        - missing_label_files    — Images without a corresponding .txt file
        - missing_image_files    — Label files without a corresponding image
        - duplicate_images       — Multiple images with identical content hash
        - duplicate_annotations  — Identical annotation lines in same file
        - unknown_class_names    — Class IDs not in data.yaml names mapping

    INFO:
        - inconsistent_pairs     — Image/label count mismatch per split

Outputs:
    data/qa_reports/annotation_qa_report.json
    data/qa_reports/annotation_qa_report.csv
    data/qa_reports/annotation_qa_report.md

Exit codes:
    0 — All checks pass (or only INFO-level issues)
    1 — CRITICAL issues found (do not proceed to training)
    2 — WARNING issues found (review before training)

Usage:
    python scripts/qa/check_annotations.py
    python scripts/qa/check_annotations.py --data-dir data/processed --output data/qa_reports/
    python scripts/qa/check_annotations.py --strict   # treat warnings as critical

DVC integration:
    This script is invoked by the qa_check DVC stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.annotation_utils import (
    check_duplicate_lines,
    parse_label_file_raw,
)
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import (
    build_hash_index,
    find_image_files,
    find_label_files,
    get_image_label_pairs,
)
from src.utils.image_utils import validate_image
from src.utils.report_utils import (
    format_severity_badge,
    timestamp_str,
    write_all_formats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPLITS: list[str] = ["train", "val", "test"]
Severity = Literal["CRITICAL", "WARNING", "INFO"]


# ─── Issue data structure ─────────────────────────────────────────────────────


@dataclass
class QAIssue:
    """A single QA issue discovered during annotation validation.

    Args:
        check:    Name of the check that discovered this issue.
        severity: Severity level (CRITICAL / WARNING / INFO).
        split:    Dataset split where the issue was found.
        file:     File path where the issue was found (as string).
        line:     Line number within the file (0 if file-level issue).
        message:  Human-readable description of the issue.
    """

    check: str
    severity: str  # "CRITICAL" | "WARNING" | "INFO"
    split: str
    file: str
    line: int = 0
    message: str = ""


@dataclass
class QAResults:
    """Aggregated results from all QA checks.

    Args:
        issues:           All discovered QAIssue objects.
        check_summaries:  Per-check summary dict {check_name: {status, count}}.
        total_images:     Total images scanned.
        total_labels:     Total label files scanned.
        total_boxes:      Total bounding boxes validated.
    """

    issues: list[QAIssue] = field(default_factory=list)
    check_summaries: dict[str, dict] = field(default_factory=dict)
    total_images: int = 0
    total_labels: int = 0
    total_boxes: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "CRITICAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "INFO")

    def add_issue(self, issue: QAIssue) -> None:
        self.issues.append(issue)

    def finalize_check(self, check: str, issue_count: int) -> None:
        """Mark a check as complete with its total issue count."""
        if issue_count == 0:
            status = "PASS"
        elif any(i.check == check and i.severity == "CRITICAL" for i in self.issues):
            status = "CRITICAL"
        elif any(i.check == check and i.severity == "WARNING" for i in self.issues):
            status = "WARNING"
        else:
            status = "INFO"

        self.check_summaries[check] = {"status": status, "count": issue_count}
        logger.info(f"Check '{check}': {status} ({issue_count} issues)")


# ─── Individual Checks ────────────────────────────────────────────────────────


def check_annotation_format(
    data_dir: Path,
    class_names: dict[int, str],
    num_classes: int,
    results: QAResults,
) -> None:
    """Check all label files for format validity, invalid class IDs, and bbox bounds.

    Covers:
        - invalid_yolo_format
        - invalid_class_ids
        - bbox_out_of_bounds
        - zero_area_boxes
        - duplicate_annotations
        - unknown_class_names
    """
    checks = [
        "invalid_yolo_format",
        "invalid_class_ids",
        "bbox_out_of_bounds",
        "zero_area_boxes",
        "duplicate_annotations",
        "unknown_class_names",
    ]
    check_counts: dict[str, int] = {c: 0 for c in checks}

    for split in SPLITS:
        labels_dir = data_dir / "labels" / split
        if not labels_dir.exists():
            continue

        for lbl_path in find_label_files(labels_dir):
            # Duplicate line check
            raw_lines = parse_label_file_raw(lbl_path)
            dupes = check_duplicate_lines(raw_lines)
            for first_idx, dupe_idx in dupes:
                results.add_issue(
                    QAIssue(
                        check="duplicate_annotations",
                        severity="WARNING",
                        split=split,
                        file=str(lbl_path),
                        line=dupe_idx + 1,
                        message=(
                            f"Line {dupe_idx + 1} is a duplicate of line "
                            f"{first_idx + 1}: '{raw_lines[dupe_idx]}'"
                        ),
                    )
                )
                check_counts["duplicate_annotations"] += 1

            # Parse and validate each annotation
            content = lbl_path.read_text(encoding="utf-8", errors="replace")
            for line_num, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.split()

                # Format check
                if len(parts) != 5:
                    results.add_issue(
                        QAIssue(
                            check="invalid_yolo_format",
                            severity="CRITICAL",
                            split=split,
                            file=str(lbl_path),
                            line=line_num,
                            message=f"Expected 5 fields, got {len(parts)}: '{stripped[:80]}'",
                        )
                    )
                    check_counts["invalid_yolo_format"] += 1
                    continue

                results.total_boxes += 1

                try:
                    class_id = int(parts[0])
                    cx, cy = float(parts[1]), float(parts[2])
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    results.add_issue(
                        QAIssue(
                            check="invalid_yolo_format",
                            severity="CRITICAL",
                            split=split,
                            file=str(lbl_path),
                            line=line_num,
                            message=f"Cannot parse numeric fields: '{stripped[:80]}'",
                        )
                    )
                    check_counts["invalid_yolo_format"] += 1
                    continue

                # Class ID range
                if class_id < 0 or class_id >= num_classes:
                    results.add_issue(
                        QAIssue(
                            check="invalid_class_ids",
                            severity="CRITICAL",
                            split=split,
                            file=str(lbl_path),
                            line=line_num,
                            message=(
                                f"class_id={class_id} outside valid range "
                                f"[0, {num_classes - 1}]"
                            ),
                        )
                    )
                    check_counts["invalid_class_ids"] += 1

                # Unknown class name check
                elif class_id not in class_names:
                    results.add_issue(
                        QAIssue(
                            check="unknown_class_names",
                            severity="WARNING",
                            split=split,
                            file=str(lbl_path),
                            line=line_num,
                            message=f"class_id={class_id} has no name in data.yaml",
                        )
                    )
                    check_counts["unknown_class_names"] += 1

                # Zero-area box
                if w <= 0.0 or h <= 0.0:
                    results.add_issue(
                        QAIssue(
                            check="zero_area_boxes",
                            severity="CRITICAL",
                            split=split,
                            file=str(lbl_path),
                            line=line_num,
                            message=f"Zero or negative dimensions: w={w}, h={h}",
                        )
                    )
                    check_counts["zero_area_boxes"] += 1

                # BBox bounds check
                for coord_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if not (0.0 <= val <= 1.0):
                        results.add_issue(
                            QAIssue(
                                check="bbox_out_of_bounds",
                                severity="CRITICAL",
                                split=split,
                                file=str(lbl_path),
                                line=line_num,
                                message=f"{coord_name}={val:.6f} outside [0, 1]",
                            )
                        )
                        check_counts["bbox_out_of_bounds"] += 1
                        break  # One error per line for bbox bounds

    for check in checks:
        results.finalize_check(check, check_counts[check])


def check_file_pairs(data_dir: Path, results: QAResults) -> None:
    """Check for missing/unpaired image and label files.

    Covers:
        - missing_label_files
        - missing_image_files
        - empty_label_files
        - inconsistent_pairs
    """
    checks = [
        "missing_label_files",
        "missing_image_files",
        "empty_label_files",
        "inconsistent_pairs",
    ]
    check_counts: dict[str, int] = {c: 0 for c in checks}

    for split in SPLITS:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists() and not labels_dir.exists():
            continue

        # Images without labels
        if images_dir.exists():
            pairs = get_image_label_pairs(images_dir, labels_dir)
            results.total_images += len(pairs)

            for img_path, lbl_path in pairs:
                if lbl_path is None:
                    results.add_issue(
                        QAIssue(
                            check="missing_label_files",
                            severity="WARNING",
                            split=split,
                            file=str(img_path),
                            message=f"No label file found for image: {img_path.name}",
                        )
                    )
                    check_counts["missing_label_files"] += 1

        # Labels without images
        if labels_dir.exists():
            label_files = find_label_files(labels_dir)
            results.total_labels += len(label_files)

            image_stems = set()
            if images_dir.exists():
                image_stems = {p.stem for p in find_image_files(images_dir)}

            for lbl_path in label_files:
                if lbl_path.stem not in image_stems:
                    results.add_issue(
                        QAIssue(
                            check="missing_image_files",
                            severity="WARNING",
                            split=split,
                            file=str(lbl_path),
                            message=f"No image found for label: {lbl_path.name}",
                        )
                    )
                    check_counts["missing_image_files"] += 1

                # Empty label check
                content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
                if not content:
                    results.add_issue(
                        QAIssue(
                            check="empty_label_files",
                            severity="WARNING",
                            split=split,
                            file=str(lbl_path),
                            message=f"Label file is empty: {lbl_path.name}",
                        )
                    )
                    check_counts["empty_label_files"] += 1

            # Inconsistent pair count
            n_images = len(find_image_files(images_dir)) if images_dir.exists() else 0
            n_labels = len(label_files)
            if n_images != n_labels:
                results.add_issue(
                    QAIssue(
                        check="inconsistent_pairs",
                        severity="INFO",
                        split=split,
                        file=str(data_dir / split),
                        message=(
                            f"Image/label count mismatch in '{split}': "
                            f"{n_images} images vs {n_labels} labels"
                        ),
                    )
                )
                check_counts["inconsistent_pairs"] += 1

    for check in checks:
        results.finalize_check(check, check_counts[check])


def check_corrupted_images(data_dir: Path, results: QAResults) -> None:
    """Validate that all images can be read and decoded.

    Covers:
        - corrupted_images
    """
    issue_count = 0

    for split in SPLITS:
        images_dir = data_dir / "images" / split
        if not images_dir.exists():
            continue

        for img_path in find_image_files(images_dir):
            ok, msg = validate_image(img_path)
            if not ok:
                results.add_issue(
                    QAIssue(
                        check="corrupted_images",
                        severity="CRITICAL",
                        split=split,
                        file=str(img_path),
                        message=msg,
                    )
                )
                issue_count += 1

    results.finalize_check("corrupted_images", issue_count)


def check_duplicate_images(data_dir: Path, results: QAResults) -> None:
    """Find duplicate image files within each split.

    Covers:
        - duplicate_images
    """
    issue_count = 0

    for split in SPLITS:
        images_dir = data_dir / "images" / split
        if not images_dir.exists():
            continue

        all_images = find_image_files(images_dir)
        hash_index = build_hash_index(all_images)

        for digest, files in hash_index.items():
            if len(files) > 1:
                file_names = [f.name for f in files]
                for dup_file in files[1:]:
                    results.add_issue(
                        QAIssue(
                            check="duplicate_images",
                            severity="WARNING",
                            split=split,
                            file=str(dup_file),
                            message=f"Duplicate of {files[0].name} (SHA-256: {digest[:12]}…). "
                            f"All copies: {file_names}",
                        )
                    )
                    issue_count += 1

    results.finalize_check("duplicate_images", issue_count)


def check_split_leakage(data_dir: Path, results: QAResults) -> None:
    """Detect the same image appearing in multiple splits (data leakage).

    Uses SHA-256 hash comparison to catch exact duplicates across splits.

    Covers:
        - train_val_leakage
        - train_test_leakage
    """
    # Build hash index per split
    split_hashes: dict[str, dict[str, Path]] = {}

    for split in SPLITS:
        images_dir = data_dir / "images" / split
        if not images_dir.exists():
            split_hashes[split] = {}
            continue

        files = find_image_files(images_dir)
        hash_index = build_hash_index(files)
        # Map hash → first file only (for reporting)
        split_hashes[split] = {h: paths[0] for h, paths in hash_index.items()}

    check_pairs = [
        ("train", "val", "train_val_leakage"),
        ("train", "test", "train_test_leakage"),
    ]

    for split_a, split_b, check_name in check_pairs:
        hashes_a = split_hashes.get(split_a, {})
        hashes_b = split_hashes.get(split_b, {})

        overlap = set(hashes_a.keys()) & set(hashes_b.keys())
        for digest in sorted(overlap):
            file_a = hashes_a[digest]
            file_b = hashes_b[digest]
            results.add_issue(
                QAIssue(
                    check=check_name,
                    severity="CRITICAL",
                    split=f"{split_a}+{split_b}",
                    file=str(file_a),
                    message=(
                        f"Same image in '{split_a}' and '{split_b}': "
                        f"{file_a.name} ↔ {file_b.name} "
                        f"(SHA-256: {digest[:12]}…)"
                    ),
                )
            )

        results.finalize_check(check_name, len(overlap))
        if overlap:
            logger.error(
                f"DATA LEAKAGE: {len(overlap)} images shared between "
                f"'{split_a}' and '{split_b}'"
            )


# ─── Report Assembly ──────────────────────────────────────────────────────────


def portable_path(path: Path) -> str:
    """Render a path for committed reports: cwd-relative, posix separators.

    Absolute machine paths (user names, OneDrive roots) must never land in
    versioned QA artifacts. Paths outside the working directory are kept
    as given rather than absolutized.
    """
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def build_qa_reports(
    results: QAResults,
    data_dir: Path,
    num_classes: int,
) -> tuple[dict, list[dict], list[dict]]:
    """Build JSON report dict, CSV rows, and Markdown sections from QAResults.

    Returns:
        (json_report, csv_rows, md_sections)
    """
    # JSON
    json_report = {
        "timestamp": timestamp_str(),
        "data_dir": portable_path(data_dir),
        "num_classes": num_classes,
        "summary": {
            "total_images": results.total_images,
            "total_labels": results.total_labels,
            "total_boxes": results.total_boxes,
            "critical_issues": results.critical_count,
            "warning_issues": results.warning_count,
            "info_issues": results.info_count,
            "total_issues": len(results.issues),
        },
        "checks": results.check_summaries,
        "issues": [
            {
                "check": i.check,
                "severity": i.severity,
                "split": i.split,
                "file": portable_path(Path(i.file)) if i.file else i.file,
                "line": i.line,
                "message": i.message,
            }
            for i in results.issues
        ],
    }

    # CSV
    csv_rows = [
        {
            "severity": i.severity,
            "check": i.check,
            "split": i.split,
            "file": Path(i.file).name if i.file else "",
            "line": i.line if i.line else "",
            "message": i.message,
        }
        for i in results.issues
    ]

    # Markdown
    overall_status = (
        "🔴 CRITICAL"
        if results.critical_count > 0
        else "🟡 WARNING" if results.warning_count > 0 else "✅ PASS"
    )

    check_table_rows = [
        [
            check,
            format_severity_badge(data["status"]),
            str(data["count"]),
        ]
        for check, data in sorted(results.check_summaries.items())
    ]

    # First 50 issues for readability
    issue_rows = [
        [
            format_severity_badge(i.severity),
            i.check,
            i.split,
            Path(i.file).name if i.file else "",
            str(i.line) if i.line else "",
            i.message[:100] + ("…" if len(i.message) > 100 else ""),
        ]
        for i in sorted(
            results.issues, key=lambda x: ["CRITICAL", "WARNING", "INFO"].index(x.severity)
        )[:50]
    ]

    md_sections = [
        {
            "heading": f"Overall Status: {overall_status}",
            "content": (
                f"- **Images scanned:** {results.total_images}\n"
                f"- **Label files scanned:** {results.total_labels}\n"
                f"- **Bounding boxes validated:** {results.total_boxes}\n"
                f"- **Critical issues:** {results.critical_count}\n"
                f"- **Warnings:** {results.warning_count}\n"
                f"- **Info:** {results.info_count}"
            ),
        },
        {
            "heading": "Check Results",
            "table": {
                "headers": ["Check", "Status", "Issues Found"],
                "rows": check_table_rows,
            },
        },
    ]

    if results.issues:
        md_sections.append(
            {
                "heading": f"Issues (showing {len(issue_rows)} of {len(results.issues)})",
                "table": {
                    "headers": ["Severity", "Check", "Split", "File", "Line", "Message"],
                    "rows": issue_rows,
                },
            }
        )

    if results.critical_count > 0:
        md_sections.append(
            {
                "heading": "Action Required",
                "content": (
                    "> [!CAUTION]\n"
                    f"> **{results.critical_count} critical issues found.** "
                    "Do not proceed to training until all CRITICAL issues are resolved."
                ),
            }
        )

    return json_report, csv_rows, md_sections


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive annotation QA on a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root data directory (must contain images/ and labels/ subdirs).",
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
        default=Path("data/qa_reports"),
        help="Output directory for QA reports.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARNING-level issues as CRITICAL (exit code 1).",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip corrupted image check (faster, for annotation-only QA).",
    )
    parser.add_argument(
        "--skip-duplicate-check",
        action="store_true",
        help="Skip duplicate image check (faster, for large datasets).",
    )
    return parser.parse_args()


def main() -> int:
    """Main QA entry point. Returns exit code (0/1/2)."""
    # Windows consoles often use cp1252, which cannot encode the emoji
    # badges printed in the summary — degrade to '?' instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()

    logger.info("=" * 60)
    logger.info("Annotation QA Pipeline — Elderly Assistant System")
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
    logger.info(f"Loaded {num_classes} classes")

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir.absolute()}")
        return 1

    results = QAResults()

    # ── Run all checks ────────────────────────────────────────────────────────
    logger.info("Running checks…")

    logger.info("  [1/5] Annotation format and class validation…")
    check_annotation_format(args.data_dir, class_names, num_classes, results)

    logger.info("  [2/5] Image/label pair validation…")
    check_file_pairs(args.data_dir, results)

    if not args.skip_image_validation:
        logger.info("  [3/5] Image integrity validation…")
        check_corrupted_images(args.data_dir, results)
    else:
        logger.info("  [3/5] Image integrity check: SKIPPED (--skip-image-validation)")

    if not args.skip_duplicate_check:
        logger.info("  [4/5] Duplicate image detection…")
        check_duplicate_images(args.data_dir, results)
    else:
        logger.info("  [4/5] Duplicate detection: SKIPPED (--skip-duplicate-check)")

    logger.info("  [5/5] Cross-split leakage detection…")
    check_split_leakage(args.data_dir, results)

    # ── Generate reports ──────────────────────────────────────────────────────
    json_report, csv_rows, md_sections = build_qa_reports(results, args.data_dir, num_classes)

    paths = write_all_formats(
        report_data=json_report,
        csv_rows=csv_rows,
        md_title="Annotation QA Report — Elderly Assistant System",
        md_sections=md_sections,
        output_dir=args.output,
        base_name="annotation_qa_report",
        csv_fieldnames=["severity", "check", "split", "file", "line", "message"],
        md_metadata={
            "Data directory": portable_path(args.data_dir),
            "Classes": num_classes,
            "Images scanned": results.total_images,
            "Boxes validated": results.total_boxes,
        },
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  QA Results")
    print("=" * 60)
    for check, data in sorted(results.check_summaries.items()):
        badge = format_severity_badge(data["status"])
        print(f"  {badge:<25}  {check}  ({data['count']} issues)")
    print()
    print(f"  Total images:  {results.total_images}")
    print(f"  Total labels:  {results.total_labels}")
    print(f"  Total boxes:   {results.total_boxes}")
    print(f"  Critical:      {results.critical_count}")
    print(f"  Warnings:      {results.warning_count}")
    print()
    for fmt, path in paths.items():
        print(f"  {fmt.upper():10s}: {path}")
    print("=" * 60)
    print()

    # ── Exit code ─────────────────────────────────────────────────────────────
    if results.critical_count > 0:
        logger.error("❌ CRITICAL ISSUES FOUND — Do not proceed to training")
        return 1

    if results.warning_count > 0:
        if args.strict:
            logger.error("❌ WARNING issues found (--strict mode: treating as CRITICAL)")
            return 1
        logger.warning("⚠️  Warnings found — review before training")
        return 2

    logger.info("✅ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

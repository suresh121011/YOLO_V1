"""
scripts.qa.run_full_qa — Master QA Orchestrator
===============================================

Single entry point for dataset QA (the ``qa_check`` DVC stage). Runs, in
order:

    1. check_annotations.py   — 15 structural checks (CRITICAL/WARNING)
    2. dataset_stats.py       — per-class statistics + imbalance reports
    3. license gate           — non-commercial sources vs allow_noncommercial
    4. label completeness     — trusted-classes map from the merged manifest
    5. image quality          — blur (Laplacian variance) and low-light
                                (mean brightness) WARNING checks (risk R01)

Results 3–5 are merged INTO ``data/qa_reports/annotation_qa_report.json``
(the DVC metric file) so one artifact carries the whole QA verdict.

Exit codes: 0 = pass, 1 = critical failures, 2 = warnings only.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.manifest import MANIFEST_FILENAME, MERGED_MANIFEST_FILENAME
from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config
from src.utils.dataset_utils import find_image_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

# Image-quality thresholds (WARNING level — mitigates documented risk R01)
BLUR_VARIANCE_THRESHOLD = 100.0  # Laplacian variance below this → blurry
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 40.0  # mean grayscale below this → too dark
MAX_LISTED_FILES = 30  # cap example filenames embedded in the report


def run_step(cmd: list[str], name: str) -> int:
    """Run a child QA script, streaming its output. Returns its exit code."""
    logger.info(f"── running {name}: {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, check=False)
    logger.info(f"── {name} exit code: {result.returncode}")
    return result.returncode


def build_license_report(
    raw_root: Path,
    sources_config_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Collect per-source licenses from manifests and evaluate the gate.

    Returns:
        (report dict, has_critical) — critical when a non-commercial source
        contributed data while allow_noncommercial is false.
    """
    config = load_sources_config(sources_config_path)
    entries: list[dict[str, Any]] = []
    violation = False

    for manifest_path in sorted(raw_root.glob(f"*/{MANIFEST_FILENAME}")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_name = str(data.get("source", manifest_path.parent.name))
        source_cfg = config.sources.get(source_name)
        noncommercial = bool(source_cfg.noncommercial) if source_cfg else False
        gate_violation = noncommercial and not config.allow_noncommercial
        violation = violation or (gate_violation and int(data.get("image_count", 0)) > 0)
        entries.append(
            {
                "source": source_name,
                "license": data.get("license", ""),
                "noncommercial": noncommercial,
                "image_count": data.get("image_count", 0),
                "gate_violation": gate_violation,
            }
        )

    return {
        "allow_noncommercial": config.allow_noncommercial,
        "sources": entries,
        "violation": violation,
    }, violation


def load_label_completeness(merged_dir: Path) -> dict[str, Any]:
    """Pull label-completeness + acceptance stats from the merged manifest."""
    manifest_path = merged_dir / MERGED_MANIFEST_FILENAME
    if not manifest_path.exists():
        logger.warning(f"No merged manifest at {manifest_path} — completeness unknown")
        return {"available": False}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "label_completeness": data.get("label_completeness", {}),
        "sources": data.get("sources", []),
        "duplicates_removed": data.get("duplicates_removed", 0),
        "filtered_out": data.get("filtered_out", 0),
    }


def check_image_quality(data_dir: Path) -> tuple[dict[str, Any], int]:
    """Blur + low-light WARNING checks over all processed images.

    Returns:
        (report dict, warning_count). Skips gracefully without OpenCV.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — blur/low-light checks skipped")
        return {"available": False}, 0

    blurry: list[str] = []
    dark: list[str] = []
    scanned = 0

    for image_path in find_image_files(data_dir / "images"):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        scanned += 1
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_VARIANCE_THRESHOLD:
            blurry.append(image_path.name)
        if float(gray.mean()) < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
            dark.append(image_path.name)

    warnings = len(blurry) + len(dark)
    logger.info(f"Image quality: {scanned} scanned, {len(blurry)} blurry, {len(dark)} low-light")
    return {
        "available": True,
        "scanned": scanned,
        "blur_variance_threshold": BLUR_VARIANCE_THRESHOLD,
        "low_light_brightness_threshold": LOW_LIGHT_BRIGHTNESS_THRESHOLD,
        "blurry_count": len(blurry),
        "blurry_samples": blurry[:MAX_LISTED_FILES],
        "low_light_count": len(dark),
        "low_light_samples": dark[:MAX_LISTED_FILES],
    }, warnings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full dataset QA suite (structural + governance).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_SOURCES_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/qa_reports"))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as critical.")
    parser.add_argument(
        "--skip-image-checks",
        action="store_true",
        help="Skip blur/low-light scanning (faster).",
    )
    parser.add_argument(
        "--exit-zero-on-warnings",
        action="store_true",
        help="Return 0 instead of 2 when only warnings are found (used by "
        "the DVC qa_check stage, where warnings must not fail the DAG; "
        "criticals still exit 1).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 (pass), 1 (critical), or 2 (warnings)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()
    logger.info("=" * 60)
    logger.info("Full QA Suite — Elderly Assistant System")
    logger.info("=" * 60)

    # 1. Structural annotation QA (writes annotation_qa_report.json)
    annotations_exit = run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "check_annotations.py"),
            "--data-dir",
            str(args.data_dir),
            "--config",
            str(args.config),
            "--output",
            str(args.output),
            *(["--strict"] if args.strict else []),
        ],
        "check_annotations",
    )

    # 2. Dataset statistics (independent reports; failures are warnings)
    stats_exit = run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR.parent / "dataset" / "dataset_stats.py"),
            "--data-dir",
            str(args.data_dir),
            "--config",
            str(args.config),
            "--output",
            str(args.output),
        ],
        "dataset_stats",
    )

    # 3–5. Governance + quality enrichment
    license_report, license_critical = build_license_report(args.raw_root, args.sources_config)
    completeness = load_label_completeness(args.merged_dir)
    if args.skip_image_checks:
        quality_report: dict[str, Any] = {"available": False}
        quality_warnings = 0
    else:
        quality_report, quality_warnings = check_image_quality(args.data_dir)

    # Merge everything into the DVC metric file
    report_path = args.output / "annotation_qa_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["license_report"] = license_report
    report["label_completeness"] = completeness
    report["image_quality"] = quality_report
    report["orchestrator"] = {
        "check_annotations_exit": annotations_exit,
        "dataset_stats_exit": stats_exit,
        "license_critical": license_critical,
        "image_quality_warnings": quality_warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Verdict
    critical = annotations_exit == 1 or license_critical
    warnings = annotations_exit == 2 or stats_exit != 0 or quality_warnings > 0
    if license_critical:
        logger.error("LICENSE GATE VIOLATION — see license_report in the QA report")

    logger.info("=" * 60)
    if critical:
        logger.error("QA VERDICT: CRITICAL FAILURES — dataset must not be released")
        return 1
    if warnings:
        if args.strict:
            logger.warning("QA VERDICT: warnings present (strict mode → exit 1)")
            return 1
        exit_code = 0 if args.exit_zero_on_warnings else 2
        logger.warning(f"QA VERDICT: warnings present (exit {exit_code})")
        return exit_code
    logger.info("QA VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

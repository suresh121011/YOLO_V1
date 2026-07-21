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
    6. eval-set overlap       — the locked Phase-3 eval set (data/eval/) must
                                share NO image, exact or flip-robust
                                near-duplicate, with train-facing data
    7. house exclusivity      — a house_id in both training captures and the
                                eval set undermines eval as an unseen-home
                                signal (WARNING, not CRITICAL — training
                                mAP is still valid, only the eval claim weakens)

Results 3–7 are merged INTO ``data/qa_reports/annotation_qa_report.json``
(the DVC metric file) so one artifact carries the whole QA verdict. Checks
6–7 are opportunistic: before any eval set exists they report
``{"available": false}`` rather than failing, so ``qa_check`` stays green
on machines/branches without Phase-3 data (see docs/04 §6).

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

from src.dataset.dedup import compute_image_hashes
from src.dataset.manifest import (
    MANIFEST_FILENAME,
    MERGED_MANIFEST_FILENAME,
    CaptureSessionManifest,
)
from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config
from src.utils.dataset_utils import build_hash_index, compute_file_hash, find_image_files

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


def check_eval_overlap(
    eval_dir: Path,
    processed_dir: Path,
    merged_dir: Path,
    hamming_threshold: int = 5,
) -> tuple[dict[str, Any], bool]:
    """Detect any image shared between the locked eval set and train-facing data.

    Two independent checks against the union of ``data/merged/images`` and
    ``data/processed/images`` (all splits): exact SHA-256 match, and
    flip-robust perceptual near-duplicate (same aHash comparison as
    src/dataset/dedup.py, catching pre-flipped copies). Eval images are
    compared only against the frozen train-facing set, never against each
    other, so eval-internal duplicates are not mistaken for leakage.

    Args:
        eval_dir:          Eval set root (e.g. data/eval/indian_home_v0).
        processed_dir:     data/processed (has an images/ subtree per split).
        merged_dir:         data/merged (flat images/ subtree).
        hamming_threshold: Hamming distance below this → near-duplicate
                           (matches configs/dataset_sources.yaml dedup
                           settings by default).

    Returns:
        (report dict, has_critical) — critical when any overlap is found.
    """
    eval_images_dir = eval_dir / "images"
    if not eval_images_dir.exists():
        return {"available": False}, False

    train_facing = find_image_files(merged_dir / "images") + find_image_files(processed_dir)
    train_hashes = build_hash_index(train_facing)
    train_shas = set(train_hashes.keys())

    train_perceptual: list[tuple[int, int, str]] = []
    for path in train_facing:
        hashes = compute_image_hashes(path, check_flip=True)
        if hashes.ahash is not None:
            train_perceptual.append(
                (
                    int(hashes.ahash, 16),
                    int(hashes.flip_ahash, 16) if hashes.flip_ahash else -1,
                    path.name,
                )
            )

    eval_images = find_image_files(eval_images_dir)
    exact_overlaps: list[str] = []
    near_overlaps: list[str] = []

    for eval_image in eval_images:
        try:
            digest = compute_file_hash(eval_image)
        except OSError:
            continue
        if digest in train_shas:
            exact_overlaps.append(f"{eval_image.name} == {train_hashes[digest][0].name}")
            continue

        hashes = compute_image_hashes(eval_image, check_flip=False)
        if hashes.ahash is None:
            continue
        eval_int = int(hashes.ahash, 16)
        for train_a, train_flip, train_name in train_perceptual:
            if (train_a ^ eval_int).bit_count() < hamming_threshold or (
                train_flip >= 0 and (train_flip ^ eval_int).bit_count() < hamming_threshold
            ):
                near_overlaps.append(f"{eval_image.name} ~ {train_name}")
                break

    total_overlaps = len(exact_overlaps) + len(near_overlaps)
    if total_overlaps:
        logger.error(
            f"EVAL SET LEAKAGE: {len(exact_overlaps)} exact + {len(near_overlaps)} "
            f"near-duplicate overlap(s) with train-facing data"
        )

    return {
        "available": True,
        "eval_image_count": len(eval_images),
        "train_facing_image_count": len(train_facing),
        "exact_overlap_count": len(exact_overlaps),
        "exact_overlaps": exact_overlaps[:MAX_LISTED_FILES],
        "near_overlap_count": len(near_overlaps),
        "near_overlaps": near_overlaps[:MAX_LISTED_FILES],
    }, total_overlaps > 0


def check_house_exclusivity(captures_root: Path, eval_dir: Path) -> dict[str, Any]:
    """Warn when a house_id appears in both training captures and the eval set.

    Training data from a house that also supplies eval images lets the model
    learn that home's furniture/lighting/layout, weakening the "unseen home"
    claim eval mAP is meant to support. This is a WARNING, not a CRITICAL —
    training on that data is still valid, only the eval-set framing weakens.

    Args:
        captures_root: data/raw/custom_captures.
        eval_dir:      Eval set root.

    Returns:
        Report dict; ``{"available": False}`` when either side has no
        session manifests yet.
    """

    def _houses(root: Path) -> set[str]:
        manifests_dir = root / "manifests"
        if not manifests_dir.exists():
            return set()
        houses = set()
        for path in sorted(manifests_dir.glob("*.json")):
            manifest = CaptureSessionManifest.load(path)
            if manifest.house_id:
                houses.add(manifest.house_id)
        return houses

    train_houses = _houses(captures_root)
    eval_houses = _houses(eval_dir)
    if not train_houses and not eval_houses:
        return {"available": False}

    shared = sorted(train_houses & eval_houses)
    if shared:
        logger.warning(f"Houses in BOTH training captures and eval set: {shared}")

    return {
        "available": True,
        "train_houses": sorted(train_houses),
        "eval_houses": sorted(eval_houses),
        "shared_houses": shared,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full dataset QA suite (structural + governance).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--merged-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/eval/indian_home_v0"),
        help="Locked Phase-3 eval set root (checked for overlap when present).",
    )
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=Path("data/raw/custom_captures"),
        help="Custom capture root (for the house-exclusivity check).",
    )
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

    # 6–7. Phase-3 eval-set guards (opportunistic — {"available": False} pre-Phase-3)
    sources_cfg = load_sources_config(args.sources_config)
    eval_report, eval_critical = check_eval_overlap(
        args.eval_dir, args.data_dir, args.merged_dir, sources_cfg.dedup.hamming_threshold
    )
    house_report = check_house_exclusivity(args.captures_root, args.eval_dir)

    # Merge everything into the DVC metric file
    report_path = args.output / "annotation_qa_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["license_report"] = license_report
    report["label_completeness"] = completeness
    report["image_quality"] = quality_report
    report["eval_set"] = {"overlap": eval_report, "house_exclusivity": house_report}
    report["orchestrator"] = {
        "check_annotations_exit": annotations_exit,
        "dataset_stats_exit": stats_exit,
        "license_critical": license_critical,
        "image_quality_warnings": quality_warnings,
        "eval_overlap_critical": eval_critical,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Verdict
    critical = annotations_exit == 1 or license_critical or eval_critical
    warnings = (
        annotations_exit == 2
        or stats_exit != 0
        or quality_warnings > 0
        or bool(house_report.get("shared_houses"))
    )
    if license_critical:
        logger.error("LICENSE GATE VIOLATION — see license_report in the QA report")
    if eval_critical:
        logger.error("EVAL SET LEAKAGE — see eval_set.overlap in the QA report")

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

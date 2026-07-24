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


def check_image_quality(
    data_dir: Path,
    blur_threshold: float = BLUR_VARIANCE_THRESHOLD,
    low_light_threshold: float = LOW_LIGHT_BRIGHTNESS_THRESHOLD,
) -> tuple[dict[str, Any], int, dict[str, list[str]]]:
    """Blur + low-light WARNING checks over all processed images.

    Args:
        data_dir:            Dataset root with an ``images/`` subtree.
        blur_threshold:      Laplacian variance below this → blurry.
        low_light_threshold: Mean grayscale below this → too dark.

    Returns:
        ``(report dict, warning_count, quarantine)`` where ``quarantine`` holds
        the *full* flagged-image lists (``blurry`` / ``low_light``) for a
        review bucket — the report dict keeps only capped samples so the DVC
        metric file stays small. Skips gracefully without OpenCV.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — blur/low-light checks skipped")
        return {"available": False}, 0, {"blurry": [], "low_light": []}

    blurry: list[str] = []
    dark: list[str] = []
    scanned = 0

    for image_path in find_image_files(data_dir / "images"):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        scanned += 1
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < blur_threshold:
            blurry.append(image_path.name)
        if float(gray.mean()) < low_light_threshold:
            dark.append(image_path.name)

    warnings = len(blurry) + len(dark)
    logger.info(f"Image quality: {scanned} scanned, {len(blurry)} blurry, {len(dark)} low-light")
    report = {
        "available": True,
        "scanned": scanned,
        "blur_variance_threshold": blur_threshold,
        "low_light_brightness_threshold": low_light_threshold,
        "blurry_count": len(blurry),
        "blurry_samples": blurry[:MAX_LISTED_FILES],
        "low_light_count": len(dark),
        "low_light_samples": dark[:MAX_LISTED_FILES],
        "quarantine_path": "data/qa_reports/image_quality_quarantine.json",
    }
    return report, warnings, {"blurry": blurry, "low_light": dark}


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


def sweep_annotation_artifacts(
    candidates_root: Path,
    batches_root: Path,
    ledger_path: Path,
    verified_labels_dir: Path,
    merged_manifest_path: Path,
) -> dict[str, Any]:
    """M3 QA sweeps over the Phase-5 annotation artifacts.

    Findings are WARNINGs, not CRITICALs — artifact-hygiene signals for a
    human to investigate, not training-blocking corruption (that is
    preflight gate G9's job, run at train time against the artifacts
    actually consumed). Returns ``{"available": False}`` when none of the
    annotation directories exist yet (pre-M1 checkouts stay green).

    Sweeps:
        orphan_candidates         — candidate images absent from the merged
                                    manifest's provenance (stale re-run vs a
                                    rebuilt merge).
        duplicate_ledger_claims   — one image claimed by more than one
                                    ``imported`` batch (should be impossible
                                    via record_verdict's conflict guard
                                    unless batch bookkeeping itself drifted).
        unused_batches            — ``imported`` batches with zero of their
                                    images actually present in the ledger
                                    (the import silently wrote nothing).
        verified_labels_orphans   — delta label files with no corresponding
                                    ledger entry at all.
    """
    if not any(p.exists() for p in (candidates_root, batches_root, ledger_path)):
        return {"available": False}

    from src.dataset.annotation.batches import BATCH_MANIFEST_FILENAME, VerificationBatchManifest
    from src.dataset.annotation.candidates import CANDIDATES_FILENAME, load_candidates
    from src.dataset.annotation.ledger import LedgerView

    provenance: dict[str, str] = {}
    if merged_manifest_path.exists():
        provenance = json.loads(merged_manifest_path.read_text(encoding="utf-8")).get(
            "image_provenance", {}
        )

    orphan_candidates: list[str] = []
    for candidates_path in sorted(candidates_root.glob(f"*/{CANDIDATES_FILENAME}")):
        try:
            artifact = load_candidates(candidates_path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping unreadable candidates artifact {candidates_path}: {e}")
            continue
        if provenance:
            for filename in artifact.get("images", {}):
                if filename not in provenance:
                    orphan_candidates.append(f"{candidates_path.parent.name}/{filename}")

    ledger_images = LedgerView.load(ledger_path).all_images()

    batches: list[VerificationBatchManifest] = []
    for manifest_path in sorted(batches_root.glob(f"vb*_*/{BATCH_MANIFEST_FILENAME}")):
        try:
            batches.append(VerificationBatchManifest.load(manifest_path))
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping unreadable batch manifest {manifest_path}: {e}")

    claimants: dict[str, list[str]] = {}
    unused_batches: list[str] = []
    for batch in batches:
        if batch.status != "imported":
            continue
        for img in batch.images:
            claimants.setdefault(img, []).append(batch.batch_id)
        if not any(img in ledger_images for img in batch.images):
            unused_batches.append(batch.batch_id)

    duplicate_claims = [
        f"{img}: {sorted(ids)}" for img, ids in sorted(claimants.items()) if len(ids) > 1
    ]

    verified_labels_orphans: list[str] = []
    if verified_labels_dir.exists():
        ledger_stems = {Path(img).stem for img in ledger_images}
        for path in sorted(verified_labels_dir.glob("*.txt")):
            if path.stem not in ledger_stems:
                verified_labels_orphans.append(path.name)

    return {
        "available": True,
        "orphan_candidates_count": len(orphan_candidates),
        "orphan_candidates": orphan_candidates[:MAX_LISTED_FILES],
        "duplicate_ledger_claims_count": len(duplicate_claims),
        "duplicate_ledger_claims": duplicate_claims[:MAX_LISTED_FILES],
        "unused_batches_count": len(unused_batches),
        "unused_batches": unused_batches[:MAX_LISTED_FILES],
        "verified_labels_orphans_count": len(verified_labels_orphans),
        "verified_labels_orphans": verified_labels_orphans[:MAX_LISTED_FILES],
    }


def sweep_l4_l5_reports(
    coverage_report_path: Path,
    quality_report_path: Path,
    data_yaml_path: Path,
    completeness_path: Path | None = None,
) -> dict[str, Any]:
    """M4 QA sweep over the L4 coverage / L5 dataset quality reports.

    Schema-validates whichever report(s) exist and flags staleness two ways:

    * **Taxonomy fingerprint** — the report's embedded fingerprint no longer
      matching the live ``configs/data.yaml`` (a taxonomy edit).
    * **Image-count drift** — the image count baked into the report no longer
      matching the live ``data/processed/completeness.json`` image total. Both
      reports are *derived* from completeness, so a mismatch means the L4/L5
      stage was never re-run after the dataset grew/shrank. This catches the
      failure mode where a 188-image report survives a 14k-image rebuild
      because the taxonomy fingerprint happens to still match.

    WARNING level, like :func:`sweep_annotation_artifacts` — schema/staleness
    hygiene for a human to investigate, not training-blocking. Returns
    ``{"available": False}`` before either report exists (pre-M4 checkouts
    stay green).
    """
    if not coverage_report_path.exists() and not quality_report_path.exists():
        return {"available": False}

    from src.dataset.annotation.coverage import validate_coverage_report
    from src.dataset.annotation.quality import validate_quality_report
    from src.dataset.completeness import taxonomy_fingerprint
    from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

    data_cfg = load_data_config(data_yaml_path)
    names = get_class_names_from_data_yaml(data_cfg)
    live_fp = taxonomy_fingerprint(int(data_cfg["nc"]), names)

    # Live image total from the completeness artifact both reports derive from.
    live_images: int | None = None
    if completeness_path is not None and completeness_path.exists():
        try:
            comp = json.loads(completeness_path.read_text(encoding="utf-8"))
            val = comp.get("stats", {}).get("images_total")
            live_images = int(val) if isinstance(val, int) else None
        except (json.JSONDecodeError, OSError, ValueError):
            live_images = None

    problems: list[str] = []
    coverage_present = coverage_report_path.exists()
    quality_present = quality_report_path.exists()

    if coverage_present:
        coverage = json.loads(coverage_report_path.read_text(encoding="utf-8"))
        problems.extend(f"coverage_report: {p}" for p in validate_coverage_report(coverage))
        if coverage.get("taxonomy_fingerprint") != live_fp:
            problems.append(
                "coverage_report: taxonomy fingerprint stale vs live configs/data.yaml — "
                "re-run `dvc repro coverage_report`"
            )
        if live_images is not None:
            cov_images = len(coverage.get("per_image", {}))
            if cov_images != live_images:
                problems.append(
                    f"coverage_report: image count {cov_images} != live dataset "
                    f"{live_images} (completeness.json) — stale, re-run "
                    "`dvc repro coverage_report`"
                )

    if quality_present:
        quality = json.loads(quality_report_path.read_text(encoding="utf-8"))
        problems.extend(f"dataset_quality_report: {p}" for p in validate_quality_report(quality))
        if quality.get("taxonomy_fingerprint") != live_fp:
            problems.append(
                "dataset_quality_report: taxonomy fingerprint stale vs live "
                "configs/data.yaml — re-run `dvc repro dataset_quality_report`"
            )
        if live_images is not None:
            q_images = quality.get("dataset_scale", {}).get("images_total")
            if isinstance(q_images, int) and q_images != live_images:
                problems.append(
                    f"dataset_quality_report: images_total {q_images} != live dataset "
                    f"{live_images} (completeness.json) — stale, re-run "
                    "`dvc repro dataset_quality_report`"
                )

    return {
        "available": True,
        "coverage_report_present": coverage_present,
        "quality_report_present": quality_present,
        "live_images_total": live_images,
        "problems_count": len(problems),
        "problems": problems[:MAX_LISTED_FILES],
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
    parser.add_argument("--candidates-root", type=Path, default=Path("data/annotation/candidates"))
    parser.add_argument("--batches-root", type=Path, default=Path("data/annotation/batches"))
    parser.add_argument(
        "--ledger-path", type=Path, default=Path("data/annotation/verification_ledger.json")
    )
    parser.add_argument(
        "--verified-labels-dir", type=Path, default=Path("data/annotation/verified_labels")
    )
    parser.add_argument(
        "--coverage-report", type=Path, default=Path("data/qa_reports/coverage_report.json")
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("data/qa_reports/dataset_quality_report.json"),
    )
    parser.add_argument(
        "--completeness",
        type=Path,
        default=Path("data/processed/completeness.json"),
        help="Live completeness artifact — its images_total is the reference "
        "for the L4/L5 image-count staleness guard.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/qa_reports"))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as critical.")
    parser.add_argument(
        "--skip-image-checks",
        action="store_true",
        help="Skip blur/low-light scanning (faster).",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=BLUR_VARIANCE_THRESHOLD,
        help="Laplacian variance below this flags an image as blurry.",
    )
    parser.add_argument(
        "--low-light-threshold",
        type=float,
        default=LOW_LIGHT_BRIGHTNESS_THRESHOLD,
        help="Mean grayscale brightness below this flags an image as low-light.",
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
        quality_quarantine: dict[str, list[str]] = {"blurry": [], "low_light": []}
    else:
        quality_report, quality_warnings, quality_quarantine = check_image_quality(
            args.data_dir, args.blur_threshold, args.low_light_threshold
        )
        # Route the full flagged lists to a review bucket (not silent pass-through).
        quarantine_path = args.output / "image_quality_quarantine.json"
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.write_text(
            json.dumps(
                {
                    "generated_by": "scripts.qa.run_full_qa.check_image_quality",
                    "blur_variance_threshold": args.blur_threshold,
                    "low_light_brightness_threshold": args.low_light_threshold,
                    "blurry_count": len(quality_quarantine["blurry"]),
                    "low_light_count": len(quality_quarantine["low_light"]),
                    "blurry": quality_quarantine["blurry"],
                    "low_light": quality_quarantine["low_light"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    # 6–7. Phase-3 eval-set guards (opportunistic — {"available": False} pre-Phase-3)
    sources_cfg = load_sources_config(args.sources_config)
    eval_report, eval_critical = check_eval_overlap(
        args.eval_dir, args.data_dir, args.merged_dir, sources_cfg.dedup.hamming_threshold
    )
    house_report = check_house_exclusivity(args.captures_root, args.eval_dir)

    # M3: Phase-5 annotation artifact hygiene sweeps
    annotation_sweeps = sweep_annotation_artifacts(
        args.candidates_root,
        args.batches_root,
        args.ledger_path,
        args.verified_labels_dir,
        args.merged_dir / MERGED_MANIFEST_FILENAME,
    )
    sweep_warnings = (
        annotation_sweeps.get("orphan_candidates_count", 0)
        + annotation_sweeps.get("duplicate_ledger_claims_count", 0)
        + annotation_sweeps.get("unused_batches_count", 0)
        + annotation_sweeps.get("verified_labels_orphans_count", 0)
        if annotation_sweeps.get("available")
        else 0
    )

    # M4: L4/L5 report schema + staleness sweep (taxonomy + image-count drift)
    l4_l5_reports = sweep_l4_l5_reports(
        args.coverage_report, args.quality_report, args.config, args.completeness
    )
    l4_l5_warnings = l4_l5_reports.get("problems_count", 0) if l4_l5_reports.get("available") else 0

    # Merge everything into the DVC metric file
    report_path = args.output / "annotation_qa_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["license_report"] = license_report
    report["label_completeness"] = completeness
    report["image_quality"] = quality_report
    report["eval_set"] = {"overlap": eval_report, "house_exclusivity": house_report}
    report["annotation_sweeps"] = annotation_sweeps
    report["l4_l5_reports"] = l4_l5_reports
    report["orchestrator"] = {
        "check_annotations_exit": annotations_exit,
        "dataset_stats_exit": stats_exit,
        "license_critical": license_critical,
        "image_quality_warnings": quality_warnings,
        "eval_overlap_critical": eval_critical,
        "annotation_sweep_warnings": sweep_warnings,
        "l4_l5_report_warnings": l4_l5_warnings,
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
        or sweep_warnings > 0
        or l4_l5_warnings > 0
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

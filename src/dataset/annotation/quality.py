"""
src.dataset.annotation.quality — L5 Dataset Quality Report
=============================================================

Aggregates the already-computed per-dimension artifacts — completeness,
coverage (L4), the verification ledger, and the merged manifest — into one
machine-readable dataset quality report (D6/plan "L5"). Deliberately does
NOT recompute anything those artifacts already own: every number here is a
pass-through or a simple derived ratio over their outputs, so this stage
never drifts from the artifacts it reports on.

Report dimensions:
    dataset_scale         — image/instance counts by split, source, class;
                            custom-capture image/house counts (progress
                            toward the v1.0 acceptance criteria, RG9).
    completeness_summary  — masking extent (how much of the taxonomy is
                            trusted per image, on average).
    coverage_summary      — residual missing-annotation risk (pass-through
                            of coverage_report.json's dataset-level stats).
    per_class_risk        — per-class coverage_report.json rows, flattened
                            here so the quality report is self-contained.
    verification_progress — ledger cell counts + verification batch
                            lifecycle counts + measured IAA (extended at M8
                            with prioritized-batch throughput tracking).

Data flow:
    data/processed/completeness.json
        + data/qa_reports/coverage_report.json
        + data/merged/merged_manifest.json
        + data/annotation/verification_ledger.json
        + data/annotation/batches/
    → build_quality_report() → dataset_quality_report.json

Failure philosophy: taxonomy fingerprint drift between the completeness or
coverage artifact and the live ``configs/data.yaml`` is a hard error — this
report is read for release-gate decisions (RG3) and must never silently
summarize stale data.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import BATCH_MANIFEST_FILENAME, VerificationBatchManifest
from src.dataset.annotation.ledger import LedgerView
from src.dataset.completeness import load_completeness, taxonomy_fingerprint
from src.dataset.manifest import CaptureSessionManifest, MergedManifest
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.report_utils import timestamp_str

logger = logging.getLogger(__name__)

QUALITY_SCHEMA_VERSION = 1
GENERATOR_SCRIPT = "scripts/dataset/17_dataset_quality_report.py"

REQUIRED_DIMENSIONS: tuple[str, ...] = (
    "dataset_scale",
    "completeness_summary",
    "coverage_summary",
    "per_class_risk",
    "verification_progress",
)


def _count_houses(capture_manifests_dir: Path | None) -> int:
    """Distinct pseudonymous house_ids among custom-capture session manifests."""
    if capture_manifests_dir is None or not capture_manifests_dir.exists():
        return 0
    houses: set[str] = set()
    for path in sorted(capture_manifests_dir.glob("*.json")):
        manifest = CaptureSessionManifest.load(path)
        if manifest.house_id:
            houses.add(manifest.house_id)
    return len(houses)


def _build_dataset_scale(
    completeness: dict[str, Any],
    merged: MergedManifest,
    capture_manifests_dir: Path | None,
    per_class_risk: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stats = completeness.get("stats", {})
    images = completeness.get("images", {})
    images_by_source = Counter(
        merged.image_provenance[name] for name in images if name in merged.image_provenance
    )
    custom_accepted = next(
        (s.get("accepted", 0) for s in merged.sources if s.get("source") == "custom_captures"),
        0,
    )
    return {
        "images_total": stats.get("images_total", len(images)),
        "images_by_split": stats.get("by_split", {}),
        "images_by_source": dict(sorted(images_by_source.items())),
        "instances_per_class": {
            name: entry.get("annotated_instances", 0) for name, entry in per_class_risk.items()
        },
        "custom_images_total": custom_accepted,
        "houses_total": _count_houses(capture_manifests_dir),
    }


def _batch_cells_verified(batch: VerificationBatchManifest, ledger_view: LedgerView) -> int:
    """Count of this batch's (image, class) cells now settled in the ledger.

    Best-effort attribution: the ledger records only the MOST RECENT
    batch_id per image (``ledger.py``'s ``record_verdict``), not per class,
    so a cell re-verified by a later batch on the same image is attributed
    to that later batch here too — scoped to ``batch.images`` x
    ``batch.target_classes`` (the only cells this batch could plausibly have
    touched), which avoids crediting a batch with classes it never targeted.
    """
    target = frozenset(batch.target_classes)
    if not target:
        return 0
    return sum(len(ledger_view.verified_class_names(image) & target) for image in batch.images)


def _build_verification_progress(ledger_view: LedgerView, batches_root: Path) -> dict[str, Any]:
    ledger_stats = dict(ledger_view.raw.get("stats", {}))

    by_status: Counter[str] = Counter()
    measured_iaa: list[float] = []
    throughput: list[dict[str, Any]] = []
    if batches_root.exists():
        for path in sorted(batches_root.glob(f"vb*_*/{BATCH_MANIFEST_FILENAME}")):
            try:
                batch = VerificationBatchManifest.load(path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping unreadable batch manifest {path}: {e}")
                continue
            by_status[batch.status] += 1
            if batch.iaa_agreement >= 0.0:
                measured_iaa.append(batch.iaa_agreement)
            throughput.append(
                {
                    "batch_id": batch.batch_id,
                    "status": batch.status,
                    "images_count": len(batch.images),
                    "expected_gain": batch.expected_gain,
                    "cells_verified": _batch_cells_verified(batch, ledger_view),
                }
            )

    imported_cells = [row["cells_verified"] for row in throughput if row["status"] == "imported"]

    return {
        "ledger_stats": ledger_stats,
        "batches_by_status": dict(sorted(by_status.items())),
        "batches_total": sum(by_status.values()),
        "mean_iaa_agreement": (round(statistics.fmean(measured_iaa), 4) if measured_iaa else None),
        "batches_with_measured_iaa": len(measured_iaa),
        # M8 (ADR-P5-02/plan §M8): per-batch predicted (expected_gain) vs.
        # achieved (cells_verified) throughput — prerequisite evidence for
        # any future expected-gain weight tuning, not itself a tuning
        # decision (no data exists yet to justify changing the D1 weights).
        "batch_throughput": throughput,
        "mean_cells_verified_per_imported_batch": (
            round(statistics.fmean(imported_cells), 4) if imported_cells else None
        ),
    }


def build_quality_report(
    completeness_path: Path,
    coverage_report_path: Path,
    merged_manifest_path: Path,
    ledger_path: Path,
    batches_root: Path,
    data_yaml_path: Path,
    capture_manifests_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the L5 dataset quality report.

    Args:
        completeness_path:    ``data/processed/completeness.json``.
        coverage_report_path: ``data/qa_reports/coverage_report.json``.
        merged_manifest_path: ``data/merged/merged_manifest.json``.
        ledger_path:          ``data/annotation/verification_ledger.json``.
        batches_root:         ``data/annotation/batches``.
        data_yaml_path:       ``configs/data.yaml`` (live taxonomy).
        capture_manifests_dir: ``data/raw/custom_captures/manifests``, or
                               ``None`` (pre-Phase-3-data checkouts).

    Returns:
        The report dict (see module docstring for dimensions).

    Raises:
        AnnotationError: On taxonomy fingerprint drift against the
                         completeness or coverage artifact.
        FileNotFoundError: If a required input file is missing.
    """
    data_cfg = load_data_config(data_yaml_path)
    names = get_class_names_from_data_yaml(data_cfg)
    nc = int(data_cfg["nc"])
    live_fp = taxonomy_fingerprint(nc, names)

    completeness = load_completeness(completeness_path)
    completeness_fp = completeness.get("taxonomy", {}).get("fingerprint")
    if completeness_fp != live_fp:
        raise AnnotationError(
            f"Completeness artifact taxonomy fingerprint {completeness_fp!r} != live "
            f"{live_fp!r} — re-run `dvc repro generate_completeness` before "
            f"dataset_quality_report."
        )

    coverage = json.loads(coverage_report_path.read_text(encoding="utf-8"))
    coverage_fp = coverage.get("taxonomy_fingerprint")
    if coverage_fp != live_fp:
        raise AnnotationError(
            f"Coverage report taxonomy fingerprint {coverage_fp!r} != live {live_fp!r} — "
            f"re-run `dvc repro coverage_report` before dataset_quality_report."
        )

    merged = MergedManifest.load(merged_manifest_path)
    ledger_view = LedgerView.load(ledger_path)

    per_class_risk: dict[str, dict[str, Any]] = dict(coverage.get("per_class", {}))
    dataset_scale = _build_dataset_scale(
        completeness, merged, capture_manifests_dir, per_class_risk
    )

    completeness_stats = completeness.get("stats", {})
    mean_trusted = float(completeness_stats.get("mean_trusted_classes_per_image", 0.0))
    completeness_summary = {
        "policies_count": len(completeness.get("policies", {})),
        "mean_trusted_classes_per_image": mean_trusted,
        "masked_cell_fraction": round(1.0 - (mean_trusted / nc), 4) if nc else 0.0,
    }

    coverage_summary = {
        **coverage.get("dataset", {}),
        **coverage.get("per_image_summary", {}),
    }

    verification_progress = _build_verification_progress(ledger_view, batches_root)

    report: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "generated_at": timestamp_str(),
        "generator": {"script": GENERATOR_SCRIPT},
        "taxonomy_fingerprint": live_fp,
        "dataset_scale": dataset_scale,
        "completeness_summary": completeness_summary,
        "coverage_summary": coverage_summary,
        "per_class_risk": per_class_risk,
        "verification_progress": verification_progress,
    }
    logger.info(
        f"Quality report built: {dataset_scale['images_total']} images, "
        f"{len(per_class_risk)} classes, "
        f"masked_cell_fraction={completeness_summary['masked_cell_fraction']}"
    )
    return report


def validate_quality_report(report: dict[str, Any]) -> list[str]:
    """Self-consistency validation. Returns problems (empty = valid)."""
    problems: list[str] = []
    for key in ("schema_version", *REQUIRED_DIMENSIONS):
        if key not in report:
            problems.append(f"missing required dimension '{key}'")
    if problems:
        return problems

    fraction = report["completeness_summary"].get("masked_cell_fraction")
    if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) <= 1.0:
        problems.append(f"completeness_summary.masked_cell_fraction {fraction!r} outside [0, 1]")

    scale = report["dataset_scale"]
    if not isinstance(scale.get("images_total"), int) or scale["images_total"] < 0:
        problems.append(f"dataset_scale.images_total invalid: {scale.get('images_total')!r}")

    return problems

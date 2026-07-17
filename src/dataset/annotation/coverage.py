"""
src.dataset.annotation.coverage — L4 Coverage Estimation (ADR-P5-06)
======================================================================

Pure-arithmetic residual missing-annotation risk estimate over artifacts
that are ALL already pinned by the time this stage runs — the auto-
annotation candidates, the human verification ledger, and the trained
labels. No model inference happens here (ADR-P5-06): every confidence
decision was already made by the auto-annotation backend when the
candidates artifact was generated.

Core idea: a candidate detection sits in one of three states for its
(image, class) cell:
    - trusted:   the image's completeness policy already trusts this class
                 exhaustively — the cell is fully labeled by construction,
                 the candidate is redundant.
    - verified:  a human settled this cell in the ledger (either verdict) —
                 already accounted for.
    - unknown:   neither of the above — a real, uncertain signal that the
                 class may be present but unlabeled.

"Unknown" candidates are discounted by a per-class **estimator precision**
calibrated from the ledger's own verified cells: a ``verified_absent``
verdict over a cell that had candidates is a real false-positive sample for
that class, and a ``present_labeled`` verdict lets candidates be IoU-matched
against the human boxes for true/false positives — free calibration data
with no held-out set required. Classes with no verified cells yet fall back
to ``coverage.estimation_conf`` (configs/annotation.yaml).

Data flow:
    data/annotation/candidates/<backend>/candidates.json  (one or more)
        + data/annotation/verification_ledger.json
        + data/processed/completeness.json (per-image trusted-class policy)
        + data/processed/labels/{train,val,test} (what actually trains)
    → build_coverage_report() → coverage_report.json (+csv/md)

Failure philosophy: taxonomy fingerprint drift between any input and the
live ``configs/data.yaml`` is a hard error (:class:`AnnotationError`) — a
stale estimate is worse than no estimate.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.candidates import CANDIDATES_FILENAME, load_candidates
from src.dataset.annotation.ledger import LedgerView
from src.dataset.completeness import load_completeness, taxonomy_fingerprint
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import compute_file_hash

logger = logging.getLogger(__name__)

COVERAGE_SCHEMA_VERSION = 1
DEFAULT_ESTIMATION_PRECISION = 0.35


def _to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Normalized (x_center, y_center, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = box
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def iou_xywhn(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two normalized xywhn boxes."""
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_candidates_to_verified(
    candidate_boxes: list[tuple[float, float, float, float]],
    human_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy IoU matching of candidate boxes against human-verified boxes.

    Args:
        candidate_boxes: This (image, class) cell's candidate detections.
        human_boxes:     Ledger-recorded boxes for the same cell (empty for
                         a ``verified_absent`` verdict — every candidate is
                         then a false positive, which the loop below already
                         produces without a special case).
        iou_threshold:   ``coverage.iou_match_threshold``.

    Returns:
        ``(true_positives, false_positives, false_negatives)``.
    """
    matched_human: set[int] = set()
    tp = 0
    for cand in candidate_boxes:
        best_iou, best_h = 0.0, -1
        for h_idx, h_box in enumerate(human_boxes):
            if h_idx in matched_human:
                continue
            iou = iou_xywhn(cand, h_box)
            if iou > best_iou:
                best_iou, best_h = iou, h_idx
        if best_iou >= iou_threshold:
            matched_human.add(best_h)
            tp += 1
    fp = len(candidate_boxes) - tp
    fn = len(human_boxes) - len(matched_human)
    return tp, fp, fn


def _load_processed_label_counts(
    processed_labels_root: Path,
    images: Mapping[str, Mapping[str, Any]],
    names: Mapping[int, str],
) -> tuple[dict[str, int], Counter[str]]:
    """Per-image total box count + per-class instance count over processed labels.

    Args:
        processed_labels_root: ``data/processed/labels``.
        images:                completeness artifact's ``images`` block
                               (filename -> {policy, split}).
        names:                Taxonomy id -> name.

    Returns:
        (annotated_total_by_image, annotated_instances_by_class).
    """
    annotated_total_by_image: dict[str, int] = {}
    annotated_instances_by_class: Counter[str] = Counter()
    for filename, entry in images.items():
        split = str(entry.get("split", ""))
        label_path = processed_labels_root / split / f"{Path(filename).stem}.txt"
        total = 0
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                total += 1
                class_id = int(parts[0])
                class_name = names.get(class_id)
                if class_name is not None:
                    annotated_instances_by_class[class_name] += 1
        annotated_total_by_image[filename] = total
    return annotated_total_by_image, annotated_instances_by_class


def _precision_for(
    class_name: str,
    calibration: Mapping[str, Mapping[str, Any]],
    estimation_conf: Mapping[str, float],
) -> float:
    """Calibrated precision for a class, falling back to the config estimate."""
    entry = calibration.get(class_name)
    if entry is not None and entry.get("estimator_precision") is not None:
        return float(entry["estimator_precision"])
    return float(
        estimation_conf.get(
            class_name, estimation_conf.get("default", DEFAULT_ESTIMATION_PRECISION)
        )
    )


def build_coverage_report(
    candidates_root: Path,
    ledger_path: Path,
    completeness_path: Path,
    processed_labels_root: Path,
    data_yaml_path: Path,
    iou_match_threshold: float,
    estimation_conf: Mapping[str, float],
) -> dict[str, Any]:
    """Build the L4 coverage report (pure arithmetic; zero inference).

    Args:
        candidates_root:       ``data/annotation/candidates`` (one subdir per
                               backend, each holding ``candidates.json``).
        ledger_path:           ``data/annotation/verification_ledger.json``.
        completeness_path:     ``data/processed/completeness.json``.
        processed_labels_root: ``data/processed/labels``.
        data_yaml_path:        ``configs/data.yaml`` (live taxonomy).
        iou_match_threshold:   ``coverage.iou_match_threshold``.
        estimation_conf:       ``coverage.estimation_conf`` (class -> prior
                               precision; must contain ``"default"``).

    Returns:
        The report dict (see module docstring for the schema).

    Raises:
        AnnotationError: On taxonomy fingerprint drift between the live
                         taxonomy and the completeness artifact or any
                         candidates artifact.
        FileNotFoundError: If ``completeness_path`` does not exist.
    """
    data_cfg = load_data_config(data_yaml_path)
    names = get_class_names_from_data_yaml(data_cfg)
    nc = int(data_cfg["nc"])
    ids_by_name = {name: cid for cid, name in names.items()}
    live_fp = taxonomy_fingerprint(nc, names)

    completeness = load_completeness(completeness_path)
    completeness_fp = completeness.get("taxonomy", {}).get("fingerprint")
    if completeness_fp != live_fp:
        raise AnnotationError(
            f"Completeness artifact taxonomy fingerprint {completeness_fp!r} != live "
            f"{live_fp!r} — re-run `dvc repro generate_completeness` before coverage_report."
        )
    images: Mapping[str, Mapping[str, Any]] = completeness.get("images", {})
    policies: Mapping[str, Mapping[str, Any]] = completeness.get("policies", {})
    trusted_ids_by_image: dict[str, frozenset[int]] = {
        filename: frozenset(
            policies.get(str(entry.get("policy", "")), {}).get("trusted_class_ids", [])
        )
        for filename, entry in images.items()
    }

    ledger_view = LedgerView.load(ledger_path)
    ledger_entries: Mapping[str, Mapping[str, Any]] = ledger_view.raw.get("entries", {})
    recorded_fp = ledger_view.taxonomy_fingerprint()
    if recorded_fp and recorded_fp != live_fp:
        raise AnnotationError(
            f"Ledger taxonomy fingerprint drift: ledger {recorded_fp!r} vs live {live_fp!r} — "
            f"reconcile before generating the coverage report."
        )

    candidates_records: list[dict[str, Any]] = []
    candidates_by_image: dict[str, dict[int, list[tuple[float, float, float, float]]]] = (
        defaultdict(dict)
    )
    for path in sorted(candidates_root.glob(f"*/{CANDIDATES_FILENAME}")):
        artifact = load_candidates(path)
        artifact_fp = artifact.get("taxonomy_fingerprint")
        if artifact_fp != live_fp:
            raise AnnotationError(
                f"Candidates artifact {path} taxonomy fingerprint {artifact_fp!r} != live "
                f"{live_fp!r} — regenerate via `dvc repro auto_annotate`."
            )
        candidates_records.append(
            {
                "backend": artifact.get("backend", path.parent.name),
                "run_id": artifact.get("run_id", ""),
                "sha256": compute_file_hash(path),
            }
        )
        for filename, entry in artifact.get("images", {}).items():
            class_boxes = candidates_by_image[filename]
            for det in entry.get("detections", []):
                class_id = int(det["class_id"])
                class_boxes.setdefault(class_id, []).append(tuple(det["bbox_xywhn"]))

    # ── Calibration: TP/FP/FN over verified ledger cells ────────────────────
    calibration_counts: dict[str, dict[str, int]] = {}
    for filename, entry in ledger_entries.items():
        for class_name, verdict in entry.get("classes", {}).items():
            verdict_class_id = ids_by_name.get(class_name)
            cand_boxes = (
                candidates_by_image.get(filename, {}).get(verdict_class_id, [])
                if verdict_class_id is not None
                else []
            )
            human_boxes = [tuple(b) for b in verdict.get("boxes", [])]
            tp, fp, fn = match_candidates_to_verified(cand_boxes, human_boxes, iou_match_threshold)
            counts = calibration_counts.setdefault(
                class_name, {"tp": 0, "fp": 0, "fn": 0, "verified_cells": 0}
            )
            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn
            counts["verified_cells"] += 1

    calibration: dict[str, dict[str, Any]] = {}
    for class_name, counts in sorted(calibration_counts.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
        calibration[class_name] = {
            "verified_cells": counts["verified_cells"],
            "estimator_precision": precision,
            "estimator_recall_proxy": recall,
        }

    # ── Unverified ("unknown") candidates + per-image estimated residual ───
    unverified_by_class: Counter[str] = Counter()
    estimated_by_image: dict[str, float] = defaultdict(float)
    for filename, class_boxes in candidates_by_image.items():
        trusted = trusted_ids_by_image.get(filename)
        if trusted is None:
            continue  # not a processed image (filtered out post-merge) — out of scope
        verified_here = ledger_view.verified_class_names(filename)
        for class_id, boxes in class_boxes.items():
            class_name = names.get(class_id)
            if class_name is None or class_id in trusted or class_name in verified_here:
                continue
            unverified_by_class[class_name] += len(boxes)
            precision_used = _precision_for(class_name, calibration, estimation_conf)
            estimated_by_image[filename] += len(boxes) * precision_used

    # ── Ledger verdict tallies (all cells, trusted or not) ──────────────────
    verified_present_by_class: Counter[str] = Counter()
    verified_absent_by_class: Counter[str] = Counter()
    for entry in ledger_entries.values():
        for class_name, verdict in entry.get("classes", {}).items():
            if verdict.get("status") == "present_labeled":
                verified_present_by_class[class_name] += 1
            else:
                verified_absent_by_class[class_name] += 1

    annotated_total_by_image, annotated_instances_by_class = _load_processed_label_counts(
        processed_labels_root, images, names
    )

    per_class: dict[str, dict[str, Any]] = {}
    for cid in sorted(names):
        class_name = names[cid]
        annotated = annotated_instances_by_class.get(class_name, 0)
        unverified = unverified_by_class.get(class_name, 0)
        precision_used = _precision_for(class_name, calibration, estimation_conf)
        residual = round(unverified * precision_used, 4)
        denom = annotated + residual
        coverage_score = 1.0 if denom <= 0 else round(annotated / denom, 4)
        per_class[class_name] = {
            "annotated_instances": annotated,
            "unverified_candidates": unverified,
            "verified_present": verified_present_by_class.get(class_name, 0),
            "verified_absent": verified_absent_by_class.get(class_name, 0),
            "residual_missing_estimate": residual,
            "coverage_score": coverage_score,
        }

    per_image: dict[str, dict[str, Any]] = {}
    completeness_values: list[float] = []
    for filename in sorted(images):
        annotated = annotated_total_by_image.get(filename, 0)
        estimated = round(estimated_by_image.get(filename, 0.0), 4)
        denom = annotated + estimated
        image_completeness = 1.0 if denom <= 0 else round(annotated / denom, 4)
        per_image[filename] = {
            "annotated": annotated,
            "estimated": estimated,
            "completeness": image_completeness,
        }
        completeness_values.append(image_completeness)

    if completeness_values:
        mean_completeness = round(statistics.fmean(completeness_values), 4)
        sorted_vals = sorted(completeness_values)
        p10_index = max(0, min(len(sorted_vals) - 1, int(0.1 * len(sorted_vals))))
        p10_completeness = round(sorted_vals[p10_index], 4)
        images_below_0_5 = sum(1 for v in completeness_values if v < 0.5)
    else:
        mean_completeness = 1.0
        p10_completeness = 1.0
        images_below_0_5 = 0

    residual_missing_total = round(
        sum(p["residual_missing_estimate"] for p in per_class.values()), 4
    )
    unknown_objects_total = sum(unverified_by_class.values())

    report: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "candidates": candidates_records,
        "taxonomy_fingerprint": live_fp,
        "method": {
            "iou_match_threshold": iou_match_threshold,
            "estimation_conf": dict(estimation_conf),
        },
        "calibration": calibration,
        "per_class": per_class,
        "per_image_summary": {
            "mean_completeness": mean_completeness,
            "p10_completeness": p10_completeness,
            "images_below_0_5": images_below_0_5,
        },
        "per_image": per_image,
        "dataset": {
            "residual_missing_total": residual_missing_total,
            "unknown_objects_total": unknown_objects_total,
        },
    }
    logger.info(
        f"Coverage report built: {len(per_class)} classes, {len(per_image)} images, "
        f"residual_missing_total={residual_missing_total}, "
        f"unknown_objects_total={unknown_objects_total}"
    )
    return report


def validate_coverage_report(report: Mapping[str, Any]) -> list[str]:
    """Self-consistency validation. Returns problems (empty = valid)."""
    problems: list[str] = []
    for key in ("schema_version", "per_class", "per_image", "per_image_summary", "dataset"):
        if key not in report:
            problems.append(f"missing required key '{key}'")
    if problems:
        return problems

    per_class = report["per_class"]
    if not isinstance(per_class, Mapping):
        problems.append("'per_class' must be an object")
    else:
        for name, entry in per_class.items():
            score = entry.get("coverage_score")
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                problems.append(f"per_class[{name!r}].coverage_score {score!r} outside [0, 1]")

    per_image = report["per_image"]
    if not isinstance(per_image, Mapping):
        problems.append("'per_image' must be an object")
    else:
        for name, entry in per_image.items():
            score = entry.get("completeness")
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                problems.append(f"per_image[{name!r}].completeness {score!r} outside [0, 1]")

    computed_total = (
        round(sum(e.get("residual_missing_estimate", 0.0) for e in per_class.values()), 4)
        if isinstance(per_class, Mapping)
        else 0.0
    )
    recorded_total = report["dataset"].get("residual_missing_total")
    if (
        isinstance(recorded_total, (int, float))
        and abs(float(recorded_total) - computed_total) > 1e-3
    ):
        problems.append(
            f"dataset.residual_missing_total {recorded_total} != sum of per-class estimates "
            f"{computed_total}"
        )
    return problems

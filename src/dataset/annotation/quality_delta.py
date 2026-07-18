"""
src.dataset.annotation.quality_delta — Quality-Report Delta (M8)
====================================================================

Compares two ``dataset_quality_report.json`` snapshots (e.g. the
dataset-v0.5.0 release vs. the current in-progress build) and computes a
per-class residual-risk / coverage-score delta. This is the artifact
behind M8's plan acceptance criterion: "coverage report shows per-
priority-class residual-risk drop vs v0.5 (delta table in changelog)".

Pure aggregation over two already-built reports (ADR-P5-06 spirit extended
to release comparisons) — recomputes nothing, never re-reads candidates or
the ledger.
"""

from __future__ import annotations

from typing import Any


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 4)


def build_quality_delta(
    baseline: dict[str, Any],
    current: dict[str, Any],
    priority_classes: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Per-class residual-missing-estimate / coverage-score delta.

    Args:
        baseline:          An earlier ``dataset_quality_report.json`` (e.g.
                           the v0.5.0 release's copy).
        current:           A later one (e.g. the current build).
        priority_classes:  Restricts ``priority_class_delta`` to these
                           class names; ``None`` includes every class.
                           ``per_class_delta`` always covers every class
                           present in either snapshot.

    Returns:
        ``{"baseline_generated_at", "current_generated_at",
        "per_class_delta": [...], "priority_class_delta": [...]}`` — each
        delta row's ``*_delta`` fields are negative when residual risk
        dropped (improvement) and positive when coverage score rose
        (also improvement); ``None`` when either side lacks the class.
    """
    baseline_risk: dict[str, dict[str, Any]] = baseline.get("per_class_risk", {})
    current_risk: dict[str, dict[str, Any]] = current.get("per_class_risk", {})
    classes = sorted(set(baseline_risk) | set(current_risk))

    rows: list[dict[str, Any]] = []
    for class_name in classes:
        base = baseline_risk.get(class_name, {})
        curr = current_risk.get(class_name, {})
        base_residual = base.get("residual_missing_estimate")
        curr_residual = curr.get("residual_missing_estimate")
        base_score = base.get("coverage_score")
        curr_score = curr.get("coverage_score")
        rows.append(
            {
                "class": class_name,
                "baseline_residual_missing_estimate": base_residual,
                "current_residual_missing_estimate": curr_residual,
                "residual_missing_delta": _delta(curr_residual, base_residual),
                "baseline_coverage_score": base_score,
                "current_coverage_score": curr_score,
                "coverage_score_delta": _delta(curr_score, base_score),
            }
        )

    priority_rows = (
        rows
        if priority_classes is None
        else [row for row in rows if row["class"] in priority_classes]
    )

    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "current_generated_at": current.get("generated_at"),
        "per_class_delta": rows,
        "priority_class_delta": priority_rows,
    }

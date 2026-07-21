"""Unit tests for the M8 quality-report delta (src.dataset.annotation.quality_delta)."""

from __future__ import annotations

import pytest

from src.dataset.annotation.quality_delta import build_quality_delta

pytestmark = pytest.mark.unit


def _report(generated_at: str, per_class_risk: dict) -> dict:
    return {"generated_at": generated_at, "per_class_risk": per_class_risk}


class TestBuildQualityDelta:
    def test_residual_drop_and_coverage_gain_are_negative_and_positive(self) -> None:
        baseline = _report(
            "2026-01-01T00:00:00Z",
            {"charger": {"residual_missing_estimate": 10.0, "coverage_score": 0.5}},
        )
        current = _report(
            "2026-02-01T00:00:00Z",
            {"charger": {"residual_missing_estimate": 4.0, "coverage_score": 0.8}},
        )
        delta = build_quality_delta(baseline, current)
        row = delta["per_class_delta"][0]
        assert row["residual_missing_delta"] == -6.0
        assert row["coverage_score_delta"] == pytest.approx(0.3)
        assert delta["baseline_generated_at"] == "2026-01-01T00:00:00Z"
        assert delta["current_generated_at"] == "2026-02-01T00:00:00Z"

    def test_class_only_in_current_has_none_baseline_and_none_delta(self) -> None:
        baseline = _report("2026-01-01T00:00:00Z", {})
        current = _report(
            "2026-02-01T00:00:00Z",
            {"wire": {"residual_missing_estimate": 2.0, "coverage_score": 0.9}},
        )
        delta = build_quality_delta(baseline, current)
        row = delta["per_class_delta"][0]
        assert row["class"] == "wire"
        assert row["baseline_residual_missing_estimate"] is None
        assert row["residual_missing_delta"] is None
        assert row["coverage_score_delta"] is None

    def test_priority_classes_filters_priority_class_delta_only(self) -> None:
        baseline = _report(
            "2026-01-01T00:00:00Z",
            {
                "charger": {"residual_missing_estimate": 10.0, "coverage_score": 0.5},
                "person": {"residual_missing_estimate": 0.0, "coverage_score": 1.0},
            },
        )
        current = _report(
            "2026-02-01T00:00:00Z",
            {
                "charger": {"residual_missing_estimate": 5.0, "coverage_score": 0.7},
                "person": {"residual_missing_estimate": 0.0, "coverage_score": 1.0},
            },
        )
        delta = build_quality_delta(baseline, current, frozenset({"charger"}))
        assert [row["class"] for row in delta["priority_class_delta"]] == ["charger"]
        # per_class_delta is never filtered — priority_classes only scopes
        # priority_class_delta.
        assert {row["class"] for row in delta["per_class_delta"]} == {"charger", "person"}

    def test_no_priority_classes_arg_includes_every_class(self) -> None:
        baseline = _report("2026-01-01T00:00:00Z", {"charger": {"residual_missing_estimate": 1.0}})
        current = _report("2026-02-01T00:00:00Z", {"charger": {"residual_missing_estimate": 1.0}})
        delta = build_quality_delta(baseline, current)
        assert delta["priority_class_delta"] == delta["per_class_delta"]

    def test_zero_delta_when_reports_identical(self) -> None:
        report = _report(
            "2026-01-01T00:00:00Z",
            {"charger": {"residual_missing_estimate": 3.0, "coverage_score": 0.6}},
        )
        delta = build_quality_delta(report, report)
        row = delta["per_class_delta"][0]
        assert row["residual_missing_delta"] == 0.0
        assert row["coverage_score_delta"] == 0.0

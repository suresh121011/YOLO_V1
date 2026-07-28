"""Unit tests for the before/after dataset comparison report.

The report is release evidence, so the failure mode that matters is not a
crash — it is a section that renders successfully while saying nothing.
The first draft read ``images_total`` from the root of
``completeness_report.json`` when the counts live under ``stats``, and printed
"images covered: None" in a report that otherwise looked complete. These tests
pin the artifact shapes rather than only the happy path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "qa" / "build_comparison_report.py"
_spec = importlib.util.spec_from_file_location("build_comparison_report", _MODULE_PATH)
assert _spec and _spec.loader
build_comparison_report = importlib.util.module_from_spec(_spec)
sys.modules["build_comparison_report"] = build_comparison_report
_spec.loader.exec_module(build_comparison_report)

completeness_of = build_comparison_report.completeness_of
source_split = build_comparison_report.source_split
shortfall = build_comparison_report.shortfall
qa_of = build_comparison_report.qa_of
class_counts_from_manifest = build_comparison_report.class_counts_from_manifest


class TestCompletenessOf:
    def test_reads_counts_from_the_stats_block(self) -> None:
        """The regression: counts are nested under `stats`, not at the root."""
        report = {
            "artifact_path": "data/processed/completeness.json",
            "stats": {
                "images_total": 17888,
                "by_split": {"train": 14323, "val": 1773, "test": 1792},
                "by_policy": {"coco": 53, "local_captures": 17698},
            },
            "policies": [{"policy": "coco"}, {"policy": "local_captures"}],
            "unused_policies": [],
        }
        result = completeness_of(report)
        assert result["images_total"] == 17888
        assert result["by_policy"]["local_captures"] == 17698
        assert result["policy_count"] == 2

    def test_missing_report_is_empty_not_none_fields(self) -> None:
        assert completeness_of(None) == {}

    def test_absent_stats_block_does_not_raise(self) -> None:
        result = completeness_of({"policies": []})
        assert result["images_total"] is None
        assert result["by_policy"] == {}


class TestSourceSplit:
    def test_public_and_local_are_partitioned(self) -> None:
        manifest = {
            "sources": [
                {"source": "coco", "accepted": 53},
                {"source": "wider_face", "accepted": 60},
                {"source": "local_captures", "accepted": 17698},
            ]
        }
        result = source_split(manifest)
        assert result["public_total"] == 113
        assert result["local_total"] == 17698
        assert "local_captures" not in result["public"]

    def test_unknown_source_counts_as_local(self) -> None:
        """A new capture channel must not be silently counted as public.

        `custom_captures` and `local_captures` are both locally collected; any
        source not on the public download list is treated the same way, so
        adding one cannot inflate the public share by default.
        """
        result = source_split({"sources": [{"source": "custom_captures", "accepted": 42}]})
        assert result["local_total"] == 42
        assert result["public_total"] == 0

    def test_missing_manifest_is_empty(self) -> None:
        assert source_split(None) == {}


class TestShortfall:
    TAXONOMY = ["person", "stove", "charger", "wet_floor"]
    CUSTOM = ("stove", "wet_floor")

    def test_lists_only_classes_below_the_floor_worst_first(self) -> None:
        counts = {"person": 5000, "stove": 267, "charger": 110, "wet_floor": 18057}
        rows = shortfall(counts, self.TAXONOMY, 200, self.CUSTOM)
        assert [r["class"] for r in rows] == ["charger"]
        assert rows[0]["short_by"] == 90

    def test_absent_class_counts_as_zero(self) -> None:
        """A class with no boxes at all must appear, not vanish."""
        rows = shortfall({}, self.TAXONOMY, 200, self.CUSTOM)
        assert {r["class"] for r in rows} == set(self.TAXONOMY)
        assert all(r["count"] == 0 for r in rows)

    def test_custom_capture_classes_are_flagged(self) -> None:
        rows = shortfall({"stove": 10, "charger": 10}, self.TAXONOMY, 200, self.CUSTOM)
        flags = {r["class"]: r["needs_custom_capture"] for r in rows}
        assert flags["stove"] is True, "stove shortfall needs Indian-home capture"
        assert flags["charger"] is False, "charger is coverable from public sources"

    def test_zero_floor_reports_nothing(self) -> None:
        """A track with no min_instances_per_class must not flag every class."""
        assert shortfall({"person": 0}, ["person"], 0, ()) == []


class TestQaOf:
    def test_pulls_leakage_from_the_checks_block(self) -> None:
        report = {
            "summary": {"total_images": 100, "total_boxes": 500, "critical_issues": 0},
            "checks": {"train_val_leakage": {"count": 0}, "train_test_leakage": {"count": 3}},
            "orchestrator": {"license_critical": False, "l4_l5_report_warnings": 2},
        }
        result = qa_of(report)
        assert result["train_test_leakage"] == 3
        assert result["l4_l5_report_warnings"] == 2
        assert result["license_critical"] is False

    def test_missing_report_is_empty(self) -> None:
        assert qa_of(None) == {}


class TestClassCounts:
    def test_reads_class_counts_from_manifest(self) -> None:
        assert class_counts_from_manifest({"class_counts": {"person": 35106}}) == {"person": 35106}

    def test_missing_manifest_is_empty(self) -> None:
        assert class_counts_from_manifest(None) == {}

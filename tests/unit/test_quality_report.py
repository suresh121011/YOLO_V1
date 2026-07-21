"""Unit tests for the L5 dataset quality report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.ledger import new_ledger, recompute_stats, record_verdict, save_ledger
from src.dataset.annotation.quality import (
    REQUIRED_DIMENSIONS,
    build_quality_report,
    validate_quality_report,
)
from src.dataset.completeness import taxonomy_fingerprint

pytestmark = pytest.mark.unit

NAMES = {0: "charger", 1: "wire", 2: "person"}
NC = 3
LIVE_FP = taxonomy_fingerprint(NC, NAMES)


def _write_data_yaml(path: Path) -> Path:
    path.write_text(
        json.dumps({"nc": NC, "names": {str(k): v for k, v in NAMES.items()}}),
        encoding="utf-8",
    )
    return path


def _write_completeness(
    path: Path,
    images: dict[str, dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    mean_trusted: float,
    fingerprint: str = LIVE_FP,
) -> Path:
    artifact = {
        "schema_version": 1,
        "taxonomy": {
            "nc": NC,
            "names": {str(k): v for k, v in NAMES.items()},
            "fingerprint": fingerprint,
        },
        "policies": policies,
        "images": images,
        "stats": {
            "images_total": len(images),
            "by_split": {"train": len(images)},
            "mean_trusted_classes_per_image": mean_trusted,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _write_coverage(path: Path, fingerprint: str = LIVE_FP) -> Path:
    report = {
        "schema_version": 1,
        "taxonomy_fingerprint": fingerprint,
        "per_class": {
            "charger": {
                "annotated_instances": 2,
                "unverified_candidates": 1,
                "verified_present": 0,
                "verified_absent": 0,
                "residual_missing_estimate": 0.35,
                "coverage_score": 0.851,
            },
            "wire": {
                "annotated_instances": 0,
                "unverified_candidates": 0,
                "verified_present": 0,
                "verified_absent": 0,
                "residual_missing_estimate": 0.0,
                "coverage_score": 1.0,
            },
            "person": {
                "annotated_instances": 5,
                "unverified_candidates": 0,
                "verified_present": 0,
                "verified_absent": 0,
                "residual_missing_estimate": 0.0,
                "coverage_score": 1.0,
            },
        },
        "per_image_summary": {
            "mean_completeness": 0.9,
            "p10_completeness": 0.5,
            "images_below_0_5": 0,
        },
        "dataset": {"residual_missing_total": 0.35, "unknown_objects_total": 1},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_merged_manifest(
    path: Path, provenance: dict[str, str], sources: list[dict[str, Any]]
) -> Path:
    manifest = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "sources": sources,
        "image_provenance": provenance,
        "duplicates_removed": 0,
        "filtered_out": 0,
        "class_counts": {},
        "label_completeness": {},
        "notes": "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class TestBuildQualityReport:
    def test_all_required_dimensions_present(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
            mean_trusted=1.0,
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco"},
            sources=[{"source": "coco", "total": 1, "accepted": 1}],
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=tmp_path / "coverage_report.json",
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        for dim in REQUIRED_DIMENSIONS:
            assert dim in report
        assert validate_quality_report(report) == []

    def test_dataset_scale_reflects_source_and_custom_counts(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={
                "a.jpg": {"policy": "coco", "split": "train"},
                "b.jpg": {"policy": "custom", "split": "val"},
            },
            policies={
                "coco": {"mode": "trusted_list", "trusted_class_ids": [0]},
                "custom": {"mode": "per_session", "trusted_class_ids": [0, 1, 2]},
            },
            mean_trusted=2.0,
        )
        cov_path = tmp_path / "coverage_report.json"
        _write_coverage(cov_path)
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco", "b.jpg": "custom_captures"},
            sources=[
                {"source": "coco", "total": 1, "accepted": 1},
                {"source": "custom_captures", "total": 1, "accepted": 1},
            ],
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=cov_path,
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        scale = report["dataset_scale"]
        assert scale["images_total"] == 2
        assert scale["images_by_source"] == {"coco": 1, "custom_captures": 1}
        assert scale["custom_images_total"] == 1
        assert scale["instances_per_class"]["charger"] == 2

    def test_masked_cell_fraction_derived_from_mean_trusted(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
            mean_trusted=1.0,  # 1 of 3 classes trusted
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco"},
            sources=[{"source": "coco", "total": 1, "accepted": 1}],
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=tmp_path / "coverage_report.json",
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        assert report["completeness_summary"]["masked_cell_fraction"] == pytest.approx(
            2.0 / 3.0, abs=1e-4
        )

    def test_verification_progress_counts_ledger_and_batches(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
            mean_trusted=1.0,
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco"},
            sources=[{"source": "coco", "total": 1, "accepted": 1}],
        )

        ledger = new_ledger()
        record_verdict(
            ledger,
            filename="a.jpg",
            source="coco",
            class_name="charger",
            status="present_labeled",
            boxes=[(0.5, 0.5, 0.2, 0.2)],
            batch_id="vb001_yolo_world",
            verifier="tester",
            method="cvat",
            cvat_task_ref="task-1",
        )
        recompute_stats(ledger, LIVE_FP)
        save_ledger(ledger, tmp_path / "ledger.json")

        batch_dir = tmp_path / "batches" / "vb001_yolo_world"
        batch_dir.mkdir(parents=True)
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "batch_id": "vb001_yolo_world",
                    "status": "imported",
                    "iaa_agreement": 0.9,
                }
            ),
            encoding="utf-8",
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=tmp_path / "coverage_report.json",
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        progress = report["verification_progress"]
        assert progress["ledger_stats"]["cells_verified"] == 1
        assert progress["batches_by_status"] == {"imported": 1}
        assert progress["mean_iaa_agreement"] == pytest.approx(0.9)

    def test_batch_throughput_counts_only_target_class_cells_actually_verified(
        self, tmp_path: Path
    ) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
            mean_trusted=1.0,
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco"},
            sources=[{"source": "coco", "total": 1, "accepted": 1}],
        )

        ledger = new_ledger()
        # Only "charger" gets a verdict — "wire" (also targeted by the batch
        # below) stays unverified, so throughput must count 1, not 2.
        record_verdict(
            ledger,
            filename="a.jpg",
            source="coco",
            class_name="charger",
            status="present_labeled",
            boxes=[(0.5, 0.5, 0.2, 0.2)],
            batch_id="vb001_yolo_world",
            verifier="tester",
            method="cvat",
            cvat_task_ref="task-1",
        )
        recompute_stats(ledger, LIVE_FP)
        save_ledger(ledger, tmp_path / "ledger.json")

        batch_dir = tmp_path / "batches" / "vb001_yolo_world"
        batch_dir.mkdir(parents=True)
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "batch_id": "vb001_yolo_world",
                    "status": "imported",
                    "images": ["a.jpg"],
                    "target_classes": ["charger", "wire"],
                    "expected_gain": 1.5,
                }
            ),
            encoding="utf-8",
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=tmp_path / "coverage_report.json",
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        progress = report["verification_progress"]
        assert progress["batch_throughput"] == [
            {
                "batch_id": "vb001_yolo_world",
                "status": "imported",
                "images_count": 1,
                "expected_gain": 1.5,
                "cells_verified": 1,
            }
        ]
        assert progress["mean_cells_verified_per_imported_batch"] == pytest.approx(1.0)

    def test_batch_throughput_empty_target_classes_counts_zero(self, tmp_path: Path) -> None:
        """Legacy/minimal batch manifests (no images/target_classes) never crash."""
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
            mean_trusted=1.0,
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(
            tmp_path / "merged_manifest.json",
            provenance={"a.jpg": "coco"},
            sources=[{"source": "coco", "total": 1, "accepted": 1}],
        )

        batch_dir = tmp_path / "batches" / "vb001_yolo_world"
        batch_dir.mkdir(parents=True)
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps({"schema_version": 1, "batch_id": "vb001_yolo_world", "status": "created"}),
            encoding="utf-8",
        )

        report = build_quality_report(
            completeness_path=tmp_path / "completeness.json",
            coverage_report_path=tmp_path / "coverage_report.json",
            merged_manifest_path=tmp_path / "merged_manifest.json",
            ledger_path=tmp_path / "ledger.json",
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )

        row = report["verification_progress"]["batch_throughput"][0]
        assert row["cells_verified"] == 0
        # "created" (not "imported") is excluded from the throughput mean.
        assert report["verification_progress"]["mean_cells_verified_per_imported_batch"] is None

    def test_taxonomy_drift_in_completeness_raises(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={},
            policies={},
            mean_trusted=0.0,
            fingerprint="sha256:stale",
        )
        _write_coverage(tmp_path / "coverage_report.json")
        _write_merged_manifest(tmp_path / "merged_manifest.json", provenance={}, sources=[])

        with pytest.raises(AnnotationError, match="fingerprint"):
            build_quality_report(
                completeness_path=tmp_path / "completeness.json",
                coverage_report_path=tmp_path / "coverage_report.json",
                merged_manifest_path=tmp_path / "merged_manifest.json",
                ledger_path=tmp_path / "ledger.json",
                batches_root=tmp_path / "batches",
                data_yaml_path=data_yaml,
            )

    def test_taxonomy_drift_in_coverage_report_raises(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={},
            policies={},
            mean_trusted=0.0,
        )
        _write_coverage(tmp_path / "coverage_report.json", fingerprint="sha256:stale")
        _write_merged_manifest(tmp_path / "merged_manifest.json", provenance={}, sources=[])

        with pytest.raises(AnnotationError, match="fingerprint"):
            build_quality_report(
                completeness_path=tmp_path / "completeness.json",
                coverage_report_path=tmp_path / "coverage_report.json",
                merged_manifest_path=tmp_path / "merged_manifest.json",
                ledger_path=tmp_path / "ledger.json",
                batches_root=tmp_path / "batches",
                data_yaml_path=data_yaml,
            )


class TestValidateQualityReport:
    def test_missing_dimension_reported(self) -> None:
        problems = validate_quality_report({"schema_version": 1})
        assert any("missing required dimension" in p for p in problems)

    def test_out_of_range_masked_fraction_reported(self) -> None:
        report = {
            "schema_version": 1,
            "dataset_scale": {"images_total": 1},
            "completeness_summary": {"masked_cell_fraction": 1.5},
            "coverage_summary": {},
            "per_class_risk": {},
            "verification_progress": {},
        }
        problems = validate_quality_report(report)
        assert any("masked_cell_fraction" in p for p in problems)

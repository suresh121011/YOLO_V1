"""Unit tests for the L4 coverage estimation report (ADR-P5-06)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.coverage import (
    build_coverage_report,
    iou_xywhn,
    match_candidates_to_verified,
    validate_coverage_report,
)
from src.dataset.annotation.ledger import new_ledger, record_verdict, save_ledger
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
        "stats": {"images_total": len(images)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _write_candidates(
    path: Path,
    backend: str,
    images: dict[str, list[dict[str, Any]]],
    fingerprint: str = LIVE_FP,
) -> Path:
    artifact = {
        "schema_version": 1,
        "run_id": f"{backend}_test_run",
        "backend": backend,
        "taxonomy_fingerprint": fingerprint,
        "images": {
            name: {"targeted_class_ids": [], "detections": dets} for name, dets in images.items()
        },
        "stats": {
            "images_processed": len(images),
            "detections_total": sum(len(d) for d in images.values()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _det(class_id: int, conf: float, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "conf": conf,
        "bbox_xywhn": list(bbox),
        "refined": False,
        "origin": "test",
    }


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class TestIouXywhn:
    def test_identical_boxes_iou_one(self) -> None:
        box = (0.5, 0.5, 0.2, 0.2)
        assert iou_xywhn(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self) -> None:
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        assert iou_xywhn(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = (0.5, 0.5, 0.2, 0.2)
        b = (0.55, 0.5, 0.2, 0.2)
        iou = iou_xywhn(a, b)
        assert 0.0 < iou < 1.0


class TestMatchCandidatesToVerified:
    def test_matched_pair_is_true_positive(self) -> None:
        cand = [(0.5, 0.5, 0.2, 0.2)]
        human = [(0.5, 0.5, 0.2, 0.2)]
        tp, fp, fn = match_candidates_to_verified(cand, human, 0.5)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_unmatched_candidate_is_false_positive(self) -> None:
        cand = [(0.1, 0.1, 0.1, 0.1)]
        human: list[tuple[float, float, float, float]] = []
        tp, fp, fn = match_candidates_to_verified(cand, human, 0.5)
        assert (tp, fp, fn) == (0, 1, 0)

    def test_unmatched_human_box_is_false_negative(self) -> None:
        cand: list[tuple[float, float, float, float]] = []
        human = [(0.5, 0.5, 0.2, 0.2)]
        tp, fp, fn = match_candidates_to_verified(cand, human, 0.5)
        assert (tp, fp, fn) == (0, 0, 1)


class TestBuildCoverageReport:
    def test_trusted_cell_candidate_is_not_unknown(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
        )
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={"a.jpg": [_det(0, 0.9, (0.5, 0.5, 0.2, 0.2))]},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", [])

        report = build_coverage_report(
            candidates_root=tmp_path / "candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )

        assert report["per_class"]["charger"]["unverified_candidates"] == 0
        assert report["dataset"]["unknown_objects_total"] == 0

    def test_untrusted_unverified_candidate_is_discounted_by_default_prior(
        self, tmp_path: Path
    ) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": []}},
        )
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={"a.jpg": [_det(0, 0.9, (0.5, 0.5, 0.2, 0.2))]},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", [])

        report = build_coverage_report(
            candidates_root=tmp_path / "candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.4},
        )

        assert report["per_class"]["charger"]["unverified_candidates"] == 1
        assert report["per_class"]["charger"]["residual_missing_estimate"] == pytest.approx(0.4)
        assert report["dataset"]["unknown_objects_total"] == 1
        assert report["dataset"]["residual_missing_total"] == pytest.approx(0.4)

    def test_verified_present_matched_candidate_calibrates_precision_one(
        self, tmp_path: Path
    ) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": []}},
        )
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={"a.jpg": [_det(0, 0.9, (0.5, 0.5, 0.2, 0.2))]},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", ["0 0.5 0.5 0.2 0.2"])

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
        save_ledger(ledger, tmp_path / "ledger.json")

        report = build_coverage_report(
            candidates_root=tmp_path / "candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )

        calib = report["calibration"]["charger"]
        assert calib["estimator_precision"] == pytest.approx(1.0)
        assert calib["estimator_recall_proxy"] == pytest.approx(1.0)
        # Verified cell -> excluded from "unknown"
        assert report["per_class"]["charger"]["unverified_candidates"] == 0
        assert report["per_class"]["charger"]["verified_present"] == 1
        assert report["per_class"]["charger"]["annotated_instances"] == 1
        assert report["per_class"]["charger"]["coverage_score"] == pytest.approx(1.0)

    def test_verified_absent_candidate_is_false_positive_calibration(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": []}},
        )
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={"a.jpg": [_det(0, 0.9, (0.5, 0.5, 0.2, 0.2))]},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", [])

        ledger = new_ledger()
        record_verdict(
            ledger,
            filename="a.jpg",
            source="coco",
            class_name="charger",
            status="verified_absent",
            boxes=[],
            batch_id="vb001_yolo_world",
            verifier="tester",
            method="cvat",
            cvat_task_ref="task-1",
        )
        save_ledger(ledger, tmp_path / "ledger.json")

        report = build_coverage_report(
            candidates_root=tmp_path / "candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )

        calib = report["calibration"]["charger"]
        assert calib["estimator_precision"] == pytest.approx(0.0)
        assert report["per_class"]["charger"]["verified_absent"] == 1
        assert report["per_class"]["charger"]["unverified_candidates"] == 0

    def test_taxonomy_drift_in_completeness_raises(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={},
            policies={},
            fingerprint="sha256:stale",
        )
        (tmp_path / "candidates").mkdir()

        with pytest.raises(AnnotationError, match="fingerprint"):
            build_coverage_report(
                candidates_root=tmp_path / "candidates",
                ledger_path=tmp_path / "ledger.json",
                completeness_path=tmp_path / "completeness.json",
                processed_labels_root=tmp_path / "labels",
                data_yaml_path=data_yaml,
                iou_match_threshold=0.5,
                estimation_conf={"default": 0.35},
            )

    def test_taxonomy_drift_in_candidates_artifact_raises(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(tmp_path / "completeness.json", images={}, policies={})
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={},
            fingerprint="sha256:stale",
        )

        with pytest.raises(AnnotationError, match="fingerprint"):
            build_coverage_report(
                candidates_root=tmp_path / "candidates",
                ledger_path=tmp_path / "ledger.json",
                completeness_path=tmp_path / "completeness.json",
                processed_labels_root=tmp_path / "labels",
                data_yaml_path=data_yaml,
                iou_match_threshold=0.5,
                estimation_conf={"default": 0.35},
            )

    def test_no_candidates_directory_produces_empty_but_valid_report(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", ["0 0.5 0.5 0.2 0.2"])

        report = build_coverage_report(
            candidates_root=tmp_path / "no_candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )

        assert report["candidates"] == []
        assert report["dataset"]["unknown_objects_total"] == 0
        assert report["per_class"]["charger"]["annotated_instances"] == 1
        assert not validate_coverage_report(report)

    def test_per_image_summary_stats(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={
                "a.jpg": {"policy": "coco", "split": "train"},
                "b.jpg": {"policy": "coco", "split": "train"},
            },
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": []}},
        )
        _write_candidates(
            tmp_path / "candidates" / "yolo_world" / "candidates.json",
            "yolo_world",
            images={"b.jpg": [_det(0, 0.9, (0.5, 0.5, 0.2, 0.2))]},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", ["0 0.5 0.5 0.2 0.2"])
        _write_label(tmp_path / "labels" / "train" / "b.txt", [])

        report = build_coverage_report(
            candidates_root=tmp_path / "candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.5},
        )

        assert report["per_image"]["a.jpg"]["completeness"] == pytest.approx(1.0)
        assert report["per_image"]["b.jpg"]["completeness"] == pytest.approx(0.0)
        assert report["per_image_summary"]["images_below_0_5"] == 1


class TestValidateCoverageReport:
    def test_valid_report_has_no_problems(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        _write_completeness(
            tmp_path / "completeness.json",
            images={"a.jpg": {"policy": "coco", "split": "train"}},
            policies={"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
        )
        _write_label(tmp_path / "labels" / "train" / "a.txt", [])
        report = build_coverage_report(
            candidates_root=tmp_path / "no_candidates",
            ledger_path=tmp_path / "ledger.json",
            completeness_path=tmp_path / "completeness.json",
            processed_labels_root=tmp_path / "labels",
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )
        assert validate_coverage_report(report) == []

    def test_missing_key_is_reported(self) -> None:
        problems = validate_coverage_report({"schema_version": 1})
        assert any("missing required key" in p for p in problems)

    def test_inconsistent_dataset_total_is_reported(self) -> None:
        report = {
            "schema_version": 1,
            "per_class": {"charger": {"coverage_score": 1.0, "residual_missing_estimate": 5.0}},
            "per_image": {},
            "per_image_summary": {},
            "dataset": {"residual_missing_total": 0.0, "unknown_objects_total": 0},
        }
        problems = validate_coverage_report(report)
        assert any("residual_missing_total" in p for p in problems)

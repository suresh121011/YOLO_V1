"""Unit tests for src.training.evaluation — report shaping with mocked results."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.training.evaluation import (
    EvalRunSpec,
    build_delta_report,
    export_confusion_matrix,
    extract_aggregate_metrics,
    extract_per_class_metrics,
    f1_score,
    wet_floor_ap50_checkpoint,
)

_NAMES = {0: "person", 1: "face", 2: "knife"}


def fake_results(
    class_rows: dict[int, tuple[float, float, float, float]],
    mp: float = 0.5,
    mr: float = 0.4,
    map50: float = 0.45,
    map5095: float = 0.3,
):
    """Build an object mimicking Ultralytics DetMetrics for our extractors.

    Args:
        class_rows: class_id → (precision, recall, ap50, ap50_95); only these
                    ids appear in ap_class_index (classes present in GT).
    """
    index = list(class_rows)

    def class_result(position: int) -> tuple[float, float, float, float]:
        return class_rows[index[position]]

    box = SimpleNamespace(
        ap_class_index=index,
        class_result=class_result,
        mp=mp,
        mr=mr,
        map50=map50,
        map=map5095,
    )
    matrix = [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    return SimpleNamespace(box=box, confusion_matrix=SimpleNamespace(matrix=matrix))


@pytest.mark.unit
class TestMetricExtraction:
    """Per-class and aggregate extraction from results objects."""

    def test_f1_score(self) -> None:
        assert f1_score(0.0, 0.0) == 0.0
        assert f1_score(1.0, 1.0) == 1.0
        assert f1_score(0.5, 0.5) == 0.5

    def test_per_class_rows_sorted_and_named(self) -> None:
        results = fake_results({2: (0.6, 0.5, 0.55, 0.35), 0: (0.8, 0.7, 0.75, 0.5)})
        rows = extract_per_class_metrics(results, _NAMES)
        assert [row["class"] for row in rows] == ["person", "knife"]
        assert rows[0] == {
            "class_id": 0,
            "class": "person",
            "precision": 0.8,
            "recall": 0.7,
            "f1": round(f1_score(0.8, 0.7), 4),
            "mAP50": 0.75,
            "mAP50_95": 0.5,
        }

    def test_absent_classes_produce_no_rows(self) -> None:
        results = fake_results({1: (0.4, 0.3, 0.35, 0.2)})
        rows = extract_per_class_metrics(results, _NAMES)
        assert [row["class"] for row in rows] == ["face"]

    def test_aggregate_metrics_include_f1(self) -> None:
        aggregate = extract_aggregate_metrics(fake_results({}, mp=0.6, mr=0.3))
        assert aggregate["precision"] == 0.6
        assert aggregate["recall"] == 0.3
        assert aggregate["f1"] == round(f1_score(0.6, 0.3), 4)


@pytest.mark.unit
class TestConfusionExport:
    """Confusion matrix JSON export."""

    def test_export_writes_labels_and_matrix(self, tmp_path: Path) -> None:
        results = fake_results({0: (0.5, 0.5, 0.5, 0.5)})
        out = export_confusion_matrix(results, _NAMES, tmp_path / "cm.json")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["labels"] == ["person", "face", "knife", "background"]
        assert payload["matrix"][0][3] == 2.0
        assert len(payload["matrix"]) == 4


@pytest.mark.unit
class TestDeltaReport:
    """Mitigated-minus-baseline delta computation."""

    @staticmethod
    def _summary(per_class: list[dict], **aggregate: float) -> dict:
        base = {"precision": 0.5, "recall": 0.5, "f1": 0.5, "mAP50": 0.5, "mAP50_95": 0.5}
        base.update(aggregate)
        return {"aggregate": base, "per_class": per_class}

    def test_aggregate_delta(self) -> None:
        baseline = self._summary([], mAP50=0.40)
        mitigated = self._summary([], mAP50=0.46)
        delta = build_delta_report(baseline, mitigated)
        assert delta["aggregate_delta"]["mAP50"] == 0.06
        assert delta["aggregate_delta"]["precision"] == 0.0

    def test_per_class_delta_matches_by_name(self) -> None:
        row = {
            "class_id": 1,
            "class": "face",
            "precision": 0.3,
            "recall": 0.2,
            "f1": 0.24,
            "mAP50": 0.25,
            "mAP50_95": 0.1,
        }
        improved = {**row, "recall": 0.5, "mAP50": 0.4}
        delta = build_delta_report(self._summary([row]), self._summary([improved]))
        face = delta["per_class_delta"][0]
        assert face["class"] == "face"
        assert face["delta_recall"] == 0.3
        assert face["delta_mAP50"] == 0.15

    def test_class_missing_in_one_arm_yields_none_deltas(self) -> None:
        row = {
            "class_id": 2,
            "class": "knife",
            "precision": 0.4,
            "recall": 0.4,
            "f1": 0.4,
            "mAP50": 0.4,
            "mAP50_95": 0.2,
        }
        delta = build_delta_report(self._summary([row]), self._summary([]))
        knife = delta["per_class_delta"][0]
        assert knife["baseline_mAP50"] == 0.4
        assert knife["mitigated_mAP50"] is None
        assert knife["delta_mAP50"] is None


@pytest.mark.unit
class TestEvalRunSpec:
    """Spec defaults."""

    def test_defaults(self) -> None:
        spec = EvalRunSpec(weights=Path("best.pt"), label="baseline")
        assert spec.split == "val"
        assert spec.device == "cpu"
        assert spec.imgsz == 640
        assert spec.seed == 42


@pytest.mark.unit
class TestWetFloorAp50Checkpoint:
    """R24 checkpoint 2 (docs/04 capture_annotation_runbook.md §8)."""

    def test_below_threshold_reopens_demotion(self) -> None:
        per_class = [{"class": "wet_floor", "mAP50": 0.22}]
        result = wet_floor_ap50_checkpoint(per_class)
        assert result["available"] is True
        assert result["ap50"] == 0.22
        assert result["reopen_demotion"] is True

    def test_at_or_above_threshold_does_not_reopen(self) -> None:
        per_class = [{"class": "wet_floor", "mAP50": 0.30}]
        result = wet_floor_ap50_checkpoint(per_class)
        assert result["reopen_demotion"] is False

    def test_wet_floor_absent_from_ground_truth_is_unavailable(self) -> None:
        per_class = [{"class": "charger", "mAP50": 0.6}]
        result = wet_floor_ap50_checkpoint(per_class)
        assert result["available"] is False
        assert result["ap50"] is None
        assert result["reopen_demotion"] is False

    def test_custom_threshold_respected(self) -> None:
        per_class = [{"class": "wet_floor", "mAP50": 0.35}]
        result = wet_floor_ap50_checkpoint(per_class, ap50_threshold=0.40)
        assert result["reopen_demotion"] is True

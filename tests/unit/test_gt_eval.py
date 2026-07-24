"""Unit tests for the annotation-quality GT eval (P9) — scoring + label loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.gt_eval import (
    ClassScore,
    score_predictions_against_gt,
)

pytestmark = pytest.mark.unit

_NAMES = {0: "person", 1: "charger", 2: "wire"}
_BOX = (0.5, 0.5, 0.2, 0.2)


class TestClassScore:
    def test_precision_recall_f1_mean_iou(self) -> None:
        s = ClassScore(0, "person", tp=6, fp=2, fn=2, matched_iou_sum=5.4)
        assert s.precision == 0.75  # 6/8
        assert s.recall == 0.75  # 6/8
        assert s.f1 == 0.75
        assert s.mean_iou == pytest.approx(0.9)  # 5.4/6

    def test_zero_denominators(self) -> None:
        s = ClassScore(0, "person")
        assert s.precision == 0.0
        assert s.recall == 0.0
        assert s.f1 == 0.0
        assert s.mean_iou == 0.0


class TestScorePredictionsAgainstGt:
    def test_perfect_match(self) -> None:
        gt = {"img1": {0: [_BOX]}}
        pred = {"img1": {0: [_BOX]}}
        report = score_predictions_against_gt(pred, gt, _NAMES, iou_threshold=0.5)
        c = report.per_class[0]
        assert (c.tp, c.fp, c.fn) == (1, 0, 0)
        assert c.precision == 1.0 and c.recall == 1.0
        assert c.mean_iou == pytest.approx(1.0)

    def test_false_positive_lowers_precision(self) -> None:
        gt = {"img1": {0: [_BOX]}}
        pred = {"img1": {0: [_BOX, (0.1, 0.1, 0.05, 0.05)]}}  # 2nd box matches no GT
        report = score_predictions_against_gt(pred, gt, _NAMES)
        c = report.per_class[0]
        assert (c.tp, c.fp, c.fn) == (1, 1, 0)
        assert c.precision == 0.5
        assert c.recall == 1.0

    def test_false_negative_lowers_recall(self) -> None:
        gt = {"img1": {0: [_BOX, (0.2, 0.2, 0.1, 0.1)]}}
        pred = {"img1": {0: [_BOX]}}  # misses the 2nd GT box
        report = score_predictions_against_gt(pred, gt, _NAMES)
        c = report.per_class[0]
        assert (c.tp, c.fp, c.fn) == (1, 0, 1)
        assert c.precision == 1.0
        assert c.recall == 0.5

    def test_below_iou_threshold_is_fp_and_fn(self) -> None:
        gt = {"img1": {0: [(0.5, 0.5, 0.2, 0.2)]}}
        pred = {"img1": {0: [(0.5, 0.5, 0.05, 0.05)]}}  # tiny concentric → IoU ~0.0625
        report = score_predictions_against_gt(pred, gt, _NAMES, iou_threshold=0.5)
        c = report.per_class[0]
        assert (c.tp, c.fp, c.fn) == (0, 1, 1)

    def test_per_class_isolation(self) -> None:
        gt = {"img1": {0: [_BOX], 1: [_BOX]}}
        pred = {"img1": {0: [_BOX]}}  # right box but nothing for class 1
        report = score_predictions_against_gt(pred, gt, _NAMES)
        assert (report.per_class[0].tp, report.per_class[0].fn) == (1, 0)
        assert (report.per_class[1].tp, report.per_class[1].fn) == (0, 1)

    def test_all_taxonomy_classes_present_even_zero_support(self) -> None:
        report = score_predictions_against_gt({}, {}, _NAMES)
        assert set(report.per_class) == {0, 1, 2}
        assert report.images_scored == 0

    def test_micro_and_macro_aggregates(self) -> None:
        # class 0: tp2 fp0 fn0 (P=R=1). class 1: tp1 fp1 fn0 (P=0.5,R=1).
        gt = {"a": {0: [_BOX], 1: [_BOX]}, "b": {0: [_BOX]}}
        pred = {
            "a": {0: [_BOX], 1: [_BOX, (0.1, 0.1, 0.05, 0.05)]},
            "b": {0: [_BOX]},
        }
        report = score_predictions_against_gt(pred, gt, _NAMES)
        micro = report.micro()  # tp3 fp1 fn0 → P=0.75 R=1.0
        assert micro["precision"] == 0.75
        assert micro["recall"] == 1.0
        macro = report.macro()  # mean over classes 0,1 (class2 no support): P=(1+0.5)/2
        assert macro["precision"] == 0.75
        assert macro["recall"] == 1.0

    def test_out_of_taxonomy_class_skipped(self) -> None:
        gt = {"img1": {99: [_BOX]}}  # 99 not in _NAMES
        pred = {"img1": {99: [_BOX]}}
        report = score_predictions_against_gt(pred, gt, _NAMES)
        assert 99 not in report.per_class
        # nothing crashed; taxonomy classes stay zeroed
        assert all(c.tp == 0 for c in report.per_class.values())

    def test_report_as_dict_shape(self) -> None:
        gt = {"img1": {0: [_BOX]}}
        report = score_predictions_against_gt({"img1": {0: [_BOX]}}, gt, _NAMES)
        d = report.as_dict()
        assert d["schema_version"] == 1
        assert d["iou_threshold"] == 0.5
        assert {"precision", "recall", "f1"} <= set(d["micro"])
        assert len(d["per_class"]) == 3
        assert d["per_class"][0]["class_name"] == "person"


class TestLoadYoloLabels:
    def test_parses_and_groups_by_class(self, tmp_path: Path) -> None:
        from scripts.qa.annotation_gt_eval import load_yolo_labels

        (tmp_path / "img1.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.2 0.2\n"
        )
        (tmp_path / "img2.txt").write_text("")  # empty (background) → no boxes
        loaded = load_yolo_labels(tmp_path)
        assert set(loaded["img1"]) == {0, 1}
        assert len(loaded["img1"][0]) == 2
        assert loaded["img2"] == {}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        from scripts.qa.annotation_gt_eval import load_yolo_labels

        (tmp_path / "img1.txt").write_text(
            "# comment\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5\nx y z a b\n1 0.1 0.1 0.1 0.1 0.1\n"
        )
        loaded = load_yolo_labels(tmp_path)
        # only the one valid 5-field numeric line survives
        assert loaded["img1"] == {0: [(0.5, 0.5, 0.2, 0.2)]}

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        from scripts.qa.annotation_gt_eval import load_yolo_labels

        with pytest.raises(FileNotFoundError):
            load_yolo_labels(tmp_path / "nope")

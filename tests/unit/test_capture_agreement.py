"""Unit tests for src.dataset.capture.agreement — dual-annotator agreement."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.capture.agreement import (
    agreement_verdict,
    compare_annotators,
    compute_iou,
    load_staged_labels,
    report_as_dict,
)
from src.dataset.capture.config import IaaSettings
from src.utils.annotation_utils import Annotation

_CLASS_NAMES = {0: "person", 1: "stove", 20: "wet_floor"}


def _ann(class_id: int, cx: float, cy: float, w: float = 0.2, h: float = 0.2) -> Annotation:
    return Annotation(class_id=class_id, cx=cx, cy=cy, w=w, h=h, line_num=0, raw="")


@pytest.mark.unit
class TestComputeIou:
    """IoU math on normalized cx/cy/w/h boxes."""

    def test_identical_boxes(self) -> None:
        assert compute_iou((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert compute_iou((0.2, 0.2, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1)) == 0.0

    def test_touching_boxes_are_zero(self) -> None:
        # Share only an edge → zero intersection area.
        assert compute_iou((0.3, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == 0.0

    def test_half_overlap(self) -> None:
        # Boxes offset by half a width: intersection 0.1×0.2, union 0.06.
        iou = compute_iou((0.5, 0.5, 0.2, 0.2), (0.6, 0.5, 0.2, 0.2))
        assert iou == pytest.approx((0.1 * 0.2) / (0.04 + 0.04 - 0.02))

    def test_contained_box(self) -> None:
        iou = compute_iou((0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.2, 0.2))
        assert iou == pytest.approx(0.04 / 0.16)


@pytest.mark.unit
class TestCompareAnnotators:
    """Greedy matching and agreement accounting."""

    def test_perfect_agreement(self) -> None:
        labels = {"img1": [_ann(0, 0.5, 0.5), _ann(1, 0.2, 0.2)]}
        report = compare_annotators(labels, dict(labels), 0.5, _CLASS_NAMES)
        assert report.overall_agreement == 1.0
        assert report.per_class["person"].matched == 1
        assert report.per_class["stove"].matched == 1
        assert report.per_image["img1"] == 1.0

    def test_slightly_shifted_boxes_still_match(self) -> None:
        labels_a = {"img1": [_ann(0, 0.50, 0.50)]}
        labels_b = {"img1": [_ann(0, 0.52, 0.50)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        assert report.per_class["person"].matched == 1
        assert 0.5 < report.per_class["person"].mean_iou < 1.0

    def test_missing_box_counts_against_agreement(self) -> None:
        labels_a = {"img1": [_ann(0, 0.5, 0.5), _ann(0, 0.2, 0.2)]}
        labels_b = {"img1": [_ann(0, 0.5, 0.5)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        stats = report.per_class["person"]
        assert stats.matched == 1
        assert stats.only_a == 1
        assert stats.only_b == 0
        assert stats.agreement == pytest.approx(0.5)

    def test_class_swap_is_double_disagreement(self) -> None:
        # Same box, different class → only_a for one class, only_b for the other.
        labels_a = {"img1": [_ann(0, 0.5, 0.5)]}
        labels_b = {"img1": [_ann(1, 0.5, 0.5)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        assert report.per_class["person"].only_a == 1
        assert report.per_class["stove"].only_b == 1
        assert report.overall_agreement == 0.0

    def test_image_only_seen_by_one_annotator(self) -> None:
        labels_a = {"img1": [_ann(0, 0.5, 0.5)], "img2": [_ann(0, 0.5, 0.5)]}
        labels_b = {"img1": [_ann(0, 0.5, 0.5)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        assert report.images_compared == 2
        assert report.per_image["img2"] == 0.0

    def test_worst_images_ranking(self) -> None:
        labels_a = {
            "good": [_ann(0, 0.5, 0.5)],
            "bad": [_ann(0, 0.2, 0.2)],
        }
        labels_b = {
            "good": [_ann(0, 0.5, 0.5)],
            "bad": [_ann(0, 0.8, 0.8)],
        }
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        worst = report.worst_images(1)
        assert worst == [("bad", 0.0)]

    def test_empty_labels_are_full_agreement(self) -> None:
        report = compare_annotators({"img1": []}, {"img1": []}, 0.5, _CLASS_NAMES)
        assert report.overall_agreement == 1.0
        assert report.per_image["img1"] == 1.0


@pytest.mark.unit
class TestAgreementVerdict:
    """Per-class gates with the wet_floor R24 override."""

    def test_pass(self) -> None:
        labels = {"img1": [_ann(1, 0.5, 0.5)]}
        report = compare_annotators(labels, dict(labels), 0.5, _CLASS_NAMES)
        verdict, failures = agreement_verdict(
            report, IaaSettings(min_agreement=0.75, wet_floor_min_agreement=0.6)
        )
        assert verdict == "pass"
        assert failures == []

    def test_fail_below_default_gate(self) -> None:
        labels_a = {"img1": [_ann(1, 0.5, 0.5), _ann(1, 0.2, 0.2)]}
        labels_b = {"img1": [_ann(1, 0.5, 0.5)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        verdict, failures = agreement_verdict(report, IaaSettings(min_agreement=0.75))
        assert verdict == "fail"
        assert failures and "stove" in failures[0]

    def test_wet_floor_uses_lower_gate(self) -> None:
        # Agreement 2/3 ≈ 0.67: fails the 0.75 default but passes wet_floor's 0.6.
        labels_a = {"img1": [_ann(20, 0.5, 0.5), _ann(20, 0.2, 0.2), _ann(20, 0.8, 0.8)]}
        labels_b = {"img1": [_ann(20, 0.5, 0.5), _ann(20, 0.2, 0.2)]}
        report = compare_annotators(labels_a, labels_b, 0.5, _CLASS_NAMES)
        assert report.per_class["wet_floor"].agreement == pytest.approx(2 / 3)

        verdict, _ = agreement_verdict(
            report, IaaSettings(min_agreement=0.75, wet_floor_min_agreement=0.6)
        )
        assert verdict == "pass"

        verdict, failures = agreement_verdict(
            report, IaaSettings(min_agreement=0.75, wet_floor_min_agreement=0.7)
        )
        assert verdict == "fail"
        assert "wet_floor" in failures[0]


@pytest.mark.unit
class TestStagedLabelsAndReport:
    """Staged-label loading and JSON payload."""

    def test_load_staged_labels(self, tmp_path: Path) -> None:
        staged = tmp_path / "staging" / "h01_kitchen_s001" / "asha"
        staged.mkdir(parents=True)
        (staged / "h01_kitchen_s001_0001.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        labels = load_staged_labels(tmp_path / "staging", "h01_kitchen_s001", "asha")
        assert list(labels) == ["h01_kitchen_s001_0001"]
        assert labels["h01_kitchen_s001_0001"][0].class_id == 1

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_staged_labels(tmp_path, "h01_kitchen_s001", "asha")

    def test_report_as_dict_shape(self) -> None:
        labels = {"img1": [_ann(0, 0.5, 0.5)]}
        report = compare_annotators(
            labels, dict(labels), 0.5, _CLASS_NAMES, annotator_a="asha", annotator_b="ravi"
        )
        verdict, failures = agreement_verdict(report, IaaSettings())
        payload = report_as_dict(report, verdict, failures)
        assert payload["annotators"] == ["asha", "ravi"]
        assert payload["verdict"] == "pass"
        assert payload["per_class"]["person"]["agreement"] == 1.0
        assert payload["worst_images"] == [{"image": "img1", "agreement": 1.0}]

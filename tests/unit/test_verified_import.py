"""Unit tests for CVAT export -> ledger import (delta extraction, verdicts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import VerificationBatchManifest
from src.dataset.annotation.ledger import LedgerView, new_ledger
from src.dataset.annotation.verified_import import (
    check_non_target_labels_unchanged,
    compute_batch_iaa_agreement,
    extract_deltas,
    import_verified_batch,
)
from src.dataset.capture.annotations import YoloExport

pytestmark = pytest.mark.unit

_CLASS_NAMES_BY_ID = {0: "person", 10: "charger", 11: "wire"}
_IDS_BY_NAME = {"person": 0, "charger": 10, "wire": 11}


def _batch(target_classes: list[str], images: list[str], **overrides) -> VerificationBatchManifest:
    defaults: dict = dict(
        batch_id="vb001_yolo_world",
        candidate_run={"backend": "yolo_world", "run_id": "r1", "candidates_sha256": "sha"},
        target_classes=target_classes,
        images=images,
        status="exported",
        cvat_task_ref="task-1",
    )
    defaults.update(overrides)
    return VerificationBatchManifest(**defaults)


class TestCheckNonTargetLabelsUnchanged:
    def test_no_base_no_export_is_clean(self, tmp_path: Path) -> None:
        assert (
            check_non_target_labels_unchanged("a.jpg", [], tmp_path / "absent.txt", frozenset({10}))
            == []
        )

    def test_identical_non_target_lines_is_clean(self, tmp_path: Path) -> None:
        base = tmp_path / "a.txt"
        base.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        problems = check_non_target_labels_unchanged(
            "a.jpg", ["0 0.5 0.5 0.2 0.2"], base, frozenset({10})
        )
        assert problems == []

    def test_reordered_non_target_lines_is_clean(self, tmp_path: Path) -> None:
        base = tmp_path / "a.txt"
        base.write_text("0 0.1 0.1 0.1 0.1\n0 0.9 0.9 0.1 0.1\n", encoding="utf-8")
        problems = check_non_target_labels_unchanged(
            "a.jpg", ["0 0.9 0.9 0.1 0.1", "0 0.1 0.1 0.1 0.1"], base, frozenset({10})
        )
        assert problems == []

    def test_edited_non_target_line_is_flagged(self, tmp_path: Path) -> None:
        base = tmp_path / "a.txt"
        base.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        problems = check_non_target_labels_unchanged(
            "a.jpg", ["0 0.51 0.5 0.2 0.2"], base, frozenset({10})
        )
        assert len(problems) == 1
        assert "a.jpg" in problems[0]

    def test_target_class_lines_are_ignored(self, tmp_path: Path) -> None:
        base = tmp_path / "a.txt"
        base.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        # Export adds a NEW target-class (10) box — not a non-target edit.
        problems = check_non_target_labels_unchanged(
            "a.jpg", ["0 0.5 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05"], base, frozenset({10})
        )
        assert problems == []


class TestExtractDeltas:
    def test_only_target_classes_kept(self) -> None:
        lines = ["0 0.5 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05", "11 0.2 0.2 0.05 0.05"]
        deltas = extract_deltas(lines, frozenset({10}))
        assert [d.class_id for d in deltas] == [10]

    def test_malformed_lines_skipped(self) -> None:
        assert extract_deltas(["not a valid line"], frozenset({10})) == []


class TestImportVerifiedBatch:
    def _full_export(self, labels: dict[str, list[str]]) -> YoloExport:
        # Full 23-class ordered names list mirroring configs/data.yaml (0..22).
        names = [
            "person",
            "face",
            "medicine_strip",
            "medicine_bottle",
            "water_bottle",
            "knife",
            "stove",
            "gas_cylinder",
            "passport",
            "book",
            "charger",
            "wire",
            "laptop",
            "monitor",
            "cupboard",
            "door",
            "chair",
            "bed",
            "toilet",
            "sink",
            "wet_floor",
            "walking_stick",
            "support_handle",
        ]
        return YoloExport(names=names, labels=labels)

    def _class_names(self) -> dict[int, str]:
        names = self._full_export({}).names
        return dict(enumerate(names))

    def test_wrong_class_order_hard_fails(self, tmp_path: Path) -> None:
        batch = _batch(["charger"], ["a.jpg"])
        export = YoloExport(names=["wrong", "order"], labels={"a": ["10 0.1 0.1 0.05 0.05"]})
        with pytest.raises(AnnotationError, match="does not match the taxonomy"):
            import_verified_batch(
                batch,
                export,
                self._class_names(),
                _IDS_BY_NAME,
                tmp_path / "merged_labels",
                tmp_path / "verified_labels",
                new_ledger(),
                {"a.jpg": "coco"},
                verifier="anno_1",
            )

    def test_non_target_edit_hard_fails(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        (merged / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        batch = _batch(["charger"], ["a.jpg"])
        export = self._full_export({"a": ["0 0.51 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05"]})
        with pytest.raises(AnnotationError, match="edited a trusted box"):
            import_verified_batch(
                batch,
                export,
                self._class_names(),
                _IDS_BY_NAME,
                merged,
                tmp_path / "verified_labels",
                new_ledger(),
                {"a.jpg": "coco"},
                verifier="anno_1",
            )

    def test_present_labeled_and_verified_absent_recorded(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        batch = _batch(["charger", "wire"], ["a.jpg"])
        export = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        ledger = new_ledger()
        result = import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            tmp_path / "verified_labels",
            ledger,
            {"a.jpg": "coco"},
            verifier="anno_1",
        )
        assert result.verdicts_recorded == 2
        view = LedgerView(raw=ledger)
        assert view.raw["entries"]["a.jpg"]["classes"]["charger"]["status"] == "present_labeled"
        assert view.raw["entries"]["a.jpg"]["classes"]["wire"]["status"] == "verified_absent"

    def test_delta_file_written_target_classes_only(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        (merged / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        verified_labels = tmp_path / "verified_labels"
        batch = _batch(["charger"], ["a.jpg"])
        export = self._full_export({"a": ["0 0.5 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05"]})
        import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            verified_labels,
            new_ledger(),
            {"a.jpg": "coco"},
            verifier="anno_1",
        )
        text = (verified_labels / "a.txt").read_text(encoding="utf-8")
        assert text.strip() == "10 0.100000 0.100000 0.050000 0.050000"

    def test_verified_absent_only_writes_no_delta_file(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        verified_labels = tmp_path / "verified_labels"
        batch = _batch(["charger"], ["a.jpg"])
        export = self._full_export({"a": []})
        import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            verified_labels,
            new_ledger(),
            {"a.jpg": "coco"},
            verifier="anno_1",
        )
        assert not (verified_labels / "a.txt").exists()

    def test_missing_provenance_source_raises(self, tmp_path: Path) -> None:
        batch = _batch(["charger"], ["a.jpg"])
        export = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        with pytest.raises(AnnotationError, match="no provenance source"):
            import_verified_batch(
                batch,
                export,
                self._class_names(),
                _IDS_BY_NAME,
                tmp_path / "merged_labels",
                tmp_path / "verified_labels",
                new_ledger(),
                {},
                verifier="anno_1",
            )

    def test_reimport_is_idempotent(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        batch = _batch(["charger"], ["a.jpg"])
        export = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        ledger = new_ledger()
        import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            tmp_path / "verified_labels",
            ledger,
            {"a.jpg": "coco"},
            verifier="anno_1",
        )
        import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            tmp_path / "verified_labels",
            ledger,
            {"a.jpg": "coco"},
            verifier="anno_1",
        )
        assert len(ledger["entries"]) == 1

    def test_partial_export_records_only_exported_images(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        batch = _batch(["charger"], ["a.jpg", "b.jpg"])
        export = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})  # b.jpg not exported
        result = import_verified_batch(
            batch,
            export,
            self._class_names(),
            _IDS_BY_NAME,
            merged,
            tmp_path / "verified_labels",
            new_ledger(),
            {"a.jpg": "coco", "b.jpg": "coco"},
            verifier="anno_1",
        )
        assert result.images_imported == 1
        assert any("b.jpg" in p for p in result.problems)


class TestComputeBatchIaaAgreement:
    def _full_export(self, labels: dict[str, list[str]]):
        names = [
            "person", "face", "medicine_strip", "medicine_bottle", "water_bottle", "knife",
            "stove", "gas_cylinder", "passport", "book", "charger", "wire", "laptop", "monitor",
            "cupboard", "door", "chair", "bed", "toilet", "sink", "wet_floor", "walking_stick",
            "support_handle",
        ]  # fmt: skip
        return YoloExport(names=names, labels=labels)

    def test_perfect_agreement_on_identical_exports(self) -> None:
        batch = _batch(["charger"], ["a.jpg", "b.jpg"], iaa_sample=["a.jpg"])
        export = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        report = compute_batch_iaa_agreement(batch, export, export, _IDS_BY_NAME)
        assert report.overall_agreement == pytest.approx(1.0)

    def test_disagreement_detected(self) -> None:
        batch = _batch(["charger"], ["a.jpg"], iaa_sample=["a.jpg"])
        primary = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        secondary = self._full_export({"a": []})
        report = compute_batch_iaa_agreement(batch, primary, secondary, _IDS_BY_NAME)
        assert report.overall_agreement == pytest.approx(0.0)

    def test_only_iaa_sample_images_considered(self) -> None:
        batch = _batch(["charger"], ["a.jpg", "b.jpg"], iaa_sample=["a.jpg"])
        primary = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"], "b": ["10 0.1 0.1 0.05 0.05"]})
        # b.jpg disagrees completely, but it's outside the IAA sample.
        secondary = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"], "b": []})
        report = compute_batch_iaa_agreement(batch, primary, secondary, _IDS_BY_NAME)
        assert report.overall_agreement == pytest.approx(1.0)

    def test_only_target_classes_considered(self) -> None:
        batch = _batch(["charger"], ["a.jpg"], iaa_sample=["a.jpg"])
        primary = self._full_export({"a": ["10 0.1 0.1 0.05 0.05", "11 0.5 0.5 0.1 0.1"]})
        # Disagree on 'wire' (11, non-target) — must not affect the gate.
        secondary = self._full_export({"a": ["10 0.1 0.1 0.05 0.05"]})
        report = compute_batch_iaa_agreement(batch, primary, secondary, _IDS_BY_NAME)
        assert report.overall_agreement == pytest.approx(1.0)

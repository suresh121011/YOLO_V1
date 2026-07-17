"""Unit tests for the verification ledger schema, IO, and read API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.ledger import (
    LedgerView,
    load_ledger,
    new_ledger,
    recompute_stats,
    record_verdict,
    save_ledger,
    validate_ledger,
)

pytestmark = pytest.mark.unit


class TestNewLedgerAndIO:
    def test_new_ledger_shape(self) -> None:
        ledger = new_ledger()
        assert ledger["schema_version"] == 1
        assert ledger["entries"] == {}
        assert ledger["stats"] == {"images": 0, "cells_verified": 0, "per_class": {}}

    def test_load_missing_file_is_empty_ledger(self, tmp_path: Path) -> None:
        assert load_ledger(tmp_path / "absent.json") == new_ledger()

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        ledger = new_ledger("fp123")
        record_verdict(
            ledger,
            "a.jpg",
            source="coco",
            class_name="charger",
            status="present_labeled",
            boxes=[(0.5, 0.5, 0.1, 0.1)],
            batch_id="vb001_yolo_world",
            verifier="anno_1",
            method="cvat",
            cvat_task_ref="task-1",
        )
        save_ledger(ledger, path)
        loaded = load_ledger(path)
        assert loaded["entries"]["a.jpg"]["classes"]["charger"]["status"] == "present_labeled"

    def test_duplicate_json_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.write_text('{"schema_version": 1, "schema_version": 1, "entries": {}, "stats": {}}')
        with pytest.raises(AnnotationError, match="Duplicate JSON key"):
            load_ledger(path)

    def test_unsupported_schema_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({"schema_version": 99, "entries": {}, "stats": {}}))
        with pytest.raises(AnnotationError, match="Unsupported ledger schema_version"):
            load_ledger(path)


class TestValidateLedger:
    def test_empty_ledger_is_valid(self) -> None:
        assert validate_ledger(new_ledger()) == []

    def test_missing_required_key(self) -> None:
        problems = validate_ledger({"schema_version": 1})
        assert any("entries" in p for p in problems)

    def test_taxonomy_fingerprint_drift(self) -> None:
        ledger = new_ledger("old_fp")
        problems = validate_ledger(ledger, expected_taxonomy_fp="new_fp")
        assert any("drift" in p for p in problems)

    def test_empty_recorded_fingerprint_is_not_drift(self) -> None:
        ledger = new_ledger("")
        assert validate_ledger(ledger, expected_taxonomy_fp="new_fp") == []

    def test_unknown_class_name(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "ghost_class", "verified_absent", [], "vb001", "v", "cvat", ""
        )
        problems = validate_ledger(ledger, class_names={"charger": 10})
        assert any("not in the taxonomy" in p for p in problems)

    def test_verified_absent_with_boxes_is_invalid(self) -> None:
        ledger = new_ledger()
        ledger["entries"]["a.jpg"] = {
            "source": "coco",
            "classes": {"charger": {"status": "verified_absent", "boxes": [[0.5, 0.5, 0.1, 0.1]]}},
        }
        problems = validate_ledger(ledger)
        assert any("verified_absent but carries" in p for p in problems)

    def test_present_labeled_without_boxes_is_invalid(self) -> None:
        ledger = new_ledger()
        ledger["entries"]["a.jpg"] = {
            "source": "coco",
            "classes": {"charger": {"status": "present_labeled", "boxes": []}},
        }
        problems = validate_ledger(ledger)
        assert any("present_labeled but carries zero boxes" in p for p in problems)


class TestRecordVerdict:
    def _ledger(self) -> dict:
        return new_ledger()

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(AnnotationError, match="Invalid verdict status"):
            record_verdict(
                self._ledger(), "a.jpg", "coco", "charger", "bogus", [], "vb1", "v", "m", ""
            )

    def test_verified_absent_with_boxes_raises(self) -> None:
        with pytest.raises(AnnotationError, match="must carry zero boxes"):
            record_verdict(
                self._ledger(),
                "a.jpg",
                "coco",
                "charger",
                "verified_absent",
                [(0.1, 0.1, 0.1, 0.1)],
                "vb1",
                "v",
                "m",
                "",
            )

    def test_present_labeled_without_boxes_raises(self) -> None:
        with pytest.raises(AnnotationError, match="must carry >=1 box"):
            record_verdict(
                self._ledger(),
                "a.jpg",
                "coco",
                "charger",
                "present_labeled",
                [],
                "vb1",
                "v",
                "m",
                "",
            )

    def test_source_mismatch_raises(self) -> None:
        ledger = self._ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        with pytest.raises(AnnotationError, match="provenance conflict"):
            record_verdict(
                ledger, "a.jpg", "openimages", "wire", "verified_absent", [], "vb2", "v", "m", ""
            )

    def test_identical_reimport_is_idempotent(self) -> None:
        ledger = self._ledger()
        record_verdict(
            ledger,
            "a.jpg",
            "coco",
            "charger",
            "present_labeled",
            [(0.5, 0.5, 0.1, 0.1)],
            "vb1",
            "anno_1",
            "cvat",
            "task-1",
        )
        before = json.loads(json.dumps(ledger))
        record_verdict(
            ledger,
            "a.jpg",
            "coco",
            "charger",
            "present_labeled",
            [(0.5, 0.5, 0.1, 0.1)],
            "vb1",
            "anno_1",
            "cvat",
            "task-1",
        )
        assert ledger == before

    def test_conflicting_verdict_without_supersedes_raises(self) -> None:
        ledger = self._ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        with pytest.raises(AnnotationError, match="conflicting verdict"):
            record_verdict(
                ledger,
                "a.jpg",
                "coco",
                "charger",
                "present_labeled",
                [(0.1, 0.1, 0.1, 0.1)],
                "vb2",
                "v",
                "m",
                "",
            )

    def test_conflicting_verdict_with_supersedes_succeeds(self) -> None:
        ledger = self._ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        record_verdict(
            ledger,
            "a.jpg",
            "coco",
            "charger",
            "present_labeled",
            [(0.1, 0.1, 0.1, 0.1)],
            "vb2",
            "v",
            "m",
            "",
            supersedes="vb1",
        )
        assert ledger["entries"]["a.jpg"]["classes"]["charger"]["status"] == "present_labeled"
        assert ledger["entries"]["a.jpg"]["supersedes"] == "vb1"

    def test_different_class_same_image_appends(self) -> None:
        ledger = self._ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        record_verdict(ledger, "a.jpg", "coco", "wire", "verified_absent", [], "vb2", "v", "m", "")
        assert set(ledger["entries"]["a.jpg"]["classes"]) == {"charger", "wire"}


class TestRecomputeStats:
    def test_counts_cells_and_images(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        record_verdict(ledger, "a.jpg", "coco", "wire", "verified_absent", [], "vb1", "v", "m", "")
        record_verdict(
            ledger, "b.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        recompute_stats(ledger, "fp123")
        assert ledger["stats"] == {
            "images": 2,
            "cells_verified": 3,
            "per_class": {"charger": 2, "wire": 1},
        }
        assert ledger["taxonomy_fingerprint"] == "fp123"
        assert ledger["updated_at"] != ""


class TestLedgerView:
    def test_verified_cells_both_verdict_kinds_count(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        record_verdict(
            ledger,
            "a.jpg",
            "coco",
            "wire",
            "present_labeled",
            [(0.1, 0.1, 0.1, 0.1)],
            "vb1",
            "v",
            "m",
            "",
        )
        view = LedgerView(raw=ledger)
        cells = view.verified_cells({"charger": 10, "wire": 11})
        assert cells == {"a.jpg": frozenset({10, 11})}

    def test_verified_cells_unknown_class_raises(self) -> None:
        ledger = new_ledger()
        record_verdict(ledger, "a.jpg", "coco", "ghost", "verified_absent", [], "vb1", "v", "m", "")
        with pytest.raises(AnnotationError, match="not in the taxonomy"):
            LedgerView(raw=ledger).verified_cells({"charger": 10})

    def test_load_missing_file(self, tmp_path: Path) -> None:
        view = LedgerView.load(tmp_path / "absent.json")
        assert view.all_images() == frozenset()

    def test_verified_class_names(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        view = LedgerView(raw=ledger)
        assert view.verified_class_names("a.jpg") == frozenset({"charger"})
        assert view.verified_class_names("missing.jpg") == frozenset()

    def test_entry_source(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        view = LedgerView(raw=ledger)
        assert view.entry_source("a.jpg") == "coco"
        assert view.entry_source("missing.jpg") is None

    def test_all_images(self) -> None:
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        record_verdict(
            ledger, "b.jpg", "coco", "charger", "verified_absent", [], "vb1", "v", "m", ""
        )
        assert LedgerView(raw=ledger).all_images() == frozenset({"a.jpg", "b.jpg"})

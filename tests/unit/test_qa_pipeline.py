"""
Unit tests for scripts.qa.check_annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.qa.check_annotations import (
    QAIssue,
    QAResults,
    build_qa_reports,
    check_annotation_format,
    check_file_pairs,
    check_split_leakage,
)

# ─── QAResults ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestQAResults:
    def test_critical_count(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("check_a", "CRITICAL", "train", "file.txt", 1, "bad"))
        r.add_issue(QAIssue("check_b", "WARNING", "val", "file2.txt", 2, "warn"))
        assert r.critical_count == 1
        assert r.warning_count == 1
        assert r.info_count == 0

    def test_finalize_check_pass(self) -> None:
        r = QAResults()
        r.finalize_check("my_check", 0)
        assert r.check_summaries["my_check"]["status"] == "PASS"
        assert r.check_summaries["my_check"]["count"] == 0

    def test_finalize_check_critical(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("my_check", "CRITICAL", "train", "", 0, "crit issue"))
        r.finalize_check("my_check", 1)
        assert r.check_summaries["my_check"]["status"] == "CRITICAL"

    def test_finalize_check_warning(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("my_check", "WARNING", "train", "", 0, "warn issue"))
        r.finalize_check("my_check", 1)
        assert r.check_summaries["my_check"]["status"] == "WARNING"


# ─── check_annotation_format ─────────────────────────────────────────────────


def _make_dataset(tmp_path: Path, split: str = "train") -> tuple[Path, Path]:
    """Create minimal dataset structure for QA tests."""
    img_dir = tmp_path / "images" / split
    lbl_dir = tmp_path / "labels" / split
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    return img_dir, lbl_dir


CLASS_NAMES = {
    i: name
    for i, name in enumerate(
        [
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
    )
}
NUM_CLASSES = 23


@pytest.mark.unit
class TestCheckAnnotationFormat:
    def test_valid_annotation_no_issues(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert results.critical_count == 0

    def test_detects_invalid_class_id(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("99 0.5 0.5 0.2 0.3\n")  # class 99 is invalid

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "invalid_class_ids" for i in results.issues)

    def test_detects_bbox_out_of_bounds(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("0 1.5 0.5 0.2 0.3\n")  # cx=1.5 out of range

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "bbox_out_of_bounds" for i in results.issues)

    def test_detects_zero_area_box(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.0 0.3\n")  # w=0

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "zero_area_boxes" for i in results.issues)

    def test_detects_invalid_format(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5\n")  # only 3 fields

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "invalid_yolo_format" for i in results.issues)

    def test_detects_duplicate_annotations(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        # Same annotation twice
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "duplicate_annotations" for i in results.issues)

    def test_missing_label_dir_skipped(self, tmp_path: Path) -> None:
        """No crash when labels/ directory doesn't exist."""
        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert results.critical_count == 0


# ─── check_file_pairs ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckFilePairs:
    def test_empty_label_file_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("")  # empty label

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "empty_label_files" for i in results.issues)

    def test_missing_label_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        # No label file created

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "missing_label_files" for i in results.issues)

    def test_missing_image_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        # Only label, no image
        (lbl_dir / "orphan.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "missing_image_files" for i in results.issues)

    def test_valid_pairs_no_issues(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert results.warning_count == 0


# ─── check_split_leakage ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckSplitLeakage:
    def test_no_leakage_no_issues(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            d = tmp_path / "images" / split
            d.mkdir(parents=True)
            (d / f"{split}_only.jpg").write_bytes(f"unique_content_{split}".encode())

        results = QAResults()
        check_split_leakage(tmp_path, results)
        assert results.critical_count == 0

    def test_train_val_leakage_detected(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            (tmp_path / "images" / split).mkdir(parents=True)

        shared_content = b"identical_image_content"
        (tmp_path / "images" / "train" / "shared.jpg").write_bytes(shared_content)
        (tmp_path / "images" / "val" / "shared_copy.jpg").write_bytes(shared_content)
        (tmp_path / "images" / "test" / "unique.jpg").write_bytes(b"different")

        results = QAResults()
        check_split_leakage(tmp_path, results)
        assert any(i.check == "train_val_leakage" for i in results.issues)
        assert any(i.severity == "CRITICAL" for i in results.issues)


# ─── build_qa_reports ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildQaReports:
    def test_reports_built_without_error(self, tmp_path: Path) -> None:
        results = QAResults()
        results.total_images = 100
        results.total_labels = 100
        results.total_boxes = 500
        results.add_issue(QAIssue("test_check", "WARNING", "train", "file.txt", 1, "msg"))
        results.finalize_check("test_check", 1)

        json_report, csv_rows, md_sections = build_qa_reports(results, tmp_path, 23)

        assert "summary" in json_report
        assert json_report["summary"]["total_images"] == 100
        assert len(csv_rows) == 1
        assert len(md_sections) > 0

    def test_json_includes_all_issues(self) -> None:
        results = QAResults()
        for i in range(5):
            results.add_issue(QAIssue(f"check_{i}", "WARNING", "train", f"f{i}.txt", i, f"msg{i}"))

        json_report, _, _ = build_qa_reports(results, Path("."), 23)
        assert len(json_report["issues"]) == 5

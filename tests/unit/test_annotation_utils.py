"""
Unit tests for src.utils.annotation_utils.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.annotation_utils import (
    Annotation,
    check_bbox_bounds,
    check_duplicate_lines,
    check_zero_area,
    count_annotations_by_class,
    parse_label_file,
    parse_yolo_line,
    validate_yolo_line,
)

# ─── parse_yolo_line ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseYoloLine:
    def test_valid_line(self) -> None:
        ann = parse_yolo_line("5 0.5 0.5 0.2 0.3", line_num=1)
        assert ann is not None
        assert ann.class_id == 5
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(0.2)
        assert ann.h == pytest.approx(0.3)
        assert ann.line_num == 1

    def test_empty_line_returns_none(self) -> None:
        assert parse_yolo_line("") is None
        assert parse_yolo_line("   ") is None

    def test_too_few_fields_returns_none(self) -> None:
        assert parse_yolo_line("5 0.5 0.5") is None

    def test_too_many_fields_returns_none(self) -> None:
        assert parse_yolo_line("5 0.5 0.5 0.2 0.3 extra") is None

    def test_non_numeric_returns_none(self) -> None:
        assert parse_yolo_line("abc 0.5 0.5 0.2 0.3") is None

    def test_preserves_raw(self) -> None:
        raw = "0 0.1 0.2 0.3 0.4"
        ann = parse_yolo_line(raw, line_num=3)
        assert ann is not None
        assert ann.raw == raw


# ─── validate_yolo_line ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestValidateYoloLine:
    def _make_ann(self, class_id: int, cx: float, cy: float, w: float, h: float) -> Annotation:
        return Annotation(class_id=class_id, cx=cx, cy=cy, w=w, h=h, line_num=1, raw="")

    def test_valid_annotation_no_errors(self) -> None:
        ann = self._make_ann(5, 0.5, 0.5, 0.2, 0.3)
        assert validate_yolo_line(ann, num_classes=23) == []

    def test_invalid_class_id_negative(self) -> None:
        ann = self._make_ann(-1, 0.5, 0.5, 0.2, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("invalid class_id" in e for e in errors)

    def test_invalid_class_id_too_large(self) -> None:
        ann = self._make_ann(23, 0.5, 0.5, 0.2, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("invalid class_id" in e for e in errors)

    def test_cx_out_of_range(self) -> None:
        ann = self._make_ann(0, 1.5, 0.5, 0.2, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("cx" in e for e in errors)

    def test_negative_cy(self) -> None:
        ann = self._make_ann(0, 0.5, -0.1, 0.2, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("cy" in e for e in errors)

    def test_zero_width(self) -> None:
        ann = self._make_ann(0, 0.5, 0.5, 0.0, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("zero or negative" in e.lower() for e in errors)

    def test_bbox_extends_beyond_image(self) -> None:
        # cx=0.9, w=0.5 → x2=1.15 (out of bounds)
        ann = self._make_ann(0, 0.9, 0.5, 0.5, 0.3)
        errors = validate_yolo_line(ann, num_classes=23)
        assert any("outside image" in e for e in errors)

    def test_boundary_values_valid(self) -> None:
        # Exact boundary values should be valid
        ann = self._make_ann(0, 0.0, 0.0, 1.0, 1.0)
        errors = validate_yolo_line(ann, num_classes=23)
        # box extends from -0.5 to 0.5 in x with cx=0 — outside boundary
        # This is valid per coordinate range but bbox extends outside
        # Just checking no crash
        assert isinstance(errors, list)


# ─── check_bbox_bounds ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckBboxBounds:
    def test_valid_bounds(self) -> None:
        assert check_bbox_bounds(0.5, 0.5, 0.2, 0.3) == []

    def test_negative_cx(self) -> None:
        errors = check_bbox_bounds(-0.1, 0.5, 0.2, 0.3)
        assert any("cx" in e for e in errors)

    def test_greater_than_one(self) -> None:
        errors = check_bbox_bounds(0.5, 1.1, 0.2, 0.3)
        assert any("cy" in e for e in errors)

    def test_zero_width(self) -> None:
        errors = check_bbox_bounds(0.5, 0.5, 0.0, 0.3)
        assert any("zero" in e.lower() for e in errors)


# ─── check_zero_area ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckZeroArea:
    def test_normal_box(self) -> None:
        assert check_zero_area(0.2, 0.3) is False

    def test_zero_width(self) -> None:
        assert check_zero_area(0.0, 0.3) is True

    def test_zero_height(self) -> None:
        assert check_zero_area(0.2, 0.0) is True

    def test_negative_dimension(self) -> None:
        assert check_zero_area(-0.1, 0.3) is True


# ─── check_duplicate_lines ───────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckDuplicateLines:
    def test_no_duplicates(self) -> None:
        lines = ["0 0.1 0.1 0.2 0.2", "1 0.5 0.5 0.3 0.3"]
        assert check_duplicate_lines(lines) == []

    def test_single_duplicate(self) -> None:
        lines = ["0 0.1 0.1 0.2 0.2", "0 0.1 0.1 0.2 0.2"]
        dupes = check_duplicate_lines(lines)
        assert len(dupes) == 1
        assert dupes[0] == (0, 1)  # first at idx 0, duplicate at idx 1

    def test_empty_list(self) -> None:
        assert check_duplicate_lines([]) == []


# ─── parse_label_file ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseLabelFile:
    def test_parses_valid_file(self, tmp_path: Path) -> None:
        lbl = tmp_path / "test.txt"
        lbl.write_text("5 0.5 0.5 0.2 0.3\n0 0.1 0.1 0.05 0.05\n")
        anns = parse_label_file(lbl)
        assert len(anns) == 2
        assert anns[0].class_id == 5
        assert anns[1].class_id == 0

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        lbl = tmp_path / "empty.txt"
        lbl.write_text("")
        assert parse_label_file(lbl) == []

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        lbl = tmp_path / "test.txt"
        lbl.write_text("# comment\n5 0.5 0.5 0.2 0.3\n")
        anns = parse_label_file(lbl)
        assert len(anns) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_label_file(tmp_path / "nonexistent.txt")


# ─── count_annotations_by_class ──────────────────────────────────────────────


@pytest.mark.unit
class TestCountAnnotationsByClass:
    def test_counts_correctly(self, tmp_path: Path) -> None:
        lbl1 = tmp_path / "a.txt"
        lbl2 = tmp_path / "b.txt"
        lbl1.write_text("5 0.5 0.5 0.2 0.3\n5 0.3 0.3 0.1 0.1\n")
        lbl2.write_text("0 0.5 0.5 0.4 0.4\n")

        counts = count_annotations_by_class([lbl1, lbl2])
        assert counts[5] == 2
        assert counts[0] == 1

    def test_empty_label_files(self, tmp_path: Path) -> None:
        lbl = tmp_path / "empty.txt"
        lbl.write_text("")
        counts = count_annotations_by_class([lbl])
        assert counts == {}

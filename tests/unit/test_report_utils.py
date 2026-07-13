"""
Unit tests for src.utils.report_utils.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.utils.report_utils import (
    format_count_pct,
    format_severity_badge,
    format_table,
    save_csv_report,
    save_json_report,
    save_markdown_report,
    timestamp_str,
    write_all_formats,
)


@pytest.mark.unit
class TestTimestampStr:
    def test_returns_string(self) -> None:
        ts = timestamp_str()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_ends_with_z(self) -> None:
        ts = timestamp_str()
        assert ts.endswith("Z")


@pytest.mark.unit
class TestSaveJsonReport:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = save_json_report(data, tmp_path / "report.json")

        with open(path) as f:
            loaded = json.load(f)

        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = save_json_report({"x": 1}, tmp_path / "deep" / "dir" / "report.json")
        assert path.exists()

    def test_returns_path(self, tmp_path: Path) -> None:
        expected = tmp_path / "out.json"
        returned = save_json_report({}, expected)
        assert returned == expected


@pytest.mark.unit
class TestSaveCsvReport:
    def test_writes_csv_with_headers(self, tmp_path: Path) -> None:
        rows = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 87},
        ]
        path = save_csv_report(rows, tmp_path / "report.csv")

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            loaded = list(reader)

        assert len(loaded) == 2
        assert loaded[0]["name"] == "Alice"
        assert loaded[1]["score"] == "87"

    def test_empty_rows_writes_empty_file(self, tmp_path: Path) -> None:
        path = save_csv_report([], tmp_path / "empty.csv")
        assert path.exists()
        assert path.read_text() == ""

    def test_custom_fieldnames_order(self, tmp_path: Path) -> None:
        rows = [{"b": 2, "a": 1}]
        path = save_csv_report(rows, tmp_path / "ordered.csv", fieldnames=["a", "b"])

        with open(path) as f:
            header = f.readline().strip()

        assert header == "a,b"


@pytest.mark.unit
class TestSaveMarkdownReport:
    def test_writes_title(self, tmp_path: Path) -> None:
        path = save_markdown_report("Test Report", [], tmp_path / "report.md")
        content = path.read_text()
        assert "# Test Report" in content

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        path = save_markdown_report("T", [], tmp_path / "r.md")
        content = path.read_text()
        assert "Generated:" in content

    def test_section_heading(self, tmp_path: Path) -> None:
        sections = [{"heading": "My Section", "content": "Hello world"}]
        path = save_markdown_report("Title", sections, tmp_path / "r.md")
        content = path.read_text()
        assert "## My Section" in content
        assert "Hello world" in content

    def test_table_section(self, tmp_path: Path) -> None:
        sections = [{
            "heading": "Table",
            "table": {
                "headers": ["Class", "Count"],
                "rows": [["knife", "42"], ["stove", "15"]],
            },
        }]
        path = save_markdown_report("T", sections, tmp_path / "r.md")
        content = path.read_text()
        assert "knife" in content
        assert "42" in content

    def test_metadata_included(self, tmp_path: Path) -> None:
        path = save_markdown_report(
            "T", [], tmp_path / "r.md",
            metadata={"Key": "Value", "Number": 99}
        )
        content = path.read_text()
        assert "**Key:**" in content
        assert "Value" in content


@pytest.mark.unit
class TestFormatTable:
    def test_empty_headers_returns_empty(self) -> None:
        assert format_table([], []) == ""

    def test_single_row(self) -> None:
        result = format_table(["A", "B"], [["1", "2"]])
        lines = result.splitlines()
        assert len(lines) == 3  # header | separator | data row
        assert "A" in lines[0]
        assert "1" in lines[2]

    def test_separator_uses_dashes(self) -> None:
        result = format_table(["Col"], [["val"]])
        lines = result.splitlines()
        assert "-" in lines[1]


@pytest.mark.unit
class TestFormatSeverityBadge:
    @pytest.mark.parametrize("severity,expected_start", [
        ("CRITICAL", "🔴"),
        ("WARNING", "🟡"),
        ("INFO", "🔵"),
        ("PASS", "✅"),
        ("UNKNOWN", "UNKNOWN"),
    ])
    def test_badge_format(self, severity: str, expected_start: str) -> None:
        badge = format_severity_badge(severity)
        assert badge.startswith(expected_start)


@pytest.mark.unit
class TestFormatCountPct:
    def test_normal(self) -> None:
        result = format_count_pct(10, 100)
        assert "10" in result
        assert "10.0%" in result

    def test_zero_total(self) -> None:
        result = format_count_pct(0, 0)
        assert "0.0%" in result

    def test_zero_count(self) -> None:
        result = format_count_pct(0, 50)
        assert "0" in result


@pytest.mark.unit
class TestWriteAllFormats:
    def test_creates_all_three_files(self, tmp_path: Path) -> None:
        paths = write_all_formats(
            report_data={"key": "val"},
            csv_rows=[{"col": "data"}],
            md_title="Test",
            md_sections=[],
            output_dir=tmp_path,
            base_name="test_report",
        )
        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["markdown"].exists()

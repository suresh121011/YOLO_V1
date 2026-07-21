"""Unit tests for the verified-labels overlay builder (ADR-P5-05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.apply import build_verified_labels_overlay

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestBuildVerifiedLabelsOverlay:
    def test_empty_verified_labels_dir_is_byte_identical_passthrough(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        _write(merged / "b.txt", "1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.1 0.1\n")
        verified = tmp_path / "verified_labels"  # never created — empty ledger
        out = tmp_path / "out"

        result = build_verified_labels_overlay(merged, verified, out)

        assert result.images_total == 2
        assert result.images_with_deltas == 0
        assert (out / "a.txt").read_bytes() == (merged / "a.txt").read_bytes()
        assert (out / "b.txt").read_bytes() == (merged / "b.txt").read_bytes()

    def test_missing_verified_labels_dir_is_also_passthrough(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        out = tmp_path / "out"

        result = build_verified_labels_overlay(merged, tmp_path / "does_not_exist", out)

        assert result.images_with_deltas == 0
        assert (out / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"

    def test_delta_appended_after_base(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        _write(verified / "a.txt", "10 0.1 0.1 0.05 0.05\n")
        out = tmp_path / "out"

        result = build_verified_labels_overlay(merged, verified, out)

        assert result.images_with_deltas == 1
        assert result.delta_lines_added == 1
        lines = (out / "a.txt").read_text(encoding="utf-8").splitlines()
        assert lines == ["0 0.5 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05"]

    def test_empty_base_file_plus_delta(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "")  # image with zero trusted-class boxes
        _write(verified / "a.txt", "10 0.1 0.1 0.05 0.05\n")
        out = tmp_path / "out"

        build_verified_labels_overlay(merged, verified, out)

        assert (out / "a.txt").read_text(encoding="utf-8") == "10 0.1 0.1 0.05 0.05\n"

    def test_missing_final_newline_in_base_is_handled(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2")  # no trailing newline
        _write(verified / "a.txt", "10 0.1 0.1 0.05 0.05\n")
        out = tmp_path / "out"

        build_verified_labels_overlay(merged, verified, out)

        lines = (out / "a.txt").read_text(encoding="utf-8").splitlines()
        assert lines == ["0 0.5 0.5 0.2 0.2", "10 0.1 0.1 0.05 0.05"]

    def test_image_without_delta_is_untouched_alongside_one_with(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        _write(merged / "b.txt", "1 0.3 0.3 0.1 0.1\n")
        _write(verified / "a.txt", "10 0.1 0.1 0.05 0.05\n")
        out = tmp_path / "out"

        result = build_verified_labels_overlay(merged, verified, out)

        assert result.images_total == 2
        assert result.images_with_deltas == 1
        assert (out / "b.txt").read_bytes() == (merged / "b.txt").read_bytes()

    def test_rebuild_is_deterministic(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        _write(verified / "a.txt", "10 0.1 0.1 0.05 0.05\n")
        out = tmp_path / "out"

        build_verified_labels_overlay(merged, verified, out)
        first = (out / "a.txt").read_bytes()
        build_verified_labels_overlay(merged, verified, out)
        second = (out / "a.txt").read_bytes()
        assert first == second

    def test_stale_overlay_file_removed_when_base_image_gone(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        verified = tmp_path / "verified_labels"
        _write(merged / "a.txt", "0 0.5 0.5 0.2 0.2\n")
        out = tmp_path / "out"
        build_verified_labels_overlay(merged, verified, out)
        assert (out / "a.txt").exists()

        (merged / "a.txt").unlink()
        _write(merged / "b.txt", "1 0.3 0.3 0.1 0.1\n")
        build_verified_labels_overlay(merged, verified, out)

        assert not (out / "a.txt").exists()
        assert (out / "b.txt").exists()

    def test_no_merged_images_produces_empty_overlay(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged_labels"
        merged.mkdir()
        out = tmp_path / "out"

        result = build_verified_labels_overlay(merged, tmp_path / "verified_labels", out)

        assert result.images_total == 0
        assert list(out.glob("*.txt")) == []

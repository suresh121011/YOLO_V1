"""
Unit tests for src.utils.dataset_utils.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.utils.dataset_utils import (
    IMAGE_EXTENSIONS,
    build_hash_index,
    compute_file_hash,
    extract_group_key,
    find_image_files,
    find_label_files,
    get_image_label_pairs,
    group_files_by_key,
)


# ─── find_image_files ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFindImageFiles:
    def test_finds_jpg_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.jpeg").write_bytes(b"")
        files = find_image_files(tmp_path)
        assert len(files) == 2

    def test_finds_png_files(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"")
        files = find_image_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "image.png"

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "label.txt").write_bytes(b"")
        (tmp_path / "readme.md").write_bytes(b"")
        assert find_image_files(tmp_path) == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert find_image_files(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert find_image_files(tmp_path / "nonexistent") == []

    def test_recursive(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.jpg").write_bytes(b"")
        (tmp_path / "top.jpg").write_bytes(b"")
        files = find_image_files(tmp_path)
        assert len(files) == 2

    def test_returns_sorted(self, tmp_path: Path) -> None:
        for name in ["z.jpg", "a.jpg", "m.jpg"]:
            (tmp_path / name).write_bytes(b"")
        files = find_image_files(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)


# ─── find_label_files ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFindLabelFiles:
    def test_finds_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "label.txt").write_bytes(b"")
        files = find_label_files(tmp_path)
        assert len(files) == 1

    def test_ignores_jpg(self, tmp_path: Path) -> None:
        (tmp_path / "image.jpg").write_bytes(b"")
        assert find_label_files(tmp_path) == []


# ─── get_image_label_pairs ────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetImageLabelPairs:
    def test_matched_pairs(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        (img_dir / "dog.jpg").write_bytes(b"")
        (lbl_dir / "dog.txt").write_text("0 0.5 0.5 0.2 0.3")

        pairs = get_image_label_pairs(img_dir, lbl_dir)
        assert len(pairs) == 1
        img, lbl = pairs[0]
        assert img.name == "dog.jpg"
        assert lbl is not None
        assert lbl.name == "dog.txt"

    def test_missing_label_returns_none(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        (img_dir / "cat.jpg").write_bytes(b"")
        # No label file created

        pairs = get_image_label_pairs(img_dir, lbl_dir)
        assert len(pairs) == 1
        _, lbl = pairs[0]
        assert lbl is None


# ─── compute_file_hash ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeFileHash:
    def test_hash_is_sha256_hex(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        digest = compute_file_hash(f)
        # SHA-256 of "hello world"
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert digest == expected

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "nonexistent.bin")


# ─── build_hash_index ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildHashIndex:
    def test_detects_duplicate(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"same")
        f2.write_bytes(b"same")
        index = build_hash_index([f1, f2])
        # Should have one hash pointing to both files
        assert any(len(v) == 2 for v in index.values())

    def test_no_duplicates(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        index = build_hash_index([f1, f2])
        assert all(len(v) == 1 for v in index.values())


# ─── extract_group_key ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractGroupKey:
    @pytest.mark.parametrize("filename,expected", [
        ("video001_frame_00042.jpg", "video001"),
        ("custom_kitchen_001.jpg", "custom_kitchen"),
        ("IMG_20240601_143022_001.jpg", "IMG_20240601_143022"),
        ("standalone.jpg", "standalone"),  # no pattern match → full stem
        # 'frame_00001' matches the last pattern: ([a-zA-Z][a-zA-Z0-9_]+)_\d+
        # capturing group is 'frame' (the alphabetic prefix)
        ("frame_00001.jpg", "frame"),
    ])
    def test_group_extraction(self, filename: str, expected: str) -> None:
        assert extract_group_key(filename) == expected


# ─── group_files_by_key ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestGroupFilesByKey:
    def test_groups_correctly(self, tmp_path: Path) -> None:
        files = [
            tmp_path / "video001_frame_001.jpg",
            tmp_path / "video001_frame_002.jpg",
            tmp_path / "video002_frame_001.jpg",
        ]
        for f in files:
            f.write_bytes(b"")

        groups = group_files_by_key(files)
        assert "video001" in groups
        assert "video002" in groups
        assert len(groups["video001"]) == 2
        assert len(groups["video002"]) == 1

    def test_empty_input(self) -> None:
        groups = group_files_by_key([])
        assert groups == {}

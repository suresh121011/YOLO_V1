"""
Unit tests for scripts.dataset.split_dataset.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.dataset.split_dataset import (
    compute_split_assignments,
    copy_split_files,
    verify_no_leakage,
)
from src.utils.dataset_utils import group_files_by_key


def _create_test_dataset(tmp_path: Path, n_groups: int = 10, images_per_group: int = 3) -> tuple[Path, Path]:
    """Create a minimal synthetic dataset for testing."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for g in range(n_groups):
        for i in range(images_per_group):
            name = f"video{g:03d}_frame_{i:04d}"
            (img_dir / f"{name}.jpg").write_bytes(b"fake_image_content")
            (lbl_dir / f"{name}.txt").write_text("5 0.5 0.5 0.2 0.3\n")

    return img_dir, lbl_dir


@pytest.mark.unit
class TestComputeSplitAssignments:
    def test_ratios_sum_to_one(self) -> None:
        groups = {f"g{i}": [Path(f"img{i}.jpg")] for i in range(100)}
        assignments = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=42)
        total = sum(len(v) for v in assignments.values())
        assert total == 100

    def test_all_groups_assigned(self) -> None:
        groups = {f"g{i}": [Path(f"img{i}.jpg")] for i in range(50)}
        assignments = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=42)
        assigned = set()
        for keys in assignments.values():
            assigned.update(keys)
        assert assigned == set(groups.keys())

    def test_deterministic_with_same_seed(self) -> None:
        groups = {f"g{i}": [Path(f"img{i}.jpg")] for i in range(30)}
        a1 = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=99)
        a2 = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=99)
        assert a1["train"] == a2["train"]
        assert a1["val"] == a2["val"]
        assert a1["test"] == a2["test"]

    def test_different_seeds_different_results(self) -> None:
        groups = {f"g{i}": [Path(f"img{i}.jpg")] for i in range(50)}
        a1 = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=1)
        a2 = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=2)
        # With 50 groups, extremely unlikely to be identical
        assert a1["train"] != a2["train"] or a1["val"] != a2["val"]

    def test_ratios_not_summing_raises(self) -> None:
        groups = {"a": [Path("a.jpg")]}
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_split_assignments(groups, 0.5, 0.3, 0.1, seed=42)

    def test_approximate_ratio_splits(self) -> None:
        """80/10/10 split of 100 groups should produce ~80/10/10."""
        groups = {f"g{i}": [Path(f"img{i}.jpg")] for i in range(100)}
        assignments = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=42)
        assert len(assignments["train"]) == pytest.approx(80, abs=2)
        assert len(assignments["val"]) == pytest.approx(10, abs=2)
        assert len(assignments["test"]) == pytest.approx(10, abs=2)

    def test_group_integrity_preserved(self) -> None:
        """All keys in assignments must be from the original groups dict."""
        groups = {f"video{i:03d}": [Path(f"frame_{j}.jpg") for j in range(5)] for i in range(20)}
        assignments = compute_split_assignments(groups, 0.8, 0.1, 0.1, seed=42)
        all_assigned_keys = [k for split_keys in assignments.values() for k in split_keys]
        assert set(all_assigned_keys) == set(groups.keys())


@pytest.mark.unit
class TestCopySplitFiles:
    def test_files_are_copied(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _create_test_dataset(tmp_path / "source", n_groups=5, images_per_group=2)
        out_dir = tmp_path / "output"

        from src.utils.dataset_utils import find_image_files
        all_images = find_image_files(img_dir)
        groups = group_files_by_key(all_images)
        assignments = compute_split_assignments(groups, 0.6, 0.2, 0.2, seed=42)

        stats = copy_split_files(groups, assignments, img_dir, lbl_dir, out_dir)

        total_copied = sum(s["images"] for s in stats.values())
        assert total_copied == len(all_images)

    def test_output_directories_created(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _create_test_dataset(tmp_path / "source", n_groups=3, images_per_group=1)
        out_dir = tmp_path / "output"

        from src.utils.dataset_utils import find_image_files
        all_images = find_image_files(img_dir)
        groups = group_files_by_key(all_images)
        assignments = compute_split_assignments(groups, 0.6, 0.2, 0.2, seed=42)
        copy_split_files(groups, assignments, img_dir, lbl_dir, out_dir)

        assert (out_dir / "images" / "train").exists()
        assert (out_dir / "images" / "val").exists()
        assert (out_dir / "images" / "test").exists()


@pytest.mark.unit
class TestVerifyNoLeakage:
    def test_no_leakage_returns_empty_list(self, tmp_path: Path) -> None:
        # Create split directories with non-overlapping files
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / "images" / split
            split_dir.mkdir(parents=True)
            (split_dir / f"{split}_img_001.jpg").write_bytes(b"content")

        leakage = verify_no_leakage(tmp_path)
        assert leakage == []

    def test_detects_leakage_between_train_and_val(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / "images" / split
            split_dir.mkdir(parents=True)

        # Put same file in train and val
        (tmp_path / "images" / "train" / "leaked.jpg").write_bytes(b"x")
        (tmp_path / "images" / "val" / "leaked.jpg").write_bytes(b"x")

        leakage = verify_no_leakage(tmp_path)
        assert "leaked.jpg" in leakage

    def test_empty_directories_no_leakage(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            (tmp_path / "images" / split).mkdir(parents=True)
        assert verify_no_leakage(tmp_path) == []

"""Unit tests for src.dataset.merge and src.dataset.negatives."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.dataset.merge import MergeSource, merge_sources
from src.dataset.negatives import select_negative_candidates, write_empty_labels
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings

PIL = pytest.importorskip("PIL", reason="Pillow required for image fixtures")
from PIL import Image  # noqa: E402

CLASS_NAMES = {0: "person", 4: "water_bottle", 5: "knife"}


def _make_image(path: Path, seed: int, size: tuple[int, int] = (400, 400)) -> Path:
    """Synthetic image whose content depends on `seed` (distinct hashes)."""
    width, height = size
    img = Image.new("L", size)
    img.putdata(
        [
            ((x // 8) * (seed * 13 % 97) + (y // 8) * 7) % 256
            for y in range(height)
            for x in range(width)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _make_source_dir(
    tmp_path: Path,
    name: str,
    images: dict[str, int],
    labels: dict[str, str],
) -> Path:
    root = tmp_path / name
    for img_name, seed in images.items():
        _make_image(root / "images" / img_name, seed=seed)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for lbl_name, content in labels.items():
        (root / "labels" / lbl_name).write_text(content, encoding="utf-8")
    return root


@pytest.mark.unit
class TestMergeSources:
    """Merge with provenance, dedup, and label rules."""

    def test_basic_merge_with_provenance(self, tmp_path: Path) -> None:
        src_a = _make_source_dir(
            tmp_path,
            "alpha",
            images={"a1.png": 1, "a2.png": 2},
            labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n", "a2.txt": "5 0.5 0.5 0.2 0.2\n"},
        )
        src_b = _make_source_dir(
            tmp_path,
            "beta",
            images={"b1.png": 3},
            labels={"b1.txt": "4 0.5 0.5 0.2 0.2\n"},
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[
                MergeSource(name="alpha", root=src_a, trusted_classes=["person", "knife"]),
                MergeSource(name="beta", root=src_b, trusted_classes=["water_bottle"]),
            ],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=CLASS_NAMES,
        )

        assert (out / "images" / "alpha_a1.png").exists()
        assert (out / "labels" / "alpha_a1.txt").exists()
        assert (out / "images" / "beta_b1.png").exists()
        assert manifest.image_provenance["alpha_a1.png"] == "alpha"
        assert manifest.image_provenance["beta_b1.png"] == "beta"
        assert manifest.class_counts == {"person": 1, "knife": 1, "water_bottle": 1}
        assert manifest.label_completeness == {
            "alpha": ["person", "knife"],
            "beta": ["water_bottle"],
        }
        # Manifest persisted and loadable
        loaded = MergedManifest.load(out / MERGED_MANIFEST_FILENAME)
        assert loaded.image_provenance == manifest.image_provenance

    def test_cross_source_duplicate_removed_first_wins(self, tmp_path: Path) -> None:
        src_a = _make_source_dir(
            tmp_path,
            "alpha",
            images={"a1.png": 7},
            labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n"},
        )
        # beta contains the SAME image content under a different name
        src_b = _make_source_dir(
            tmp_path,
            "beta",
            images={"b1.png": 7},
            labels={"b1.txt": "0 0.5 0.5 0.2 0.2\n"},
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[
                MergeSource(name="alpha", root=src_a),
                MergeSource(name="beta", root=src_b),
            ],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=CLASS_NAMES,
        )

        assert manifest.duplicates_removed == 1
        assert (out / "images" / "alpha_a1.png").exists()
        assert not (out / "images" / "beta_b1.png").exists()
        beta_stats = next(s for s in manifest.sources if s["source"] == "beta")
        assert beta_stats["duplicates"] == 1

    def test_missing_and_empty_labels_dropped(self, tmp_path: Path) -> None:
        src = _make_source_dir(
            tmp_path,
            "alpha",
            images={"has_label.png": 1, "no_label.png": 2, "empty_label.png": 3},
            labels={"has_label.txt": "0 0.5 0.5 0.2 0.2\n", "empty_label.txt": ""},
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[MergeSource(name="alpha", root=src)],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=CLASS_NAMES,
        )

        stats = manifest.sources[0]
        assert stats["accepted"] == 1
        assert stats["missing_labels"] == 2

    def test_negatives_source_allows_empty_labels(self, tmp_path: Path) -> None:
        src = _make_source_dir(
            tmp_path,
            "negatives",
            images={"neg1.png": 11},
            labels={"neg1.txt": ""},
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[MergeSource(name="negatives", root=src, allow_empty_labels=True)],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=CLASS_NAMES,
        )
        assert manifest.sources[0]["accepted"] == 1
        assert (out / "labels" / "negatives_neg1.txt").exists()


@pytest.mark.unit
class TestNegatives:
    """Negative-candidate selection and empty-label writing."""

    def test_selection_excludes_taxonomy_classes(self) -> None:
        index = {
            "img_a": {"pizza", "fork"},
            "img_b": {"person", "pizza"},
            "img_c": set(),
            "img_d": {"knife"},
        }
        selected = select_negative_candidates(index, excluded_classes={"person", "knife"}, count=10)
        assert set(selected) == {"img_a", "img_c"}

    def test_selection_deterministic_and_capped(self) -> None:
        index = {f"img_{i}": set() for i in range(50)}
        first = select_negative_candidates(index, excluded_classes=set(), count=5, seed=42)
        second = select_negative_candidates(index, excluded_classes=set(), count=5, seed=42)
        assert first == second
        assert len(first) == 5

    def test_write_empty_labels(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        _make_image(images_dir / "n1.png", seed=1)
        _make_image(images_dir / "n2.png", seed=2)

        written = write_empty_labels(images_dir, tmp_path / "labels")
        assert written == 2
        assert (tmp_path / "labels" / "n1.txt").read_text(encoding="utf-8") == ""

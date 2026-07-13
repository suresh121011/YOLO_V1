"""Unit tests for src.dataset.remap — class remapping into the taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.remap import (
    NUM_CLASSES,
    REMAP_SENTINEL_FILENAME,
    REMAP_TABLES,
    build_id_mapping,
    remap_label_dir,
)


def _make_source(tmp_path: Path, classes: dict[str, str], labels: dict[str, str]) -> Path:
    """Create a fake raw source dir with source_classes.json + label files."""
    source_dir = tmp_path / "raw_source"
    labels_dir = source_dir / "labels"
    labels_dir.mkdir(parents=True)
    (source_dir / "source_classes.json").write_text(json.dumps(classes), encoding="utf-8")
    for name, content in labels.items():
        (labels_dir / name).write_text(content, encoding="utf-8")
    return source_dir


@pytest.mark.unit
class TestRemapTables:
    """Static validity of the documented remap tables."""

    def test_all_targets_within_taxonomy(self) -> None:
        for table_name, table in REMAP_TABLES.items():
            for class_name, target in table.items():
                assert 0 <= target < NUM_CLASSES, f"{table_name}:{class_name} → {target}"

    def test_docs_coco_table_correspondence(self) -> None:
        # docs table: {1:0, 44:4, 49:5, 73:12, 72:13, 62:16, 65:17, 70:18, 81:19, 84:9}
        coco = REMAP_TABLES["coco"]
        assert coco["person"] == 0
        assert coco["bottle"] == 4
        assert coco["knife"] == 5
        assert coco["tv"] == 13
        assert coco["book"] == 9
        assert len(coco) == 10

    def test_expected_tables_exist(self) -> None:
        for name in ("coco", "openimages", "wider_face", "roboflow", "identity"):
            assert name in REMAP_TABLES


@pytest.mark.unit
class TestBuildIdMapping:
    """Local-id → taxonomy-id mapping construction."""

    def test_mapped_and_unmapped_classes(self) -> None:
        mapping = build_id_mapping({"0": "person", "1": "pizza"}, REMAP_TABLES["coco"])
        assert mapping[0] == 0
        assert mapping[1] is None  # pizza not in taxonomy → drop

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError):
            build_id_mapping({"0": "bad"}, {"bad": 99})


@pytest.mark.unit
class TestRemapLabelDir:
    """End-to-end remap of a fake source directory."""

    def test_remaps_and_drops(self, tmp_path: Path) -> None:
        source_dir = _make_source(
            tmp_path,
            classes={"0": "person", "1": "bottle", "2": "pizza"},
            labels={
                "img1.txt": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.1 0.1 0.1 0.1\n",
                "img2.txt": "1 0.6 0.6 0.2 0.3\n",
            },
        )
        result = remap_label_dir(source_dir, REMAP_TABLES["coco"])

        assert result.files_processed == 2
        assert result.annotations_remapped == 3
        assert result.annotations_dropped == 1
        assert result.dropped_by_class == {"pizza": 1}

        img1 = (source_dir / "labels" / "img1.txt").read_text(encoding="utf-8")
        assert img1.splitlines() == ["0 0.5 0.5 0.2 0.2", "4 0.3 0.3 0.1 0.1"]
        img2 = (source_dir / "labels" / "img2.txt").read_text(encoding="utf-8")
        assert img2.splitlines() == ["4 0.6 0.6 0.2 0.3"]

    def test_sentinel_prevents_double_remap(self, tmp_path: Path) -> None:
        source_dir = _make_source(
            tmp_path,
            classes={"0": "person"},
            labels={"img1.txt": "0 0.5 0.5 0.2 0.2\n"},
        )
        first = remap_label_dir(source_dir, REMAP_TABLES["coco"])
        assert not first.skipped
        assert (source_dir / REMAP_SENTINEL_FILENAME).exists()

        second = remap_label_dir(source_dir, REMAP_TABLES["coco"])
        assert second.skipped
        assert second.files_processed == 0
        # Labels untouched by the second (skipped) run
        content = (source_dir / "labels" / "img1.txt").read_text(encoding="utf-8")
        assert content.splitlines() == ["0 0.5 0.5 0.2 0.2"]

    def test_missing_source_classes_raises(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "empty_source"
        (source_dir / "labels").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            remap_label_dir(source_dir, REMAP_TABLES["coco"])

    def test_malformed_lines_dropped(self, tmp_path: Path) -> None:
        source_dir = _make_source(
            tmp_path,
            classes={"0": "person"},
            labels={"img1.txt": "not_a_class 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"},
        )
        result = remap_label_dir(source_dir, REMAP_TABLES["coco"])
        assert result.annotations_remapped == 1
        assert result.annotations_dropped == 1

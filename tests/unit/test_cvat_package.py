"""Unit tests for CVAT YOLO 1.1 pre-annotation packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.dataset.annotation.cvat_package import (
    build_cvat_labels_spec,
    build_preannotation_labels,
    build_preannotation_zip,
)

pytestmark = pytest.mark.unit

_NAMES_BY_ID = {0: "person", 1: "face", 2: "charger", 3: "wire"}


class TestBuildCvatLabelsSpec:
    def test_taxonomy_id_order(self) -> None:
        spec = build_cvat_labels_spec(_NAMES_BY_ID)
        assert spec == [
            {"name": "person", "attributes": []},
            {"name": "face", "attributes": []},
            {"name": "charger", "attributes": []},
            {"name": "wire", "attributes": []},
        ]

    def test_unordered_input_still_sorted_by_id(self) -> None:
        shuffled = {3: "wire", 0: "person", 2: "charger", 1: "face"}
        spec = build_cvat_labels_spec(shuffled)
        assert [s["name"] for s in spec] == ["person", "face", "charger", "wire"]

    def test_every_label_has_attributes_array(self) -> None:
        # CVAT's Raw label editor (validateParsedLabel) rejects any label whose
        # `attributes` is not an array — this is the exact check that produced
        # the "labels: [object Object]" / POST /api/tasks 400 failure.
        spec = build_cvat_labels_spec(_NAMES_BY_ID)
        assert all(isinstance(label["attributes"], list) for label in spec)


class TestBuildPreannotationLabels:
    def test_no_base_no_candidates_is_empty(self, tmp_path: Path) -> None:
        assert build_preannotation_labels([], tmp_path / "absent.txt") == ""

    def test_base_labels_preserved(self, tmp_path: Path) -> None:
        base = tmp_path / "img.txt"
        base.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        text = build_preannotation_labels([], base)
        assert text == "0 0.5 0.5 0.2 0.2\n"

    def test_candidates_appended_after_base(self, tmp_path: Path) -> None:
        base = tmp_path / "img.txt"
        base.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        detections = [{"class_id": 2, "conf": 0.9, "bbox_xywhn": [0.1, 0.1, 0.05, 0.05]}]
        text = build_preannotation_labels(detections, base)
        lines = text.splitlines()
        assert lines[0] == "0 0.5 0.5 0.2 0.2"
        assert lines[-1] == "2 0.100000 0.100000 0.050000 0.050000"

    def test_candidates_only_sorted_by_class_then_bbox(self, tmp_path: Path) -> None:
        detections = [
            {"class_id": 3, "conf": 0.5, "bbox_xywhn": [0.9, 0.9, 0.1, 0.1]},
            {"class_id": 2, "conf": 0.9, "bbox_xywhn": [0.1, 0.1, 0.05, 0.05]},
        ]
        text = build_preannotation_labels(detections, tmp_path / "absent.txt")
        lines = text.splitlines()
        assert lines[0].startswith("2 ")
        assert lines[1].startswith("3 ")


class TestBuildPreannotationZip:
    def _candidate_images(self) -> dict[str, dict]:
        return {
            "a.jpg": {
                "detections": [{"class_id": 2, "conf": 0.9, "bbox_xywhn": [0.1, 0.1, 0.05, 0.05]}]
            },
            "b.jpg": {
                "detections": [{"class_id": 3, "conf": 0.7, "bbox_xywhn": [0.2, 0.2, 0.1, 0.1]}]
            },
        }

    def test_zip_layout(self, tmp_path: Path) -> None:
        merged_labels = tmp_path / "labels"
        merged_labels.mkdir()
        out_zip = tmp_path / "batch" / "preannotations.zip"

        sha = build_preannotation_zip(
            batch_images=["a.jpg", "b.jpg"],
            candidate_images=self._candidate_images(),
            merged_labels_dir=merged_labels,
            class_names_by_id=_NAMES_BY_ID,
            out_zip=out_zip,
        )
        assert out_zip.exists()
        assert len(sha) == 64

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())
            assert names == {
                "obj.names",
                "obj.data",
                "train.txt",
                "obj_train_data/a.txt",
                "obj_train_data/b.txt",
            }
            assert zf.read("obj.names").decode("utf-8").splitlines() == [
                "person",
                "face",
                "charger",
                "wire",
            ]
            assert "classes = 4" in zf.read("obj.data").decode("utf-8")
            train = zf.read("train.txt").decode("utf-8")
            assert "data/obj_train_data/a.jpg" in train
            assert "data/obj_train_data/b.jpg" in train
            assert zf.read("obj_train_data/a.txt").decode("utf-8").strip() == (
                "2 0.100000 0.100000 0.050000 0.050000"
            )

    def test_base_labels_included_in_union(self, tmp_path: Path) -> None:
        merged_labels = tmp_path / "labels"
        merged_labels.mkdir()
        (merged_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        out_zip = tmp_path / "batch" / "preannotations.zip"

        build_preannotation_zip(
            batch_images=["a.jpg"],
            candidate_images={"a.jpg": self._candidate_images()["a.jpg"]},
            merged_labels_dir=merged_labels,
            class_names_by_id=_NAMES_BY_ID,
            out_zip=out_zip,
        )
        with zipfile.ZipFile(out_zip) as zf:
            text = zf.read("obj_train_data/a.txt").decode("utf-8")
            assert "0 0.5 0.5 0.2 0.2" in text
            assert "2 0.100000 0.100000 0.050000 0.050000" in text

    def test_deterministic_across_rebuilds(self, tmp_path: Path) -> None:
        merged_labels = tmp_path / "labels"
        merged_labels.mkdir()
        out1 = tmp_path / "run1.zip"
        out2 = tmp_path / "run2.zip"
        images = self._candidate_images()

        sha1 = build_preannotation_zip(
            ["a.jpg", "b.jpg"], images, merged_labels, _NAMES_BY_ID, out1
        )
        sha2 = build_preannotation_zip(
            ["a.jpg", "b.jpg"], images, merged_labels, _NAMES_BY_ID, out2
        )
        assert sha1 == sha2
        assert out1.read_bytes() == out2.read_bytes()

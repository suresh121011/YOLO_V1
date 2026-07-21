"""Unit tests for the cross_dataset backend (L3, ADR-P5-08) — no ML, reads
merge.py's link file and replays it as candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.annotation.backends.cross_dataset import CrossDatasetBackend
from src.dataset.annotation.base import BackendConfig

pytestmark = pytest.mark.unit

_IDS_BY_NAME = {"person": 0, "charger": 1, "wire": 2}


def _config(links_path: Path) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "cross_dataset",
        {
            "enabled": True,
            "weights": "",
            "weights_sha256": "",
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": {"charger": ["cross_dataset"], "wire": ["cross_dataset"]},
            "thresholds": {"default": 0.25},
            "links_path": str(links_path),
        },
    )


def _write_links(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestCrossDatasetBackend:
    def test_linked_boxes_emitted_for_targeted_classes(self, tmp_path: Path) -> None:
        links_path = tmp_path / "cross_dataset_links.json"
        _write_links(
            links_path,
            {"alpha_a1.jpg": [{"source": "beta", "boxes": [[1, 0.5, 0.5, 0.2, 0.2]]}]},
        )
        backend = CrossDatasetBackend()
        backend.load(_config(links_path), "cpu", _IDS_BY_NAME)
        detections = backend.annotate(Path("alpha_a1.jpg"), target_class_ids=(1, 2))
        assert len(detections) == 1
        assert detections[0].class_id == 1
        assert detections[0].conf == 1.0
        assert detections[0].bbox_xywhn == (0.5, 0.5, 0.2, 0.2)
        assert detections[0].origin == "cross_dataset:beta"

    def test_box_filtered_out_when_class_not_targeted(self, tmp_path: Path) -> None:
        links_path = tmp_path / "cross_dataset_links.json"
        _write_links(
            links_path,
            {"alpha_a1.jpg": [{"source": "beta", "boxes": [[1, 0.5, 0.5, 0.2, 0.2]]}]},
        )
        backend = CrossDatasetBackend()
        backend.load(_config(links_path), "cpu", _IDS_BY_NAME)
        detections = backend.annotate(Path("alpha_a1.jpg"), target_class_ids=(2,))  # not 1
        assert detections == []

    def test_image_with_no_link_returns_empty(self, tmp_path: Path) -> None:
        links_path = tmp_path / "cross_dataset_links.json"
        _write_links(links_path, {})
        backend = CrossDatasetBackend()
        backend.load(_config(links_path), "cpu", _IDS_BY_NAME)
        assert backend.annotate(Path("no_link.jpg"), target_class_ids=(1, 2)) == []

    def test_missing_links_file_is_not_an_error(self, tmp_path: Path) -> None:
        """The first merge before any duplicates have ever been seen —
        cross_dataset_links.json may not exist yet."""
        backend = CrossDatasetBackend()
        backend.load(_config(tmp_path / "nonexistent.json"), "cpu", _IDS_BY_NAME)
        assert backend.annotate(Path("x.jpg"), target_class_ids=(1,)) == []

    def test_multiple_linked_sources_all_emitted(self, tmp_path: Path) -> None:
        links_path = tmp_path / "cross_dataset_links.json"
        _write_links(
            links_path,
            {
                "alpha_a1.jpg": [
                    {"source": "beta", "boxes": [[1, 0.2, 0.2, 0.1, 0.1]]},
                    {"source": "gamma", "boxes": [[2, 0.8, 0.8, 0.1, 0.1]]},
                ]
            },
        )
        backend = CrossDatasetBackend()
        backend.load(_config(links_path), "cpu", _IDS_BY_NAME)
        detections = backend.annotate(Path("alpha_a1.jpg"), target_class_ids=(1, 2))
        assert {d.origin for d in detections} == {"cross_dataset:beta", "cross_dataset:gamma"}

    def test_annotate_before_load_raises(self) -> None:
        backend = CrossDatasetBackend()
        with pytest.raises(RuntimeError):
            backend.annotate(Path("x.jpg"), target_class_ids=(1,))

    def test_fingerprint_has_no_weights(self, tmp_path: Path) -> None:
        links_path = tmp_path / "cross_dataset_links.json"
        _write_links(links_path, {})
        backend = CrossDatasetBackend()
        backend.load(_config(links_path), "cpu", _IDS_BY_NAME)
        fp = backend.fingerprint()
        assert fp.backend == "cross_dataset"
        assert fp.weights_sha256 == ""
        assert fp.weights_path == ""

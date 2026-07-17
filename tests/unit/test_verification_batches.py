"""Unit tests for verification batch planning, manifests, and lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.batches import (
    BATCH_MANIFEST_FILENAME,
    VerificationBatchManifest,
    already_batched_images,
    build_batch_manifests,
    image_expected_gain,
    next_batch_id,
    plan_batches,
)

pytestmark = pytest.mark.unit

_NAMES_BY_ID = {0: "person", 1: "face", 10: "charger", 11: "wire"}
_PRIORITY = frozenset({"charger", "wire"})


def _det(class_id: int, conf: float) -> dict:
    return {"class_id": class_id, "conf": conf, "bbox_xywhn": [0.5, 0.5, 0.1, 0.1]}


def _candidates(images: dict[str, list[dict]], run_id: str = "run1") -> dict:
    return {
        "run_id": run_id,
        "images": {name: {"detections": dets} for name, dets in images.items()},
    }


class TestVerificationBatchManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = VerificationBatchManifest(
            batch_id="vb001_yolo_world",
            candidate_run={"backend": "yolo_world", "run_id": "r1", "candidates_sha256": "abc"},
            target_classes=["charger", "wire"],
            images=["a.jpg", "b.jpg"],
            status="created",
            expected_gain=1.5,
        )
        path = tmp_path / BATCH_MANIFEST_FILENAME
        manifest.save(path)
        loaded = VerificationBatchManifest.load(path)
        assert loaded == manifest

    def test_unknown_keys_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / BATCH_MANIFEST_FILENAME
        path.write_text('{"batch_id": "vb001_x", "future_field": 42}', encoding="utf-8")
        loaded = VerificationBatchManifest.load(path)
        assert loaded.batch_id == "vb001_x"


class TestImageExpectedGain:
    def test_priority_class_weighted_double(self) -> None:
        priority_only = image_expected_gain([_det(10, 0.5)], _NAMES_BY_ID, _PRIORITY)
        non_priority = image_expected_gain([_det(0, 0.5)], _NAMES_BY_ID, _PRIORITY)
        assert priority_only == pytest.approx(2 * non_priority)

    def test_sums_across_detections(self) -> None:
        gain = image_expected_gain([_det(0, 0.5), _det(1, 0.5)], _NAMES_BY_ID, _PRIORITY)
        assert gain == pytest.approx(1.0)


class TestPlanBatches:
    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(AnnotationError, match="positive"):
            plan_batches(_candidates({}), frozenset(), _NAMES_BY_ID, _PRIORITY, 0)

    def test_skips_images_with_no_detections(self) -> None:
        candidates = _candidates({"a.jpg": [], "b.jpg": [_det(10, 0.9)]})
        drafts = plan_batches(candidates, frozenset(), _NAMES_BY_ID, _PRIORITY, batch_size=200)
        assert len(drafts) == 1
        assert drafts[0].images == ("b.jpg",)

    def test_skips_already_batched_images(self) -> None:
        candidates = _candidates({"a.jpg": [_det(10, 0.9)], "b.jpg": [_det(11, 0.8)]})
        drafts = plan_batches(
            candidates, frozenset({"a.jpg"}), _NAMES_BY_ID, _PRIORITY, batch_size=200
        )
        assert len(drafts) == 1
        assert drafts[0].images == ("b.jpg",)

    def test_ranked_by_descending_gain_then_chunked(self) -> None:
        candidates = _candidates(
            {
                "low.jpg": [_det(0, 0.1)],
                "high.jpg": [_det(10, 0.9)],
                "mid.jpg": [_det(0, 0.5)],
            }
        )
        drafts = plan_batches(candidates, frozenset(), _NAMES_BY_ID, _PRIORITY, batch_size=2)
        assert len(drafts) == 2
        # First chunk = the two highest-gain images (sorted alphabetically within chunk).
        assert set(drafts[0].images) == {"high.jpg", "mid.jpg"}
        assert drafts[1].images == ("low.jpg",)

    def test_target_classes_derived_from_chunk_detections(self) -> None:
        candidates = _candidates({"a.jpg": [_det(10, 0.9), _det(11, 0.8)]})
        drafts = plan_batches(candidates, frozenset(), _NAMES_BY_ID, _PRIORITY, batch_size=200)
        assert drafts[0].target_classes == ("charger", "wire")

    def test_deterministic_tiebreak_by_filename(self) -> None:
        candidates = _candidates({"z.jpg": [_det(0, 0.5)], "a.jpg": [_det(0, 0.5)]})
        drafts = plan_batches(candidates, frozenset(), _NAMES_BY_ID, _PRIORITY, batch_size=200)
        assert drafts[0].images == ("a.jpg", "z.jpg")


class TestAlreadyBatchedImages:
    def _write_batch(
        self, batches_root: Path, batch_id: str, images: list[str], status: str
    ) -> None:
        manifest = VerificationBatchManifest(batch_id=batch_id, images=images, status=status)
        manifest.save(batches_root / batch_id / BATCH_MANIFEST_FILENAME)

    def test_empty_when_no_batches(self, tmp_path: Path) -> None:
        assert already_batched_images(tmp_path / "batches") == frozenset()

    def test_active_statuses_claim_images(self, tmp_path: Path) -> None:
        root = tmp_path / "batches"
        self._write_batch(root, "vb001_yolo_world", ["a.jpg"], status="created")
        self._write_batch(root, "vb002_yolo_world", ["b.jpg"], status="staged")
        assert already_batched_images(root) == frozenset({"a.jpg", "b.jpg"})

    def test_imported_status_releases_claim(self, tmp_path: Path) -> None:
        root = tmp_path / "batches"
        self._write_batch(root, "vb001_yolo_world", ["a.jpg"], status="imported")
        assert already_batched_images(root) == frozenset()


class TestNextBatchId:
    def test_first_batch_is_001(self, tmp_path: Path) -> None:
        assert next_batch_id(tmp_path / "batches", "yolo_world") == "vb001_yolo_world"

    def test_increments_past_existing(self, tmp_path: Path) -> None:
        root = tmp_path / "batches"
        (root / "vb001_yolo_world").mkdir(parents=True)
        (root / "vb003_yolo_world").mkdir(parents=True)
        assert next_batch_id(root, "yolo_world") == "vb004_yolo_world"

    def test_ignores_non_matching_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "batches"
        root.mkdir(parents=True)
        (root / "cvat_labels.json").write_text("[]", encoding="utf-8")
        (root / "not_a_batch").mkdir()
        assert next_batch_id(root, "yolo_world") == "vb001_yolo_world"


class TestBuildBatchManifests:
    def test_assigns_sequential_ids_and_provenance(self, tmp_path: Path) -> None:
        candidates = _candidates(
            {"a.jpg": [_det(10, 0.9)], "b.jpg": [_det(11, 0.8)]}, run_id="run42"
        )
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="yolo_world",
            candidates_sha256="deadbeef",
            batches_root=tmp_path / "batches",
            class_names_by_id=_NAMES_BY_ID,
            priority_classes=_PRIORITY,
            batch_size=1,
        )
        assert [m.batch_id for m in manifests] == ["vb001_yolo_world", "vb002_yolo_world"]
        for m in manifests:
            assert m.candidate_run == {
                "backend": "yolo_world",
                "run_id": "run42",
                "candidates_sha256": "deadbeef",
            }
            assert m.status == "created"

    def test_continues_numbering_past_existing_batches(self, tmp_path: Path) -> None:
        batches_root = tmp_path / "batches"
        (batches_root / "vb001_yolo_world").mkdir(parents=True)
        candidates = _candidates({"a.jpg": [_det(10, 0.9)]})
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="yolo_world",
            candidates_sha256="x",
            batches_root=batches_root,
            class_names_by_id=_NAMES_BY_ID,
            priority_classes=_PRIORITY,
            batch_size=200,
        )
        assert manifests[0].batch_id == "vb002_yolo_world"

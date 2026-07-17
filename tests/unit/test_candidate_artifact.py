"""Unit tests for the candidate-label artifact (build/save/load/validate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.base import AnnotationError, Detection, ModelFingerprint
from src.dataset.annotation.candidates import (
    CANDIDATES_SCHEMA_VERSION,
    ImageCandidates,
    build_candidates_artifact,
    compute_run_id,
    load_candidates,
    save_candidates,
    validate_candidates,
)

pytestmark = pytest.mark.unit

_NC = 23
_NAMES = {10: "charger", 11: "wire", 14: "cupboard"}
_TAXO_FP = "sha256:test-taxonomy"


def _model_fp() -> ModelFingerprint:
    return ModelFingerprint(
        backend="fake",
        weights_path="",
        weights_sha256="",
        library_versions={"ultralytics": "8.3.0"},
        device="cpu",
        prompt_fingerprint="sha256:abcdef1234567890",
    )


def _artifact(**image_overrides: ImageCandidates) -> dict[str, Any]:
    images = {
        "coco_000001.jpg": ImageCandidates(
            targeted_class_ids=(10, 11),
            detections=(
                Detection(10, 0.9, (0.5, 0.5, 0.2, 0.2), origin="fake"),
                Detection(11, 0.8, (0.3, 0.3, 0.1, 0.1), origin="fake"),
            ),
        ),
        "coco_000002.jpg": ImageCandidates(targeted_class_ids=(14,), detections=()),
    }
    images.update(image_overrides)
    return build_candidates_artifact(
        backend="fake",
        model=_model_fp(),
        taxonomy_fp=_TAXO_FP,
        inputs={"images_root": "data/merged/images", "merged_manifest_sha256": "x"},
        determinism={"seed": 0, "deterministic_algorithms": True, "image_order": "sorted"},
        images=images,
        runtime_s=1.234,
        class_names_by_id=_NAMES,
        git_commit="abc1234",
    )


class TestBuild:
    def test_run_id_is_deterministic_no_wall_clock(self) -> None:
        assert compute_run_id("fake", "sha256:abcdef1234567890", "abc1234") == (
            "fake_abc1234_abcdef12"
        )
        assert _artifact()["run_id"] == "fake_abc1234_abcdef12"

    def test_stats_computed(self) -> None:
        artifact = _artifact()
        assert artifact["stats"]["images_processed"] == 2
        assert artifact["stats"]["detections_total"] == 2
        assert artifact["stats"]["per_class"] == {"charger": 1, "wire": 1}
        assert artifact["stats"]["runtime_s"] == 1.234

    def test_images_sorted_for_stable_output(self) -> None:
        artifact = _artifact()
        assert list(artifact["images"]) == sorted(artifact["images"])


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        artifact = _artifact()
        path = tmp_path / "candidates.json"
        save_candidates(artifact, path)
        loaded = load_candidates(path)
        assert loaded["run_id"] == artifact["run_id"]
        assert loaded["images"].keys() == artifact["images"].keys()

    def test_unknown_keys_tolerated(self, tmp_path: Path) -> None:
        artifact = _artifact()
        artifact["future_field"] = {"anything": 1}
        path = tmp_path / "candidates.json"
        save_candidates(artifact, path)
        assert load_candidates(path)["future_field"] == {"anything": 1}

    def test_duplicate_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.json"
        path.write_text(
            f'{{"schema_version": {CANDIDATES_SCHEMA_VERSION}, "run_id": "a", "run_id": "b"}}',
            encoding="utf-8",
        )
        with pytest.raises(AnnotationError, match="Duplicate JSON key"):
            load_candidates(path)

    def test_unsupported_schema_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.json"
        path.write_text('{"schema_version": 99}', encoding="utf-8")
        with pytest.raises(AnnotationError, match="schema_version"):
            load_candidates(path)

    def test_invalid_json_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(AnnotationError, match="Invalid candidates JSON"):
            load_candidates(path)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_candidates(tmp_path / "absent.json")


class TestValidate:
    def test_clean_artifact(self) -> None:
        assert validate_candidates(_artifact(), _NC, _TAXO_FP) == []

    def test_taxonomy_drift_flagged(self) -> None:
        problems = validate_candidates(_artifact(), _NC, "sha256:other")
        assert any("taxonomy fingerprint drift" in p for p in problems)

    def test_missing_required_key(self) -> None:
        artifact = _artifact()
        del artifact["stats"]
        assert any("'stats'" in p for p in validate_candidates(artifact, _NC))

    def test_class_id_out_of_range(self) -> None:
        artifact = _artifact(
            bad=ImageCandidates((99,), (Detection(99, 0.5, (0.5, 0.5, 0.1, 0.1)),))
        )
        problems = validate_candidates(artifact, _NC)
        assert any("out of range" in p for p in problems)

    def test_untargeted_detection_flagged(self) -> None:
        artifact = _artifact(
            bad=ImageCandidates((10,), (Detection(11, 0.5, (0.5, 0.5, 0.1, 0.1)),))
        )
        problems = validate_candidates(artifact, _NC)
        assert any("not in the image's targeted set" in p for p in problems)

    def test_conf_out_of_range(self) -> None:
        artifact = _artifact(
            bad=ImageCandidates((10,), (Detection(10, 1.5, (0.5, 0.5, 0.1, 0.1)),))
        )
        assert any("conf" in p for p in validate_candidates(artifact, _NC))

    def test_degenerate_and_out_of_bounds_boxes(self) -> None:
        artifact = _artifact(
            zero=ImageCandidates((10,), (Detection(10, 0.5, (0.5, 0.5, 0.0, 0.1)),)),
            outside=ImageCandidates((11,), (Detection(11, 0.5, (1.5, 0.5, 0.1, 0.1)),)),
        )
        problems = validate_candidates(artifact, _NC)
        assert any("degenerate box" in p for p in problems)
        assert any("outside [0, 1]" in p for p in problems)

    def test_stats_mismatch_flagged(self) -> None:
        artifact = _artifact()
        artifact["stats"]["detections_total"] = 99
        assert any("detections_total" in p for p in validate_candidates(artifact, _NC))


class TestLedgerBootstrap:
    def test_committed_empty_ledger_is_valid_minimal_json(self) -> None:
        ledger_path = Path("data/annotation/verification_ledger.json")
        raw = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 1
        assert raw["entries"] == {}

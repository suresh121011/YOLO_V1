"""Unit tests for scripts.dataset.12_auto_annotate.run_backend — the M1
candidate-generation orchestration (targeting -> annotate -> filter ->
artifact), previously untested (final-audit finding). Uses FakeAnnotator
(offline, model-free) — never a real GPU backend.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.dataset.annotation.base import BackendConfig
from src.dataset.annotation.candidates import CANDIDATES_FILENAME, load_candidates
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from unit.annotation_fakes import FakeAnnotator  # noqa: F401 — registers "fake"

auto_annotate = importlib.import_module("scripts.dataset.12_auto_annotate")

pytestmark = pytest.mark.unit

_CLASS_NAMES_BY_ID = {0: "person", 1: "charger", 2: "wire"}
_IDS_BY_NAME = {v: k for k, v in _CLASS_NAMES_BY_ID.items()}
_TAXONOMY_FP = "sha256:test-fixture"


def _backend_config(thresholds: dict[str, float]) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "fake",
        {
            "enabled": True,
            "weights": "",
            "weights_sha256": "",
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": {
                "person": ["a person"],
                "charger": ["phone charger"],
                "wire": ["electrical wire"],
            },
            "thresholds": thresholds,
        },
    )


def _setup_merged_dir(tmp_path: Path) -> Path:
    merged_dir = tmp_path / "merged"
    (merged_dir / "images").mkdir(parents=True)
    (merged_dir / "images" / "img1.jpg").write_bytes(b"")
    manifest = MergedManifest(
        image_provenance={"img1.jpg": "coco"}, label_completeness={"coco": []}
    )
    manifest.save(merged_dir / MERGED_MANIFEST_FILENAME)
    return merged_dir


def _run(tmp_path: Path, thresholds: dict[str, float]) -> dict:
    merged_dir = _setup_merged_dir(tmp_path)
    backend_cfg = _backend_config(thresholds)
    out_path = auto_annotate.run_backend(
        backend_name="fake",
        backend_cfg=backend_cfg,
        refinement_cfg={"enabled": False},
        device="cpu",
        merged_dir=merged_dir,
        manifest=MergedManifest.load(merged_dir / MERGED_MANIFEST_FILENAME),
        policies={"coco": "trusted_list"},
        ids_by_name=_IDS_BY_NAME,
        names_by_id=_CLASS_NAMES_BY_ID,
        taxonomy_fp=_TAXONOMY_FP,
        verified_cells={},
        ledger_path=tmp_path / "no_ledger.json",
        output_root=tmp_path / "candidates",
    )
    assert out_path == tmp_path / "candidates" / "fake" / CANDIDATES_FILENAME
    return load_candidates(out_path)


class TestPerClassThresholdFiltering:
    """FakeAnnotator emits conf = 0.9 - 0.1*i for the i-th targeted class id
    (sorted: person=0 -> 0.9, charger=1 -> 0.8, wire=2 -> 0.7)."""

    def test_default_threshold_keeps_all_three(self, tmp_path: Path) -> None:
        artifact = _run(tmp_path, {"default": 0.25})
        classes = {d["class_id"] for d in artifact["images"]["img1.jpg"]["detections"]}
        assert classes == {0, 1, 2}

    def test_per_class_threshold_above_emitted_confidence_filters_it_out(
        self, tmp_path: Path
    ) -> None:
        """wire's real conf (0.7) is below an 0.85 per-class threshold — it
        must never reach the candidate artifact, even though it cleared the
        low conf_floor and the backend returned it."""
        artifact = _run(tmp_path, {"default": 0.25, "wire": 0.85})
        classes = {d["class_id"] for d in artifact["images"]["img1.jpg"]["detections"]}
        assert classes == {0, 1}  # person, charger — wire filtered
        assert 2 not in classes

    def test_per_class_threshold_below_emitted_confidence_keeps_it(self, tmp_path: Path) -> None:
        artifact = _run(tmp_path, {"default": 0.25, "wire": 0.5})
        classes = {d["class_id"] for d in artifact["images"]["img1.jpg"]["detections"]}
        assert 2 in classes

    def test_threshold_at_exactly_the_emitted_confidence_keeps_it(self, tmp_path: Path) -> None:
        """>= threshold, not >."""
        artifact = _run(tmp_path, {"default": 0.25, "person": 0.9})
        classes = {d["class_id"] for d in artifact["images"]["img1.jpg"]["detections"]}
        assert 0 in classes

    def test_all_classes_filtered_below_threshold_yields_empty_detections(
        self, tmp_path: Path
    ) -> None:
        artifact = _run(tmp_path, {"default": 0.95})
        assert artifact["images"]["img1.jpg"]["detections"] == []


class TestRunBackendArtifactShape:
    def test_written_artifact_is_valid_json_with_taxonomy_fingerprint(self, tmp_path: Path) -> None:
        merged_dir = _setup_merged_dir(tmp_path)
        backend_cfg = _backend_config({"default": 0.25})
        out_path = auto_annotate.run_backend(
            backend_name="fake",
            backend_cfg=backend_cfg,
            refinement_cfg={"enabled": False},
            device="cpu",
            merged_dir=merged_dir,
            manifest=MergedManifest.load(merged_dir / MERGED_MANIFEST_FILENAME),
            policies={"coco": "trusted_list"},
            ids_by_name=_IDS_BY_NAME,
            names_by_id=_CLASS_NAMES_BY_ID,
            taxonomy_fp=_TAXONOMY_FP,
            verified_cells={},
            ledger_path=tmp_path / "no_ledger.json",
            output_root=tmp_path / "candidates",
        )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["taxonomy_fingerprint"] == _TAXONOMY_FP
        assert payload["backend"] == "fake"

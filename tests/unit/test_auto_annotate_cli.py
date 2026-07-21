"""Unit tests for scripts.dataset.12_auto_annotate.verify_determinism — the
``--verify-determinism`` re-run-and-diff routine (previously untested; the
final-audit Fix-7 test-coverage gap on already-correct code).

Uses ``FakeAnnotator`` (deterministic, model-free, offline) exactly like
``test_run_backend.py`` — never a real GPU backend. Because the Fake is
bit-deterministic, a clean re-run must diff empty, and a tampered artifact
must be detected.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.dataset.annotation.base import BackendConfig
from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from unit.annotation_fakes import FakeAnnotator  # noqa: F401 — import registers "fake"

auto_annotate = importlib.import_module("scripts.dataset.12_auto_annotate")

pytestmark = pytest.mark.unit

_CLASS_NAMES_BY_ID = {0: "person", 1: "charger", 2: "wire"}
_IDS_BY_NAME = {v: k for k, v in _CLASS_NAMES_BY_ID.items()}
_TAXONOMY_FP = "sha256:test-fixture"


def _backend_config() -> BackendConfig:
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
            "thresholds": {"default": 0.25},
        },
    )


def _write_artifact(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a real candidate artifact with the Fake backend."""
    merged_dir = tmp_path / "merged"
    (merged_dir / "images").mkdir(parents=True)
    (merged_dir / "images" / "img1.jpg").write_bytes(b"")
    MergedManifest(image_provenance={"img1.jpg": "coco"}, label_completeness={"coco": []}).save(
        merged_dir / MERGED_MANIFEST_FILENAME
    )
    out_path = auto_annotate.run_backend(
        backend_name="fake",
        backend_cfg=_backend_config(),
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
    return merged_dir, out_path


def _verify(merged_dir: Path, out_path: Path) -> list[str]:
    return auto_annotate.verify_determinism(
        out_path,
        "fake",
        _backend_config(),
        {"enabled": False},
        "cpu",
        merged_dir,
        _IDS_BY_NAME,
    )


class TestVerifyDeterminism:
    def test_clean_rerun_reports_no_mismatch(self, tmp_path: Path) -> None:
        merged_dir, out_path = _write_artifact(tmp_path)
        assert _verify(merged_dir, out_path) == []

    def test_tampered_confidence_is_detected(self, tmp_path: Path) -> None:
        merged_dir, out_path = _write_artifact(tmp_path)
        artifact = json.loads(out_path.read_text(encoding="utf-8"))
        # Corrupt a recorded confidence: the deterministic fresh re-run (which
        # the Fake reproduces exactly) no longer matches the stored record.
        artifact["images"]["img1.jpg"]["detections"][0]["conf"] = 0.123
        out_path.write_text(json.dumps(artifact), encoding="utf-8")
        mismatches = _verify(merged_dir, out_path)
        assert len(mismatches) == 1
        assert "img1.jpg" in mismatches[0]

    def test_extra_recorded_detection_is_detected(self, tmp_path: Path) -> None:
        merged_dir, out_path = _write_artifact(tmp_path)
        artifact = json.loads(out_path.read_text(encoding="utf-8"))
        # Inject a phantom detection the fresh run will never reproduce.
        artifact["images"]["img1.jpg"]["detections"].append(
            {
                "class_id": 0,
                "conf": 0.5,
                "bbox_xywhn": [0.1, 0.1, 0.1, 0.1],
                "refined": False,
                "origin": "fake",
            }
        )
        out_path.write_text(json.dumps(artifact), encoding="utf-8")
        mismatches = _verify(merged_dir, out_path)
        assert len(mismatches) == 1

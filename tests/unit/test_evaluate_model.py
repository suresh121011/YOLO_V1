"""Unit tests for scripts.training.evaluate_model — CLI logic with mocked evaluation.

Never mocks ultralytics.YOLO itself (matches the house pattern in
test_mitigation_evaluation.py) — run_single_eval is monkeypatched at the
module boundary so these tests exercise only evaluate_model.py's own
argument-to-spec wiring and output-path selection.
"""

from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

evaluate_model = importlib.import_module("scripts.training.evaluate_model")

pytestmark = pytest.mark.unit


def _fake_summary(**overrides: Any) -> dict[str, Any]:
    """A fresh summary dict each call — evaluate_model.run() mutates its
    input in place (merges wet_floor_checkpoint), so tests must never share
    one dict instance across calls."""
    base: dict[str, Any] = {
        "spec": {"weights": "models/yolo11n/weights/best.pt", "label": "val"},
        "aggregate": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "mAP50": 0.72, "mAP50_95": 0.5},
        "per_class": [],
        "confusion_matrix": "confusion_matrix_val.json",
    }
    base.update(overrides)
    return base


def _args(tmp_path: Path, **overrides: Any) -> Namespace:
    base = {
        "weights": Path("models/yolo11n/weights/best.pt"),
        "split": "val",
        "data": None,
        "label": None,
        "imgsz": 640,
        "device": "cpu",
        "seed": 42,
        "out_dir": tmp_path / "out",
        "eval_report_out": tmp_path / "eval_report.json",
    }
    base.update(overrides)
    return Namespace(**base)


class TestSplitToDataYamlWiring:
    def test_val_split_uses_default_data_yaml_and_ultralytics_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_single_eval(spec: Any, out_dir: Path) -> dict[str, Any]:
            captured["spec"] = spec
            return _fake_summary()

        monkeypatch.setattr(evaluate_model, "run_single_eval", fake_run_single_eval)
        assert evaluate_model.run(_args(tmp_path)) == 0
        assert captured["spec"].data_yaml == evaluate_model.DEFAULT_DATA_YAML
        assert captured["spec"].split == "val"

    def test_eval_split_maps_to_eval_data_yaml_and_ultralytics_test_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_single_eval(spec: Any, out_dir: Path) -> dict[str, Any]:
            captured["spec"] = spec
            return _fake_summary()

        monkeypatch.setattr(evaluate_model, "run_single_eval", fake_run_single_eval)
        assert evaluate_model.run(_args(tmp_path, split="eval")) == 0
        assert captured["spec"].data_yaml == evaluate_model.EVAL_SPLIT_DATA_YAML
        assert captured["spec"].split == "test"

    def test_explicit_data_override_wins_over_split_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        override = Path("configs/custom_data.yaml")

        def fake_run_single_eval(spec: Any, out_dir: Path) -> dict[str, Any]:
            captured["spec"] = spec
            return _fake_summary()

        monkeypatch.setattr(evaluate_model, "run_single_eval", fake_run_single_eval)
        assert evaluate_model.run(_args(tmp_path, split="eval", data=override)) == 0
        assert captured["spec"].data_yaml == override


class TestEvalReportOutput:
    def test_eval_split_writes_flat_eval_report_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            evaluate_model, "run_single_eval", lambda spec, out_dir: _fake_summary()
        )
        args = _args(tmp_path, split="eval")
        assert evaluate_model.run(args) == 0
        payload = json.loads(args.eval_report_out.read_text(encoding="utf-8"))
        assert payload["aggregate"]["mAP50"] == 0.72

    def test_non_eval_split_does_not_write_eval_report_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            evaluate_model, "run_single_eval", lambda spec, out_dir: _fake_summary()
        )
        args = _args(tmp_path, split="val")
        assert evaluate_model.run(args) == 0
        assert not args.eval_report_out.exists()


class TestWetFloorCheckpointWiring:
    def test_low_wet_floor_ap50_is_flagged_in_written_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            evaluate_model,
            "run_single_eval",
            lambda spec, out_dir: _fake_summary(per_class=[{"class": "wet_floor", "mAP50": 0.10}]),
        )
        args = _args(tmp_path, split="eval")
        assert evaluate_model.run(args) == 0
        payload = json.loads(args.eval_report_out.read_text(encoding="utf-8"))
        assert payload["wet_floor_checkpoint"]["available"] is True
        assert payload["wet_floor_checkpoint"]["reopen_demotion"] is True

    def test_healthy_wet_floor_ap50_does_not_reopen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            evaluate_model,
            "run_single_eval",
            lambda spec, out_dir: _fake_summary(per_class=[{"class": "wet_floor", "mAP50": 0.50}]),
        )
        args = _args(tmp_path, split="eval")
        assert evaluate_model.run(args) == 0
        payload = json.loads(args.eval_report_out.read_text(encoding="utf-8"))
        assert payload["wet_floor_checkpoint"]["reopen_demotion"] is False

    def test_no_wet_floor_ground_truth_is_unavailable_not_a_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            evaluate_model, "run_single_eval", lambda spec, out_dir: _fake_summary()
        )
        args = _args(tmp_path, split="eval")
        assert evaluate_model.run(args) == 0
        payload = json.loads(args.eval_report_out.read_text(encoding="utf-8"))
        assert payload["wet_floor_checkpoint"]["available"] is False


class TestErrorHandling:
    def test_missing_checkpoint_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_not_found(spec: Any, out_dir: Path) -> dict[str, Any]:
            raise FileNotFoundError("Checkpoint not found: nope.pt")

        monkeypatch.setattr(evaluate_model, "run_single_eval", raise_not_found)
        assert evaluate_model.run(_args(tmp_path)) == 1


class TestEvalDataYamlTaxonomy:
    def test_eval_data_yaml_matches_data_yaml_taxonomy(self) -> None:
        data_cfg = load_data_config(Path("configs/data.yaml"))
        eval_cfg = load_data_config(Path("configs/eval_data.yaml"))
        assert eval_cfg["nc"] == data_cfg["nc"]
        assert get_class_names_from_data_yaml(eval_cfg) == get_class_names_from_data_yaml(data_cfg)

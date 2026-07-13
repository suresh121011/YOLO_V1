"""
Unit tests for scripts.training.train_yolo — config loading and argument logic.

Note: Actual YOLO training is not executed in unit tests.
The model.train() call is mocked to avoid requiring GPU, weights files,
or a full dataset.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.utils.config_helpers import load_training_config, resolve_device

# ─── Training config loading ──────────────────────────────────────────────────


@pytest.mark.unit
class TestLoadTrainingConfig:
    def test_loads_valid_config(self, tmp_path: Path) -> None:
        config = {
            "model": {"base": "yolo11n.pt", "device": "cpu"},
            "training": {"epochs": 50, "batch": 8},
            "output": {"project": "models", "name": "test"},
        }
        cfg_path = tmp_path / "train.yaml"
        cfg_path.write_text(yaml.dump(config))

        loaded = load_training_config(cfg_path)
        assert loaded["model"]["base"] == "yolo11n.pt"
        assert loaded["training"]["epochs"] == 50

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config(tmp_path / "nonexistent.yaml")

    def test_missing_section_warns_not_raises(self, tmp_path: Path) -> None:
        """Config missing 'output' section should warn but not raise."""
        config = {"model": {"base": "yolo11n.pt"}, "training": {"epochs": 10}}
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text(yaml.dump(config))

        # Should not raise
        loaded = load_training_config(cfg_path)
        assert "model" in loaded

    def test_yolo11n_config_loads(self) -> None:
        """Validate the actual project training config is parseable."""
        config_path = Path("configs/training/yolo11n_config.yaml")
        if not config_path.exists():
            pytest.skip("configs/training/yolo11n_config.yaml not found")

        loaded = load_training_config(config_path)
        assert "model" in loaded
        assert "training" in loaded
        assert loaded["training"]["epochs"] > 0


# ─── Device resolution ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestResolveDevice:
    def test_cpu_passthrough(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert resolve_device("cuda") == "cuda"

    def test_mps_passthrough(self) -> None:
        assert resolve_device("mps") == "mps"

    def test_auto_returns_valid_device(self) -> None:
        device = resolve_device("auto")
        assert device in ("cuda", "mps", "cpu")

    def test_auto_falls_back_to_cpu_without_torch(self) -> None:
        """When torch is not importable, 'auto' should fall back to 'cpu'."""
        import sys

        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            result = resolve_device("auto")
            assert result == "cpu"
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            elif "torch" in sys.modules:
                del sys.modules["torch"]


# ─── Extract metrics ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractMetrics:
    def test_extracts_standard_metrics(self) -> None:
        from scripts.training.train_yolo import extract_metrics

        mock_results = MagicMock()
        mock_results.results_dict = {
            "metrics/precision(B)": 0.85,
            "metrics/recall(B)": 0.78,
            "metrics/mAP50(B)": 0.82,
            "metrics/mAP50-95(B)": 0.60,
        }

        metrics = extract_metrics(mock_results)
        assert metrics["precision"] == pytest.approx(0.85, abs=1e-4)
        assert metrics["recall"] == pytest.approx(0.78, abs=1e-4)
        assert metrics["mAP50"] == pytest.approx(0.82, abs=1e-4)
        assert metrics["mAP50_95"] == pytest.approx(0.60, abs=1e-4)

    def test_missing_keys_return_zero(self) -> None:
        from scripts.training.train_yolo import extract_metrics

        mock_results = MagicMock()
        mock_results.results_dict = {}

        metrics = extract_metrics(mock_results)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_exception_returns_empty_dict(self) -> None:
        from scripts.training.train_yolo import extract_metrics

        mock_results = MagicMock()
        mock_results.results_dict = MagicMock(side_effect=AttributeError)

        metrics = extract_metrics(mock_results)
        assert isinstance(metrics, dict)


# ─── save_metrics_json ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSaveMetricsJson:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        import json

        from scripts.training.train_yolo import save_metrics_json

        metrics = {"mAP50": 0.82, "precision": 0.85}
        save_metrics_json(metrics, tmp_path)

        assert (tmp_path / "metrics.json").exists()
        loaded = json.loads((tmp_path / "metrics.json").read_text())
        assert loaded["mAP50"] == pytest.approx(0.82)

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        from scripts.training.train_yolo import save_metrics_json

        output_dir = tmp_path / "new" / "nested" / "dir"
        save_metrics_json({}, output_dir)
        assert output_dir.exists()


# ─── W&B setup ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSetupWandb:
    def test_disabled_returns_false(self) -> None:
        from scripts.training.train_yolo import _setup_wandb

        result = _setup_wandb({"enabled": False}, "test-run")
        assert result is False

    def test_missing_wandb_package_returns_false(self) -> None:
        from scripts.training.train_yolo import _setup_wandb

        with patch.dict("sys.modules", {"wandb": None}):
            result = _setup_wandb({"enabled": True}, "test-run")
            assert result is False

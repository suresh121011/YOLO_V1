"""
Unit tests for src.config.config_loader.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from src.config.config_loader import SystemConfig


class TestSystemConfigLoad:
    """Test SystemConfig.load() method."""

    @pytest.mark.unit
    def test_load_from_valid_config(self, tmp_path: Path) -> None:
        """Should load successfully from a valid feature_flags.yaml."""
        flags = tmp_path / "feature_flags.yaml"
        flags.write_text(
            yaml.dump({
                "components": {"smolvlm_analysis": False, "tts_output": True},
                "classes": {"passport": False},
                "rules": {"stove_unattended": True},
                "runtime": {"target_fps": 15, "confidence_threshold": 0.25},
            })
        )
        config = SystemConfig.load(str(flags), thresholds_path="/nonexistent.yaml")
        assert config.is_component_enabled("tts_output") is True
        assert config.is_component_enabled("smolvlm_analysis") is False
        assert config.is_class_enabled("passport") is False
        assert config.is_rule_enabled("stove_unattended") is True

    @pytest.mark.unit
    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when config file is missing."""
        with pytest.raises(FileNotFoundError):
            SystemConfig.load(str(tmp_path / "nonexistent.yaml"))

    @pytest.mark.unit
    def test_default_values(self, tmp_path: Path) -> None:
        """Should use safe defaults for missing keys."""
        flags = tmp_path / "feature_flags.yaml"
        flags.write_text("{}")  # Empty config
        config = SystemConfig.load(str(flags), thresholds_path="/nonexistent.yaml")
        assert config.is_component_enabled("yolo_detection") is True  # default: enabled
        assert config.is_class_enabled("person") is True
        assert config.get_runtime("target_fps") == 15

    @pytest.mark.unit
    def test_class_threshold_defaults(self, tmp_path: Path) -> None:
        """Safety classes should use lower default thresholds."""
        flags = tmp_path / "feature_flags.yaml"
        flags.write_text("{}")
        config = SystemConfig.load(str(flags), thresholds_path="/nonexistent.yaml")
        assert config.get_class_threshold("knife") < config.get_class_threshold("book")
        assert config.get_class_threshold("wet_floor") <= 0.25

"""
src.config.config_loader — Centralized Configuration Loader
============================================================

Loads and validates all YAML configuration files for the Elderly Assistant System.

This is the single source of truth for runtime configuration. All pipeline
components receive a SystemConfig instance — no component reads YAML directly.

Design principles:
    - All paths configurable (no hardcoded strings in pipeline code)
    - Feature flags expose every behavioral toggle
    - Fail-safe defaults: missing keys default to safe values
    - Immutable after load: config is a frozen dataclass

Usage:
    config = SystemConfig.load("configs/feature_flags.yaml")
    if config.is_component_enabled("smolvlm_analysis"):
        ...
    threshold = config.get_class_threshold("knife")  # 0.20

See also:
    - configs/feature_flags.yaml  for all available keys
    - configs/class_thresholds.yaml for per-class confidence overrides
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─── Default values (used when key is missing from YAML) ─────────────────────

_DEFAULT_RUNTIME: dict[str, Any] = {
    "target_fps": 15,
    "camera_index": 0,
    "confidence_threshold": 0.25,
    "smolvlm_every_n_frames": 5,
    "smolvlm_timeout_ms": 2000,
    "alert_cooldown_multiplier": 1.0,
    "max_alerts_per_minute": 6,
    "memory_window_frames": 150,
    "tts_speed": 1.0,
    "tts_volume": 0.8,
    "tts_language": "en_IN",
    "log_retention_days": 7,
    "active_learning_conf_min": 0.25,
    "active_learning_conf_max": 0.55,
}

_DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "knife": 0.20,
    "stove": 0.22,
    "gas_cylinder": 0.22,
    "wire": 0.22,
    "wet_floor": 0.20,
    "medicine_strip": 0.25,
    "medicine_bottle": 0.25,
    "passport": 0.40,
    "person": 0.30,
    "face": 0.35,
    "walking_stick": 0.28,
    "support_handle": 0.28,
    "door": 0.30,
}


# ─── Configuration Dataclasses ───────────────────────────────────────────────


@dataclass
class SystemConfig:
    """Complete runtime configuration for the Elderly Assistant System.

    Loaded from configs/feature_flags.yaml and configs/class_thresholds.yaml.
    All pipeline components receive this object — no component reads YAML directly.

    Args:
        components:        Component enable/disable flags
        classes:           Per-class detection enable/disable flags
        rules:             Per-rule enable/disable flags
        runtime:           Runtime behavior parameters
        class_thresholds:  Per-class confidence threshold overrides
        config_path:       Path to the feature_flags.yaml that was loaded

    Example:
        config = SystemConfig.load("configs/feature_flags.yaml")
        config.is_component_enabled("smolvlm_analysis")   # False (default)
        config.get_class_threshold("knife")               # 0.20
        config.is_class_enabled("passport")               # False (privacy)
        config.is_rule_enabled("stove_unattended")        # True
    """

    components: dict[str, bool] = field(default_factory=dict)
    classes: dict[str, bool] = field(default_factory=dict)
    rules: dict[str, bool] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    class_thresholds: dict[str, float] = field(default_factory=dict)
    config_path: str = "configs/feature_flags.yaml"

    @classmethod
    def load(
        cls,
        flags_path: str = "configs/feature_flags.yaml",
        thresholds_path: str = "configs/class_thresholds.yaml",
    ) -> SystemConfig:
        """Load configuration from YAML files.

        Args:
            flags_path:      Path to feature_flags.yaml
            thresholds_path: Path to class_thresholds.yaml

        Returns:
            Populated SystemConfig instance.

        Raises:
            FileNotFoundError: If flags_path does not exist.
        """
        flags_file = Path(flags_path)
        if not flags_file.exists():
            raise FileNotFoundError(
                f"Feature flags config not found: {flags_path}\n"
                f"Expected location: {flags_file.absolute()}"
            )

        with open(flags_file, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Load class thresholds (optional — uses defaults if not found)
        class_thresholds = dict(_DEFAULT_CLASS_THRESHOLDS)
        thresholds_file = Path(thresholds_path)
        if thresholds_file.exists():
            with open(thresholds_file, encoding="utf-8") as f:
                thresh_data: dict[str, Any] = yaml.safe_load(f) or {}
            class_thresholds.update(thresh_data.get("class_thresholds", {}))
        else:
            logger.warning(f"Class thresholds config not found: {thresholds_path} — using defaults")

        # Merge runtime defaults with YAML values
        runtime = dict(_DEFAULT_RUNTIME)
        runtime.update(data.get("runtime", {}))

        config = cls(
            components=data.get("components", {}),
            classes=data.get("classes", {}),
            rules=data.get("rules", {}),
            runtime=runtime,
            class_thresholds=class_thresholds,
            config_path=str(flags_file.absolute()),
        )

        logger.info(f"Configuration loaded from: {flags_path}")
        return config

    # ─── Query Methods ────────────────────────────────────────────────────────

    def is_component_enabled(self, component: str) -> bool:
        """Return True if a pipeline component is enabled.

        Args:
            component: Component name (e.g., "smolvlm_analysis", "tts_output")

        Returns:
            True if enabled, True if not specified (fail-open for components).
        """
        return self.components.get(component, True)

    def is_class_enabled(self, class_name: str) -> bool:
        """Return True if detections for this class should be processed.

        Args:
            class_name: YOLO class name (e.g., "knife", "passport")

        Returns:
            True if enabled, True if not specified. False for "passport" by default.
        """
        return self.classes.get(class_name, True)

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Return True if this safety rule should be evaluated.

        Args:
            rule_id: Rule identifier from risk_rules.yaml (e.g., "stove_unattended")

        Returns:
            True if enabled, True if not specified.
        """
        return self.rules.get(rule_id, True)

    def get_class_threshold(self, class_name: str) -> float:
        """Return the confidence threshold for a specific class.

        Safety-critical classes (knife, stove, gas_cylinder, wire, wet_floor)
        have lower thresholds than the global default to prefer recall.

        Args:
            class_name: YOLO class name

        Returns:
            Per-class threshold, or global confidence_threshold if not specified.
        """
        global_thresh = self.runtime.get("confidence_threshold", 0.25)
        return float(self.class_thresholds.get(class_name, global_thresh))

    def get_runtime(self, key: str, default: Any = None) -> Any:
        """Get a runtime configuration value by key.

        Args:
            key:     Runtime parameter key (e.g., "smolvlm_every_n_frames")
            default: Default value if key not found

        Returns:
            The runtime value, or default.
        """
        return self.runtime.get(key, default)

    def __repr__(self) -> str:
        enabled_components = [k for k, v in self.components.items() if v]
        return (
            f"SystemConfig("
            f"components={enabled_components}, "
            f"fps_target={self.runtime.get('target_fps', 15)}, "
            f"path={self.config_path!r})"
        )

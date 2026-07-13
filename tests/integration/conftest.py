"""
Integration tests conftest.

Integration tests:
    - Test pairs of components together (e.g., EventMemory + RuleEngine)
    - May use real YAML files from configs/
    - Should NOT require YOLO model weights or camera hardware
    - All tests complete in < 5 seconds
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Project root for resolving config paths in integration tests
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def risk_rules_path() -> Path:
    """Path to the real risk_rules.yaml config file."""
    return PROJECT_ROOT / "configs" / "risk_rules.yaml"


@pytest.fixture
def feature_flags_path() -> Path:
    """Path to the real feature_flags.yaml config file."""
    return PROJECT_ROOT / "configs" / "feature_flags.yaml"


@pytest.fixture
def class_thresholds_path() -> Path:
    """Path to the real class_thresholds.yaml config file."""
    return PROJECT_ROOT / "configs" / "class_thresholds.yaml"

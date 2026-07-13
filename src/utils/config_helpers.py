"""
src.utils.config_helpers — YAML Configuration Helpers
======================================================

Shared helpers for loading YAML configuration files and resolving runtime
settings across dataset, training, and inference scripts.

All scripts should load configuration through these helpers rather than
calling yaml.safe_load() directly, ensuring consistent error handling and
logging throughout the tooling layer.

Usage:
    from src.utils.config_helpers import load_yaml, load_training_config, resolve_device

    data_cfg = load_yaml("configs/data.yaml")
    train_cfg = load_training_config("configs/training/yolo11n_config.yaml")
    device = resolve_device("auto")  # → "cuda" | "mps" | "cpu"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─── Required top-level keys for training configs ────────────────────────────

_REQUIRED_TRAINING_KEYS: list[str] = ["model", "training", "output"]


# ─── YAML Loaders ─────────────────────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents as a dict. Returns empty dict if file is empty.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.absolute()}")

    with open(p, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    logger.debug(f"Loaded YAML: {p}")
    return data


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a training configuration YAML file.

    Validates that required top-level keys are present. Missing keys produce
    warnings but do not raise — callers may supply defaults.

    Args:
        path: Path to the training config YAML (e.g., yolo11n_config.yaml).

    Returns:
        Parsed training config dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    config = load_yaml(path)

    for key in _REQUIRED_TRAINING_KEYS:
        if key not in config:
            logger.warning(
                f"Training config missing required section '{key}' in {path}. "
                f"Using defaults for that section."
            )

    logger.info(f"Training config loaded from: {path}")
    return config


def load_data_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load the YOLO dataset configuration file.

    Args:
        path: Path to data.yaml. Defaults to configs/data.yaml.

    Returns:
        Parsed data config with keys: path, train, val, test, nc, names.

    Raises:
        FileNotFoundError: If data.yaml does not exist.
        ValueError: If required keys (nc, names) are missing.
    """
    config = load_yaml(path)

    required = ["nc", "names"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(
            f"data.yaml at '{path}' is missing required keys: {missing}"
        )

    logger.info(f"Data config loaded: {config.get('nc', 0)} classes from {path}")
    return config


# ─── Device Resolution ────────────────────────────────────────────────────────


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device string.

    Args:
        device_str: One of "auto", "cuda", "mps", "cpu", or a CUDA device
            index string (e.g., "0", "cuda:0").

    Returns:
        Resolved device string: "cuda", "mps", or "cpu".
    """
    if device_str not in ("auto",):
        logger.debug(f"Using specified device: {device_str}")
        return device_str

    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            logger.info("Auto device: CUDA selected")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto device: MPS (Apple Silicon) selected")
            return "mps"
    except ImportError:
        logger.debug("torch not available for device resolution — defaulting to cpu")

    logger.info("Auto device: CPU selected")
    return "cpu"


# ─── Config Accessors ─────────────────────────────────────────────────────────


def get_class_names_from_data_yaml(data_config: dict[str, Any]) -> dict[int, str]:
    """Extract class ID → name mapping from a parsed data.yaml dict.

    Args:
        data_config: Parsed data.yaml dict (from load_data_config).

    Returns:
        Dict mapping integer class IDs to class name strings.
    """
    names: dict[int, str] = {}
    raw = data_config.get("names", {})

    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        names = {i: str(name) for i, name in enumerate(raw)}

    return names


def get_dataset_paths(
    data_config: dict[str, Any],
    base_path: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve dataset split paths from a parsed data.yaml dict.

    Args:
        data_config: Parsed data.yaml dict (from load_data_config).
        base_path: Override for the 'path' key in data.yaml.

    Returns:
        Dict with keys 'root', 'train', 'val', 'test' as resolved Paths.
    """
    root = Path(base_path or data_config.get("path", "data/processed"))

    return {
        "root": root,
        "train": root / data_config.get("train", "images/train"),
        "val": root / data_config.get("val", "images/val"),
        "test": root / data_config.get("test", "images/test"),
    }

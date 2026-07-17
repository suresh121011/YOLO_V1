"""Golden regression tests for scripts.training.train_yolo.build_train_kwargs.

The Phase-4 backward-compatibility contract: with mitigation disabled, the
kwargs passed to Ultralytics ``model.train()`` are byte-identical to the
pre-Phase-4 pipeline — same keys, same values, and never a ``trainer`` key.
These goldens pin that contract; any diff here is a behavioral change to the
stock training path and must be an explicit, reviewed decision.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import pytest

train_yolo = importlib.import_module("scripts.training.train_yolo")

#: A fully-populated training config mirroring configs/training/yolo11n_config.yaml
#: at the Phase-4 branch point (pre-Phase-4 semantics).
_FULL_CFG: dict[str, Any] = {
    "model": {"base": "yolo11n.pt", "task": "detect", "device": "cpu"},
    "training": {
        "data": "configs/data.yaml",
        "epochs": 150,
        "imgsz": 640,
        "batch": 16,
        "patience": 25,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "warmup_bias_lr": 0.1,
        "close_mosaic": 15,
    },
    "augmentation": {
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.4,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.1,
        "copy_paste": 0.1,
    },
    "output": {
        "project": "models",
        "name": "yolo11n",
        "exist_ok": True,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
    },
}

#: The exact pre-Phase-4 model.train() kwargs for _FULL_CFG with default CLI.
_GOLDEN_FULL: dict[str, Any] = {
    "data": "configs/data.yaml",
    "epochs": 150,
    "imgsz": 640,
    "batch": 16,
    "patience": 25,
    "device": "cpu",
    "project": "models",
    "name": "yolo11n",
    "exist_ok": True,
    "save": True,
    "save_period": 10,
    "val": True,
    "plots": True,
    "verbose": True,
    "resume": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "warmup_bias_lr": 0.1,
    "close_mosaic": 15,
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.4,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.8,
    "mixup": 0.1,
    "copy_paste": 0.1,
}


def make_args(**overrides: Any) -> argparse.Namespace:
    """CLI namespace with train_yolo.py's defaults (all overrides unset)."""
    defaults: dict[str, Any] = {
        "config": "configs/training/yolo11n_config.yaml",
        "data": "configs/data.yaml",
        "model": None,
        "name": None,
        "epochs": None,
        "batch": None,
        "imgsz": None,
        "device": None,
        "patience": None,
        "resume": False,
        "mitigation": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
class TestGoldenKwargs:
    """Byte-identity guard for the disabled-mitigation path."""

    def test_full_config_matches_pre_phase4_golden_exactly(self) -> None:
        kwargs = train_yolo.build_train_kwargs(make_args(), _FULL_CFG)
        assert kwargs == _GOLDEN_FULL  # exact keys AND values

    def test_no_trainer_key_ever_present(self) -> None:
        kwargs = train_yolo.build_train_kwargs(make_args(), _FULL_CFG)
        assert "trainer" not in kwargs

    def test_mitigation_section_does_not_leak_into_kwargs(self) -> None:
        cfg = dict(_FULL_CFG)
        cfg["missing_annotation_mitigation"] = {"enabled": True}
        kwargs = train_yolo.build_train_kwargs(make_args(), cfg)
        assert kwargs == _GOLDEN_FULL  # section is invisible to kwargs assembly

    def test_key_order_is_stable(self) -> None:
        kwargs = train_yolo.build_train_kwargs(make_args(), _FULL_CFG)
        assert list(kwargs) == list(_GOLDEN_FULL)

    def test_missing_augmentation_section_adds_no_aug_keys(self) -> None:
        cfg = {k: v for k, v in _FULL_CFG.items() if k != "augmentation"}
        kwargs = train_yolo.build_train_kwargs(make_args(), cfg)
        golden = {k: v for k, v in _GOLDEN_FULL.items() if k not in _FULL_CFG["augmentation"]}
        assert kwargs == golden

    def test_empty_config_uses_script_defaults(self) -> None:
        kwargs = train_yolo.build_train_kwargs(make_args(device="cpu"), {})
        assert kwargs["epochs"] == 150
        assert kwargs["batch"] == 16
        assert kwargs["optimizer"] == "AdamW"
        assert kwargs["name"] == "yolo11n"
        assert "mosaic" not in kwargs  # no augmentation section ⇒ no aug keys

    def test_cli_overrides_win_over_config(self) -> None:
        kwargs = train_yolo.build_train_kwargs(
            make_args(epochs=7, batch=4, imgsz=320, patience=3, name="custom"), _FULL_CFG
        )
        assert kwargs["epochs"] == 7
        assert kwargs["batch"] == 4
        assert kwargs["imgsz"] == 320
        assert kwargs["patience"] == 3
        assert kwargs["name"] == "custom"

    def test_shipped_yolo11n_config_produces_stock_kwargs(self) -> None:
        # Guard on the real shipped config: values match the YAML and the
        # mitigation section stays invisible to the kwargs assembly.
        yaml = pytest.importorskip("yaml")
        with open("configs/training/yolo11n_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kwargs = train_yolo.build_train_kwargs(make_args(device="cpu"), cfg)
        assert kwargs == _GOLDEN_FULL

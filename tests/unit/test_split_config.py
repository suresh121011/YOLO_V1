"""Unit tests for src.dataset.split_config — split configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.split_config import SplitSettings, load_split_settings


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dataset_split_config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadSplitSettings:
    """YAML loading and validation."""

    def test_loads_values_from_yaml(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            """
split:
  strategy: stratified_group
  train_ratio: 0.70
  val_ratio: 0.20
  test_ratio: 0.10
  seed: 7
  group_by_capture: false
  source_dir: data/merged
  output_dir: data/processed
""",
        )
        settings = load_split_settings(path)
        assert settings.strategy == "stratified_group"
        assert settings.train_ratio == 0.70
        assert settings.val_ratio == 0.20
        assert settings.seed == 7
        assert settings.group_by_capture is False
        assert settings.source_dir == Path("data/merged")

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        settings = load_split_settings(tmp_path / "missing.yaml")
        assert settings == SplitSettings()

    def test_partial_yaml_merges_defaults(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "split:\n  seed: 99\n")
        settings = load_split_settings(path)
        assert settings.seed == 99
        assert settings.train_ratio == 0.80  # default preserved
        assert settings.strategy == "group_aware"

    def test_bad_ratio_sum_raises(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            "split:\n  train_ratio: 0.9\n  val_ratio: 0.2\n  test_ratio: 0.1\n",
        )
        with pytest.raises(ValueError):
            load_split_settings(path)

    def test_repo_config_is_loadable(self) -> None:
        # The actual configs/dataset_split_config.yaml must always parse.
        repo_config = Path(__file__).resolve().parents[2] / "configs" / "dataset_split_config.yaml"
        settings = load_split_settings(repo_config)
        assert settings.strategy == "group_aware"
        assert abs(settings.train_ratio + settings.val_ratio + settings.test_ratio - 1.0) < 1e-6

    def test_house_settings_default(self, tmp_path: Path) -> None:
        settings = load_split_settings(tmp_path / "missing.yaml")
        assert settings.house_pattern == r"(?:^|_)(h\d{2,})(?=_)"
        assert settings.holdout_houses == ()

    def test_house_settings_from_yaml(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            "split:\n  strategy: leave_one_house_out\n  holdout_houses: [h02, h03]\n",
        )
        settings = load_split_settings(path)
        assert settings.strategy == "leave_one_house_out"
        assert settings.holdout_houses == ("h02", "h03")


@pytest.mark.unit
class TestWithOverrides:
    """CLI-over-YAML precedence."""

    def test_none_overrides_keep_config_values(self) -> None:
        base = SplitSettings(seed=42, train_ratio=0.8)
        resolved = base.with_overrides(seed=None, train_ratio=None)
        assert resolved == base

    def test_explicit_overrides_win(self) -> None:
        base = SplitSettings()
        resolved = base.with_overrides(
            seed=1,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            source_dir=Path("x"),
            output_dir=Path("y"),
            strategy="stratified_group",
        )
        assert resolved.seed == 1
        assert resolved.train_ratio == 0.7
        assert resolved.source_dir == Path("x")
        assert resolved.strategy == "stratified_group"
        # base is frozen and unchanged
        assert base.seed == 42

    def test_house_settings_preserved_across_overrides(self) -> None:
        base = SplitSettings(house_pattern="custom", holdout_houses=("h01",))
        resolved = base.with_overrides(seed=1)
        assert resolved.house_pattern == "custom"
        assert resolved.holdout_houses == ("h01",)

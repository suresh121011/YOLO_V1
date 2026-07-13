"""Unit tests for src.dataset.sources_config — acquisition configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.sources_config import load_sources_config

MINIMAL_CONFIG = """
mode: smoke
smoke:
  limit_per_source: 10
allow_noncommercial: false
dedup:
  hamming_threshold: 3
  check_flips: true
sources:
  coco:
    enabled: true
    output_dir: data/raw/coco_filtered
    license: "CC BY 4.0"
    remap_table: coco
    trusted_classes: [person]
    class_caps: {person: 800}
  wider_face:
    enabled: true
    noncommercial: true
    license: "research-only"
  disabled_source:
    enabled: false
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dataset_sources.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadSourcesConfig:
    """YAML loading, defaults and validation."""

    def test_loads_minimal_config(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.mode == "smoke"
        assert config.limit == 10
        assert config.dedup.hamming_threshold == 3
        assert config.sources["coco"].remap_table == "coco"
        # Non-field keys land in options
        assert config.sources["coco"].options["class_caps"] == {"person": 800}

    def test_full_mode_has_no_limit(self, tmp_path: Path) -> None:
        config = load_sources_config(
            _write(tmp_path, MINIMAL_CONFIG.replace("mode: smoke", "mode: full"))
        )
        assert config.limit is None

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_sources_config(
                _write(tmp_path, MINIMAL_CONFIG.replace("mode: smoke", "mode: turbo"))
            )

    def test_missing_sources_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_sources_config(_write(tmp_path, "mode: smoke\n"))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_sources_config(tmp_path / "missing.yaml")

    def test_repo_config_is_loadable(self) -> None:
        repo_config = Path(__file__).resolve().parents[2] / "configs" / "dataset_sources.yaml"
        config = load_sources_config(repo_config)
        assert config.mode in ("smoke", "full")
        for expected in ("coco", "openimages", "roboflow", "wider_face", "negatives"):
            assert expected in config.sources, f"missing source '{expected}'"


@pytest.mark.unit
class TestLicenseGate:
    """allow_noncommercial governance gate."""

    def test_noncommercial_source_blocked_when_gate_closed(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.allow_noncommercial is False
        assert config.is_source_allowed("wider_face") is False
        assert config.is_source_allowed("coco") is True

    def test_noncommercial_source_allowed_when_gate_open(self, tmp_path: Path) -> None:
        body = MINIMAL_CONFIG.replace("allow_noncommercial: false", "allow_noncommercial: true")
        config = load_sources_config(_write(tmp_path, body))
        assert config.is_source_allowed("wider_face") is True

    def test_disabled_and_unknown_sources_not_allowed(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.is_source_allowed("disabled_source") is False
        assert config.is_source_allowed("does_not_exist") is False

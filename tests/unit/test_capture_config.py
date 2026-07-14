"""Unit tests for src.dataset.capture.config — capture configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.capture.config import (
    CaptureConfig,
    IaaSettings,
    load_capture_config,
    parse_session_id,
)

_FULL_YAML = """
capture:
  inbox_dir: staging/inbox
  captures_root: data/raw/custom_captures
  eval_root: data/eval/indian_home_v0
  session_id_pattern: "^h\\\\d{2}_[a-z_]+_s\\\\d{3}$"
  rooms: [kitchen, bedroom]
  lighting: [daylight, dim]
  image:
    min_dim: 512
    allowed_extensions: [jpg, .PNG]
    strip_metadata: false
    max_file_mb: 10
consent:
  registry_path: consent/registry.yaml
  reference_pattern: "^C-\\\\d{3}$"
  required: false
annotation:
  staging_dir: staging/annotations
  min_labeled_fraction: 0.9
  iaa:
    iou_threshold: 0.4
    min_agreement: 0.8
    wet_floor_min_agreement: 0.5
targets:
  total_images: 100
  min_instances_per_class: 10
  custom_classes: [stove, passport]
  min_houses: 2
"""


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "capture_config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadCaptureConfig:
    """YAML loading, defaults and validation."""

    def test_loads_all_sections(self, tmp_path: Path) -> None:
        config = load_capture_config(_write_config(tmp_path, _FULL_YAML))
        assert config.inbox_dir == Path("staging/inbox")
        assert config.rooms == ("kitchen", "bedroom")
        assert config.lighting == ("daylight", "dim")
        assert config.image.min_dim == 512
        assert config.image.strip_metadata is False
        assert config.image.max_file_mb == 10
        assert config.consent.registry_path == Path("consent/registry.yaml")
        assert config.consent.required is False
        assert config.annotation.min_labeled_fraction == 0.9
        assert config.annotation.iaa.iou_threshold == 0.4
        assert config.annotation.iaa.wet_floor_min_agreement == 0.5
        assert config.targets.total_images == 100
        assert config.targets.custom_classes == ("stove", "passport")

    def test_extensions_normalized_lowercase_with_dot(self, tmp_path: Path) -> None:
        config = load_capture_config(_write_config(tmp_path, _FULL_YAML))
        assert config.image.allowed_extensions == (".jpg", ".png")

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_capture_config(tmp_path / "missing.yaml")
        assert config == CaptureConfig()
        assert len(config.targets.custom_classes) == 8

    def test_partial_yaml_merges_defaults(self, tmp_path: Path) -> None:
        config = load_capture_config(_write_config(tmp_path, "targets:\n  total_images: 5\n"))
        assert config.targets.total_images == 5
        assert config.targets.min_instances_per_class == 200  # default preserved
        assert config.image.min_dim == 480

    def test_bad_session_pattern_raises(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, 'capture:\n  session_id_pattern: "([unclosed"\n')
        with pytest.raises(ValueError, match="session_id_pattern"):
            load_capture_config(path)

    def test_bad_consent_pattern_raises(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, 'consent:\n  reference_pattern: "([unclosed"\n')
        with pytest.raises(ValueError, match="reference_pattern"):
            load_capture_config(path)

    def test_out_of_range_fraction_raises(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "annotation:\n  min_labeled_fraction: 1.5\n")
        with pytest.raises(ValueError, match="min_labeled_fraction"):
            load_capture_config(path)

    def test_out_of_range_iaa_threshold_raises(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "annotation:\n  iaa:\n    iou_threshold: 0.0\n")
        with pytest.raises(ValueError, match="iou_threshold"):
            load_capture_config(path)

    def test_nonpositive_target_raises(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "targets:\n  total_images: 0\n")
        with pytest.raises(ValueError, match="positive"):
            load_capture_config(path)

    def test_captures_root_drift_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sources = tmp_path / "dataset_sources.yaml"
        sources.write_text(
            "mode: smoke\nsources:\n  custom_captures:\n    output_dir: data/raw/elsewhere\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            load_capture_config(_write_config(tmp_path, _FULL_YAML), sources_config_path=sources)
        assert any("captures_root" in r.message for r in caplog.records)

    def test_repo_config_is_loadable(self) -> None:
        # The actual configs/capture_config.yaml must always parse and agree
        # with dataset_sources.yaml on the captures root.
        root = Path(__file__).resolve().parents[2]
        config = load_capture_config(
            root / "configs" / "capture_config.yaml",
            sources_config_path=root / "configs" / "dataset_sources.yaml",
        )
        assert config.captures_root == Path("data/raw/custom_captures")
        assert config.targets.min_instances_per_class == 200
        assert len(config.targets.custom_classes) == 8
        assert config.image.strip_metadata is True
        assert config.consent.required is True


@pytest.mark.unit
class TestSessionIdHelpers:
    """Session ID grammar helpers."""

    def test_parse_session_id(self) -> None:
        assert parse_session_id("h01_kitchen_s001") == ("h01", "kitchen")
        assert parse_session_id("h12_pooja_room_s003") == ("h12", "pooja_room")

    def test_parse_session_id_rejects_malformed(self) -> None:
        with pytest.raises(ValueError):
            parse_session_id("h01s001")

    def test_validate_session_id_ok(self) -> None:
        assert CaptureConfig().validate_session_id("h01_kitchen_s001") == []

    def test_validate_session_id_bad_grammar(self) -> None:
        problems = CaptureConfig().validate_session_id("house1-kitchen-1")
        assert problems and "pattern" in problems[0]

    def test_validate_session_id_unknown_room(self) -> None:
        problems = CaptureConfig().validate_session_id("h01_garage_s001")
        assert problems and "room" in problems[0]


@pytest.mark.unit
class TestIaaSettings:
    """Per-class agreement gate resolution."""

    def test_wet_floor_override(self) -> None:
        iaa = IaaSettings(min_agreement=0.75, wet_floor_min_agreement=0.6)
        assert iaa.min_agreement_for("wet_floor") == 0.6
        assert iaa.min_agreement_for("stove") == 0.75


@pytest.mark.unit
class TestWithOverrides:
    """CLI-over-YAML precedence."""

    def test_none_overrides_keep_config_values(self) -> None:
        base = CaptureConfig()
        assert base.with_overrides() == base

    def test_explicit_overrides_win(self) -> None:
        base = CaptureConfig()
        resolved = base.with_overrides(inbox_dir=Path("local/inbox"))
        assert resolved.inbox_dir == Path("local/inbox")
        assert resolved.captures_root == base.captures_root
        assert base.inbox_dir == Path("data/capture_inbox")  # frozen, unchanged

"""Unit tests for src.training.mitigation_config — typed mitigation settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.mitigation_config import MITIGATION_SECTION, MitigationConfig


@pytest.mark.unit
class TestFromTrainingConfig:
    """Section parsing and backward-compat defaults."""

    def test_absent_section_yields_disabled_defaults(self) -> None:
        config = MitigationConfig.from_training_config({"model": {}, "training": {}})
        assert config.enabled is False
        assert config.completeness_path == Path("data/processed/completeness.json")
        assert config.on_unknown_image == "error"
        assert config.mixing_augmentation_policy == "forbid"
        assert config.log_mask_stats is True

    def test_empty_section_yields_disabled_defaults(self) -> None:
        config = MitigationConfig.from_training_config({MITIGATION_SECTION: None})
        assert config.enabled is False

    def test_full_section_parses(self) -> None:
        config = MitigationConfig.from_training_config(
            {
                MITIGATION_SECTION: {
                    "enabled": True,
                    "completeness_path": "custom/completeness.json",
                    "on_unknown_image": "warn_full_supervision",
                    "mixing_augmentation_policy": "warn",
                    "log_mask_stats": False,
                }
            }
        )
        assert config.enabled is True
        assert config.completeness_path == Path("custom/completeness.json")
        assert config.on_unknown_image == "warn_full_supervision"
        assert config.mixing_augmentation_policy == "warn"
        assert config.log_mask_stats is False

    def test_unknown_key_is_error_naming_key(self) -> None:
        with pytest.raises(ValueError, match="mosiac_policy"):
            MitigationConfig.from_training_config(
                {MITIGATION_SECTION: {"enabled": True, "mosiac_policy": "warn"}}
            )

    def test_non_mapping_section_is_error(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            MitigationConfig.from_training_config({MITIGATION_SECTION: "yes"})

    def test_invalid_on_unknown_image_names_key_and_values(self) -> None:
        with pytest.raises(ValueError, match="on_unknown_image.*error.*warn_full_supervision"):
            MitigationConfig.from_training_config(
                {MITIGATION_SECTION: {"on_unknown_image": "explode"}}
            )

    def test_invalid_mixing_policy_names_key_and_values(self) -> None:
        with pytest.raises(ValueError, match="mixing_augmentation_policy.*forbid"):
            MitigationConfig.from_training_config(
                {MITIGATION_SECTION: {"mixing_augmentation_policy": "shrug"}}
            )


@pytest.mark.unit
class TestWithOverrides:
    """CLI-over-YAML precedence semantics."""

    def test_override_replaces_field(self) -> None:
        base = MitigationConfig()
        updated = base.with_overrides(enabled=True)
        assert updated.enabled is True
        assert base.enabled is False  # frozen original untouched

    def test_none_overrides_are_ignored(self) -> None:
        base = MitigationConfig(enabled=True)
        assert base.with_overrides(enabled=None).enabled is True

    def test_unknown_override_is_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown MitigationConfig override"):
            MitigationConfig().with_overrides(turbo=True)

    def test_override_result_is_validated(self) -> None:
        with pytest.raises(ValueError, match="on_unknown_image"):
            MitigationConfig().with_overrides(on_unknown_image="explode")

    def test_frozen_dataclass_is_immutable(self) -> None:
        config = MitigationConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

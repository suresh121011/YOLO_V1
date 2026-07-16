"""
src.training.mitigation_config — Missing-Annotation Mitigation Config
=====================================================================

Typed surface for the optional ``missing_annotation_mitigation:`` section of
the training config (configs/training/yolo11n_config.yaml). Follows the house
config pattern: frozen dataclass + imperative validation raising ValueError
naming the offending key + ``with_overrides()`` for CLI-over-YAML precedence.

Backward compatibility contract: an absent section (or ``enabled: false``)
yields a disabled config, and the training script then takes exactly the
pre-Phase-4 code path — no completeness read, no custom trainer, no torch
import from this package.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Config section name inside the training YAML.
MITIGATION_SECTION = "missing_annotation_mitigation"

VALID_ON_UNKNOWN_IMAGE = ("error", "warn_full_supervision")
VALID_MIXING_AUG_POLICIES = ("forbid", "warn", "ignore")

#: Ultralytics train kwargs that composite multiple images into one sample.
#: Only the primary image's filename survives into batch["im_file"], so
#: per-image masks are unsound while these are active (ADR-P4-04).
MIXING_AUG_KEYS = ("mosaic", "mixup", "copy_paste")


@dataclass(frozen=True)
class MitigationConfig:
    """Missing-annotation mitigation settings.

    Attributes:
        enabled:                  Master switch. False ⇒ byte-for-byte stock
                                  training behavior.
        completeness_path:        Per-image completeness artifact (produced by
                                  the generate_completeness DVC stage).
        on_unknown_image:         Loss behavior for a batch image absent from
                                  the artifact: "error" (fail fast, default)
                                  or "warn_full_supervision" (log once and
                                  train that image unmasked).
        mixing_augmentation_policy: Preflight stance on mosaic/mixup/copy_paste
                                  being active while mitigation is enabled:
                                  "forbid" (gate fails, default), "warn", or
                                  "ignore".
        log_mask_stats:           Log per-epoch masking statistics (share of
                                  masked (image, class) cells).
    """

    enabled: bool = False
    completeness_path: Path = Path("data/processed/completeness.json")
    on_unknown_image: str = "error"
    mixing_augmentation_policy: str = "forbid"
    log_mask_stats: bool = True

    @classmethod
    def from_training_config(cls, train_cfg: dict[str, Any]) -> MitigationConfig:
        """Build from a parsed training config dict.

        Args:
            train_cfg: Full training config (output of load_training_config).
                       An absent/empty section yields disabled defaults.

        Returns:
            A validated MitigationConfig.

        Raises:
            ValueError: If the section has invalid keys or values.
        """
        section = train_cfg.get(MITIGATION_SECTION) or {}
        if not isinstance(section, dict):
            raise ValueError(
                f"Training config section '{MITIGATION_SECTION}' must be a mapping, "
                f"got {type(section).__name__}"
            )

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in training config section "
                f"'{MITIGATION_SECTION}'. Valid keys: {sorted(known)}"
            )

        config = cls(
            enabled=bool(section.get("enabled", False)),
            completeness_path=Path(section.get("completeness_path", cls.completeness_path)),
            on_unknown_image=str(section.get("on_unknown_image", cls.on_unknown_image)),
            mixing_augmentation_policy=str(
                section.get("mixing_augmentation_policy", cls.mixing_augmentation_policy)
            ),
            log_mask_stats=bool(section.get("log_mask_stats", True)),
        )
        config.validate()
        return config

    def with_overrides(self, **overrides: Any) -> MitigationConfig:
        """Return a copy with the given fields replaced (CLI precedence).

        Args:
            **overrides: Field name → new value. None values are ignored so
                         unset CLI flags pass through cleanly.

        Returns:
            A new validated MitigationConfig.

        Raises:
            ValueError: On unknown field names or invalid resulting values.
        """
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(k for k in overrides if k not in known)
        if unknown:
            raise ValueError(f"Unknown MitigationConfig override(s): {unknown}")
        effective = {k: v for k, v in overrides.items() if v is not None}
        config = dataclasses.replace(self, **effective)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate field values.

        Raises:
            ValueError: Naming the offending key and the accepted values.
        """
        if self.on_unknown_image not in VALID_ON_UNKNOWN_IMAGE:
            raise ValueError(
                f"{MITIGATION_SECTION}.on_unknown_image must be one of "
                f"{VALID_ON_UNKNOWN_IMAGE}, got '{self.on_unknown_image}'"
            )
        if self.mixing_augmentation_policy not in VALID_MIXING_AUG_POLICIES:
            raise ValueError(
                f"{MITIGATION_SECTION}.mixing_augmentation_policy must be one of "
                f"{VALID_MIXING_AUG_POLICIES}, got '{self.mixing_augmentation_policy}'"
            )
        if not str(self.completeness_path):
            raise ValueError(f"{MITIGATION_SECTION}.completeness_path must be a non-empty path")

"""
src.dataset.split_config — Split Configuration Loader
=====================================================

Loads ``configs/dataset_split_config.yaml`` into typed settings and merges
them with CLI overrides. This makes the split configuration file the single
source of truth for ratios/seed/strategy while keeping every value
overridable from the command line (explicit CLI flag > YAML > built-in
default).

Consumed by:
    scripts/dataset/split_dataset.py
    scripts/dataset/generate_splits.py
    src/dataset/splitting/registry.py (strategy selection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.utils.config_helpers import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_CONFIG_PATH = Path("configs/dataset_split_config.yaml")


@dataclass(frozen=True)
class SplitSettings:
    """Resolved dataset-split settings (YAML merged over built-in defaults)."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    strategy: str = "group_aware"
    group_by_capture: bool = True
    source_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/processed")
    # M3 (ADR-P5-05): when set, labels are read from this overlay directory
    # instead of ``source_dir / "labels"`` — images still come from
    # ``source_dir / "images"`` (no image duplication). None preserves the
    # pre-M3 behavior exactly.
    source_labels_dir: Path | None = None
    # leave_one_house_out settings (ignored by other strategies)
    house_pattern: str = r"(?:^|_)(h\d{2,})(?=_)"
    holdout_houses: tuple[str, ...] = ()

    def with_overrides(
        self,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int | None = None,
        strategy: str | None = None,
        source_dir: Path | None = None,
        output_dir: Path | None = None,
        source_labels_dir: Path | None = None,
    ) -> SplitSettings:
        """Return a copy with any non-``None`` CLI overrides applied."""
        return SplitSettings(
            train_ratio=self.train_ratio if train_ratio is None else train_ratio,
            val_ratio=self.val_ratio if val_ratio is None else val_ratio,
            test_ratio=self.test_ratio if test_ratio is None else test_ratio,
            seed=self.seed if seed is None else seed,
            strategy=self.strategy if strategy is None else strategy,
            group_by_capture=self.group_by_capture,
            source_dir=self.source_dir if source_dir is None else source_dir,
            output_dir=self.output_dir if output_dir is None else output_dir,
            source_labels_dir=(
                self.source_labels_dir if source_labels_dir is None else source_labels_dir
            ),
            house_pattern=self.house_pattern,
            holdout_houses=self.holdout_houses,
        )


def load_split_settings(path: Path | None = None) -> SplitSettings:
    """Load split settings from YAML, falling back to built-in defaults.

    A missing or empty file is not an error: the built-in defaults are
    returned with a warning so the pipeline stays runnable in minimal
    checkouts.

    Args:
        path: Path to the split config YAML. Defaults to
              ``configs/dataset_split_config.yaml``.

    Returns:
        Resolved :class:`SplitSettings`.

    Raises:
        ValueError: If the ``split:`` section contains non-numeric ratios.
    """
    config_path = path if path is not None else DEFAULT_SPLIT_CONFIG_PATH
    defaults = SplitSettings()

    try:
        raw = load_yaml(config_path)
    except FileNotFoundError:
        logger.warning(f"Split config not found at {config_path} — using built-in defaults")
        return defaults

    section = raw.get("split", {})
    if not isinstance(section, dict):
        logger.warning(f"'split:' section missing/invalid in {config_path} — using defaults")
        return defaults

    settings = SplitSettings(
        train_ratio=float(section.get("train_ratio", defaults.train_ratio)),
        val_ratio=float(section.get("val_ratio", defaults.val_ratio)),
        test_ratio=float(section.get("test_ratio", defaults.test_ratio)),
        seed=int(section.get("seed", defaults.seed)),
        strategy=str(section.get("strategy", defaults.strategy)),
        group_by_capture=bool(section.get("group_by_capture", defaults.group_by_capture)),
        source_dir=Path(section.get("source_dir", defaults.source_dir)),
        output_dir=Path(section.get("output_dir", defaults.output_dir)),
        source_labels_dir=(
            Path(section["source_labels_dir"]) if section.get("source_labels_dir") else None
        ),
        house_pattern=str(section.get("house_pattern", defaults.house_pattern)),
        holdout_houses=tuple(section.get("holdout_houses", []) or []),
    )

    total = settings.train_ratio + settings.val_ratio + settings.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios in {config_path} must sum to 1.0, got {total:.6f}")

    logger.info(
        f"Split config loaded from {config_path}: "
        f"{settings.train_ratio}/{settings.val_ratio}/{settings.test_ratio}, "
        f"seed={settings.seed}, strategy={settings.strategy}"
    )
    return settings

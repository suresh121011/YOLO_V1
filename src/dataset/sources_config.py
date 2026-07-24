"""
src.dataset.sources_config — Acquisition Configuration Loader
=============================================================

Typed loader for ``configs/dataset_sources.yaml``, the single source of
truth for Phase-2 dataset acquisition: smoke/full mode, per-source URLs,
class caps, license metadata, label-completeness (``trusted_classes``),
dedup and indoor-filter settings.

The same YAML doubles as the DVC params file for the acquisition stages,
so flipping ``mode: smoke`` → ``mode: full`` invalidates and re-runs the
download stages on the next ``dvc repro``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config_helpers import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_SOURCES_CONFIG_PATH = Path("configs/dataset_sources.yaml")

VALID_MODES = ("smoke", "full")

# Keys lifted into SourceConfig fields; everything else lands in `options`.
_SOURCE_FIELD_KEYS = frozenset(
    {"enabled", "output_dir", "license", "noncommercial", "trusted_classes", "remap_table"}
)


@dataclass(frozen=True)
class DedupSettings:
    """Perceptual-deduplication settings (merge stage, pre-split)."""

    hamming_threshold: int = 5
    check_flips: bool = True
    #: aHash grid size; the hash is ``hash_size**2`` bits. 8 → 64-bit (default,
    #: coarse). Raising to 16 → 256-bit discriminates genuinely-distinct capture
    #: frames an 8×8 hash wrongly merges (P2 finding). NOTE ``hamming_threshold``
    #: is an ABSOLUTE bit count, so a larger grid makes a fixed threshold
    #: proportionally stricter (fewer drops) — retune the two together.
    hash_size: int = 8


@dataclass(frozen=True)
class IndoorFilterSettings:
    """Indoor/quality heuristic thresholds (docs/03 dataset_templates.md)."""

    enabled: bool = True
    brightness_outdoor_threshold: int = 160
    portrait_aspect_max: float = 0.6
    min_image_dim: int = 320


@dataclass
class SourceConfig:
    """Configuration for one acquisition source.

    Attributes:
        name:            Source identifier (config key, e.g. "coco").
        enabled:         Whether the source participates in the pipeline.
        output_dir:      Destination under data/raw/.
        license:         Human-readable license string (goes into manifests).
        noncommercial:   True for research-only sources (license gate).
        trusted_classes: Classes this source labels exhaustively.
        remap_table:     Name of the remap table in src/dataset/remap.py.
        options:         Source-specific settings (URLs, caps, slugs, counts).
    """

    name: str
    enabled: bool = True
    output_dir: Path = Path("data/raw")
    license: str = ""
    noncommercial: bool = False
    trusted_classes: list[str] = field(default_factory=list)
    remap_table: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourcesConfig:
    """Resolved acquisition configuration (see module docstring)."""

    mode: str = "smoke"
    smoke_limit: int = 60
    allow_noncommercial: bool = True
    dedup: DedupSettings = field(default_factory=DedupSettings)
    indoor_filter: IndoorFilterSettings = field(default_factory=IndoorFilterSettings)
    raw_root: Path = Path("data/raw")
    interim_root: Path = Path("data/interim")
    merged_root: Path = Path("data/merged")
    downloads_cache: Path = Path("data/downloads_cache")
    sources: dict[str, SourceConfig] = field(default_factory=dict)

    @property
    def limit(self) -> int | None:
        """Per-source image cap: ``smoke_limit`` in smoke mode, else None."""
        return self.smoke_limit if self.mode == "smoke" else None

    def is_source_allowed(self, name: str) -> bool:
        """Return True if a source is enabled and passes the license gate.

        Non-commercial sources are excluded whenever
        ``allow_noncommercial`` is false — this is the governance gate for
        shippable dataset builds.
        """
        source = self.sources.get(name)
        if source is None or not source.enabled:
            return False
        if source.noncommercial and not self.allow_noncommercial:
            logger.warning(
                f"Source '{name}' skipped: non-commercial license and allow_noncommercial is false"
            )
            return False
        return True


def load_sources_config(path: Path | None = None) -> SourcesConfig:
    """Load and validate ``configs/dataset_sources.yaml``.

    Args:
        path: Config path; defaults to configs/dataset_sources.yaml.

    Returns:
        Resolved :class:`SourcesConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If ``mode`` is invalid or ``sources`` is missing.
    """
    config_path = path if path is not None else DEFAULT_SOURCES_CONFIG_PATH
    raw = load_yaml(config_path)

    mode = str(raw.get("mode", "smoke")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}' in {config_path}; expected one of {VALID_MODES}")

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError(f"'sources:' section missing or empty in {config_path}")

    dedup_raw = raw.get("dedup", {}) or {}
    filter_raw = raw.get("indoor_filter", {}) or {}
    paths_raw = raw.get("paths", {}) or {}
    smoke_raw = raw.get("smoke", {}) or {}

    sources: dict[str, SourceConfig] = {}
    for name, cfg in raw_sources.items():
        cfg = cfg or {}
        sources[name] = SourceConfig(
            name=name,
            enabled=bool(cfg.get("enabled", True)),
            output_dir=Path(cfg.get("output_dir", Path("data/raw") / name)),
            license=str(cfg.get("license", "")),
            noncommercial=bool(cfg.get("noncommercial", False)),
            trusted_classes=list(cfg.get("trusted_classes", []) or []),
            remap_table=str(cfg.get("remap_table", "")),
            options={k: v for k, v in cfg.items() if k not in _SOURCE_FIELD_KEYS},
        )

    config = SourcesConfig(
        mode=mode,
        smoke_limit=int(smoke_raw.get("limit_per_source", 60)),
        allow_noncommercial=bool(raw.get("allow_noncommercial", True)),
        dedup=DedupSettings(
            hamming_threshold=int(dedup_raw.get("hamming_threshold", 5)),
            check_flips=bool(dedup_raw.get("check_flips", True)),
            hash_size=int(dedup_raw.get("hash_size", 8)),
        ),
        indoor_filter=IndoorFilterSettings(
            enabled=bool(filter_raw.get("enabled", True)),
            brightness_outdoor_threshold=int(filter_raw.get("brightness_outdoor_threshold", 160)),
            portrait_aspect_max=float(filter_raw.get("portrait_aspect_max", 0.6)),
            min_image_dim=int(filter_raw.get("min_image_dim", 320)),
        ),
        raw_root=Path(paths_raw.get("raw_root", "data/raw")),
        interim_root=Path(paths_raw.get("interim_root", "data/interim")),
        merged_root=Path(paths_raw.get("merged_root", "data/merged")),
        downloads_cache=Path(paths_raw.get("downloads_cache", "data/downloads_cache")),
        sources=sources,
    )

    logger.info(
        f"Sources config loaded from {config_path}: mode={config.mode}, "
        f"{len(config.sources)} sources, limit={config.limit}"
    )
    return config

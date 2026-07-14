"""
src.dataset.capture.config — Capture Configuration Loader
=========================================================

Typed loader for ``configs/capture_config.yaml``, the single source of
truth for the Phase-3 custom capture workflow: inbox/captures/eval paths,
session-ID grammar, image intake requirements, consent settings,
annotation import thresholds and collection targets.

Follows the project loader pattern (frozen dataclasses, imperative
validation, CLI-over-YAML precedence via :meth:`CaptureConfig.with_overrides`).

Consumed by:
    scripts/dataset/08_ingest_capture_session.py
    scripts/dataset/09_import_annotations.py
    scripts/dataset/10_capture_progress.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from src.utils.config_helpers import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_CAPTURE_CONFIG_PATH = Path("configs/capture_config.yaml")

#: The 8 classes with no adequate public-dataset coverage (configs/data.yaml).
DEFAULT_CUSTOM_CLASSES: tuple[str, ...] = (
    "gas_cylinder",
    "medicine_strip",
    "wet_floor",
    "walking_stick",
    "support_handle",
    "stove",
    "passport",
    "cupboard",
)

_DEFAULT_ROOMS: tuple[str, ...] = (
    "kitchen",
    "bedroom",
    "bathroom",
    "hall",
    "balcony",
    "corridor",
    "pooja_room",
    "staircase",
)

_DEFAULT_LIGHTING: tuple[str, ...] = ("daylight", "tubelight", "dim", "night_flash", "mixed")


@dataclass(frozen=True)
class ImageRequirements:
    """Intake requirements applied to every image at ingest."""

    min_dim: int = 480
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    strip_metadata: bool = True
    max_file_mb: int = 25


@dataclass(frozen=True)
class ConsentSettings:
    """Consent-record verification settings (privacy: no PII in the repo)."""

    registry_path: Path = Path("data/consent/consent_registry.yaml")
    reference_pattern: str = r"^CONSENT-h\d{2}-\d{4}-\d{3}$"
    required: bool = True


@dataclass(frozen=True)
class IaaSettings:
    """Inter-annotator agreement thresholds (dual-annotator workflow)."""

    iou_threshold: float = 0.50
    min_agreement: float = 0.75
    wet_floor_min_agreement: float = 0.60

    def min_agreement_for(self, class_name: str) -> float:
        """Return the agreement gate for a class (wet_floor has an R24 override)."""
        if class_name == "wet_floor":
            return self.wet_floor_min_agreement
        return self.min_agreement


@dataclass(frozen=True)
class AnnotationSettings:
    """Annotation staging/import settings."""

    staging_dir: Path = Path("data/capture_inbox/annotations")
    min_labeled_fraction: float = 0.95
    iaa: IaaSettings = field(default_factory=IaaSettings)


@dataclass(frozen=True)
class CollectionTargets:
    """Phase-3 collection acceptance targets (dataset governance)."""

    total_images: int = 2000
    min_instances_per_class: int = 200
    custom_classes: tuple[str, ...] = DEFAULT_CUSTOM_CLASSES
    min_houses: int = 3


@dataclass(frozen=True)
class CaptureConfig:
    """Resolved capture-workflow configuration (see module docstring)."""

    inbox_dir: Path = Path("data/capture_inbox")
    captures_root: Path = Path("data/raw/custom_captures")
    eval_root: Path = Path("data/eval/indian_home_v0")
    session_id_pattern: str = r"^h\d{2}_[a-z_]+_s\d{3}$"
    rooms: tuple[str, ...] = _DEFAULT_ROOMS
    lighting: tuple[str, ...] = _DEFAULT_LIGHTING
    image: ImageRequirements = field(default_factory=ImageRequirements)
    consent: ConsentSettings = field(default_factory=ConsentSettings)
    annotation: AnnotationSettings = field(default_factory=AnnotationSettings)
    targets: CollectionTargets = field(default_factory=CollectionTargets)

    def with_overrides(
        self,
        inbox_dir: Path | None = None,
        captures_root: Path | None = None,
        eval_root: Path | None = None,
    ) -> CaptureConfig:
        """Return a copy with any non-``None`` CLI overrides applied."""
        return replace(
            self,
            inbox_dir=self.inbox_dir if inbox_dir is None else inbox_dir,
            captures_root=self.captures_root if captures_root is None else captures_root,
            eval_root=self.eval_root if eval_root is None else eval_root,
        )

    def validate_session_id(self, session_id: str) -> list[str]:
        """Check a session ID against the grammar and the known-rooms list.

        Returns:
            List of problems (empty when the session ID is valid).
        """
        problems: list[str] = []
        if not re.match(self.session_id_pattern, session_id):
            problems.append(
                f"session id '{session_id}' does not match pattern {self.session_id_pattern}"
            )
            return problems
        room = parse_session_id(session_id)[1]
        if room not in self.rooms:
            problems.append(f"session id room '{room}' not in configured rooms {list(self.rooms)}")
        return problems


def parse_session_id(session_id: str) -> tuple[str, str]:
    """Split a session ID into ``(house_id, room)``.

    The grammar is ``h{NN}_{room}_s{NNN}`` — house is the first ``_``-separated
    token, the session counter is the last, and everything between is the room.

    Args:
        session_id: e.g. ``"h01_pooja_room_s002"``.

    Returns:
        Tuple ``(house_id, room)``, e.g. ``("h01", "pooja_room")``.

    Raises:
        ValueError: If the ID has fewer than three ``_``-separated tokens.
    """
    parts = session_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"session id '{session_id}' is not of the form h{{NN}}_{{room}}_s{{NNN}}")
    return parts[0], "_".join(parts[1:-1])


def _normalized_extensions(raw: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Lowercase extensions and ensure each starts with a dot."""
    normalized: list[str] = []
    for ext in raw:
        ext = str(ext).lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(normalized)


def _compile_or_raise(pattern: str, name: str, config_path: Path) -> None:
    """Raise ValueError when a configured regex does not compile."""
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid {name} regex in {config_path}: {e}") from e


def _validate(config: CaptureConfig, config_path: Path) -> None:
    """Imperative range/consistency validation. Raises ValueError on error."""
    _compile_or_raise(config.session_id_pattern, "capture.session_id_pattern", config_path)
    _compile_or_raise(config.consent.reference_pattern, "consent.reference_pattern", config_path)

    if config.image.min_dim <= 0:
        raise ValueError(f"capture.image.min_dim must be positive in {config_path}")
    if config.image.max_file_mb <= 0:
        raise ValueError(f"capture.image.max_file_mb must be positive in {config_path}")
    if not config.image.allowed_extensions:
        raise ValueError(f"capture.image.allowed_extensions must be non-empty in {config_path}")
    if not config.rooms:
        raise ValueError(f"capture.rooms must be non-empty in {config_path}")
    if not config.lighting:
        raise ValueError(f"capture.lighting must be non-empty in {config_path}")

    ann = config.annotation
    if not 0.0 < ann.min_labeled_fraction <= 1.0:
        raise ValueError(f"annotation.min_labeled_fraction must be in (0, 1] in {config_path}")
    for name, value in (
        ("iaa.iou_threshold", ann.iaa.iou_threshold),
        ("iaa.min_agreement", ann.iaa.min_agreement),
        ("iaa.wet_floor_min_agreement", ann.iaa.wet_floor_min_agreement),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"annotation.{name} must be in (0, 1] in {config_path}")

    targets = config.targets
    if targets.total_images <= 0 or targets.min_instances_per_class <= 0:
        raise ValueError(f"targets counts must be positive in {config_path}")
    if targets.min_houses <= 0:
        raise ValueError(f"targets.min_houses must be positive in {config_path}")
    if not targets.custom_classes:
        raise ValueError(f"targets.custom_classes must be non-empty in {config_path}")


def _check_sources_alignment(config: CaptureConfig, sources_config_path: Path | None) -> None:
    """Warn when captures_root drifts from sources.custom_captures.output_dir."""
    from src.dataset.sources_config import DEFAULT_SOURCES_CONFIG_PATH, load_sources_config

    path = sources_config_path if sources_config_path is not None else DEFAULT_SOURCES_CONFIG_PATH
    try:
        sources = load_sources_config(path)
    except (FileNotFoundError, ValueError):
        return  # nothing to cross-check in minimal checkouts
    custom = sources.sources.get("custom_captures")
    if custom is not None and Path(custom.output_dir) != config.captures_root:
        logger.warning(
            f"capture.captures_root ({config.captures_root}) differs from "
            f"sources.custom_captures.output_dir ({custom.output_dir}) — "
            f"the merge stage reads the latter; align the two configs"
        )


def load_capture_config(
    path: Path | None = None,
    sources_config_path: Path | None = None,
) -> CaptureConfig:
    """Load capture settings from YAML, falling back to built-in defaults.

    A missing file is not an error: built-in defaults are returned with a
    warning so unit tests and minimal checkouts stay runnable.

    Args:
        path:                Config path; defaults to configs/capture_config.yaml.
        sources_config_path: dataset_sources.yaml path used only for the
                             captures_root consistency warning.

    Returns:
        Resolved :class:`CaptureConfig`.

    Raises:
        ValueError: If a regex does not compile or a threshold is out of range.
    """
    config_path = path if path is not None else DEFAULT_CAPTURE_CONFIG_PATH
    defaults = CaptureConfig()

    try:
        raw = load_yaml(config_path)
    except FileNotFoundError:
        logger.warning(f"Capture config not found at {config_path} — using built-in defaults")
        return defaults

    capture_raw = raw.get("capture", {}) or {}
    image_raw = capture_raw.get("image", {}) or {}
    consent_raw = raw.get("consent", {}) or {}
    ann_raw = raw.get("annotation", {}) or {}
    iaa_raw = ann_raw.get("iaa", {}) or {}
    targets_raw = raw.get("targets", {}) or {}

    image_defaults = ImageRequirements()
    consent_defaults = ConsentSettings()
    iaa_defaults = IaaSettings()
    ann_defaults = AnnotationSettings()
    targets_defaults = CollectionTargets()

    config = CaptureConfig(
        inbox_dir=Path(capture_raw.get("inbox_dir", defaults.inbox_dir)),
        captures_root=Path(capture_raw.get("captures_root", defaults.captures_root)),
        eval_root=Path(capture_raw.get("eval_root", defaults.eval_root)),
        session_id_pattern=str(capture_raw.get("session_id_pattern", defaults.session_id_pattern)),
        rooms=tuple(capture_raw.get("rooms", list(defaults.rooms)) or []),
        lighting=tuple(capture_raw.get("lighting", list(defaults.lighting)) or []),
        image=ImageRequirements(
            min_dim=int(image_raw.get("min_dim", image_defaults.min_dim)),
            allowed_extensions=_normalized_extensions(
                image_raw.get("allowed_extensions", list(image_defaults.allowed_extensions)) or []
            ),
            strip_metadata=bool(image_raw.get("strip_metadata", image_defaults.strip_metadata)),
            max_file_mb=int(image_raw.get("max_file_mb", image_defaults.max_file_mb)),
        ),
        consent=ConsentSettings(
            registry_path=Path(consent_raw.get("registry_path", consent_defaults.registry_path)),
            reference_pattern=str(
                consent_raw.get("reference_pattern", consent_defaults.reference_pattern)
            ),
            required=bool(consent_raw.get("required", consent_defaults.required)),
        ),
        annotation=AnnotationSettings(
            staging_dir=Path(ann_raw.get("staging_dir", ann_defaults.staging_dir)),
            min_labeled_fraction=float(
                ann_raw.get("min_labeled_fraction", ann_defaults.min_labeled_fraction)
            ),
            iaa=IaaSettings(
                iou_threshold=float(iaa_raw.get("iou_threshold", iaa_defaults.iou_threshold)),
                min_agreement=float(iaa_raw.get("min_agreement", iaa_defaults.min_agreement)),
                wet_floor_min_agreement=float(
                    iaa_raw.get("wet_floor_min_agreement", iaa_defaults.wet_floor_min_agreement)
                ),
            ),
        ),
        targets=CollectionTargets(
            total_images=int(targets_raw.get("total_images", targets_defaults.total_images)),
            min_instances_per_class=int(
                targets_raw.get("min_instances_per_class", targets_defaults.min_instances_per_class)
            ),
            custom_classes=tuple(
                targets_raw.get("custom_classes", list(targets_defaults.custom_classes)) or []
            ),
            min_houses=int(targets_raw.get("min_houses", targets_defaults.min_houses)),
        ),
    )

    _validate(config, config_path)
    _check_sources_alignment(config, sources_config_path)

    logger.info(
        f"Capture config loaded from {config_path}: captures_root={config.captures_root}, "
        f"{len(config.targets.custom_classes)} custom classes, "
        f"targets {config.targets.total_images} images / "
        f"{config.targets.min_instances_per_class} per class"
    )
    return config

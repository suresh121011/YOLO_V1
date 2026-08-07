"""
src.scenario_engine.taxonomy — Class Capability Map
===================================================

Answers the question scenarios actually need, which is **not** the one a
taxonomy fingerprint answers.

A fingerprint says "is the taxonomy the same?". A scenario needs "is this class
actually detectable and enabled?" — and those come apart. The documented R24
decision demotes ``wet_floor`` to scene-level with **class ID 20 reserved and no
renumber**, so ``nc`` stays 23, ``names[20]`` stays ``wet_floor``, and the
fingerprint is byte-identical while the detector emits nothing. Every scenario
requiring it would keep compiling green forever.

The same blindness is already live: ``configs/feature_flags.yaml`` sets
``passport: false`` for privacy, and since Phase-6 M1 that is genuinely enforced
in the detector — so a scenario requiring ``passport`` is dead at runtime and no
fingerprint would notice.

Hence the capability map (ADR-P6-05). It is derived, never authored, and the
compiler hard-fails any scenario whose trigger references a class that is not
bbox-detectable **and** enabled.

Layering note
-------------
``taxonomy_fingerprint`` deliberately **re-implements** the algorithm from
``src/dataset/completeness.py`` rather than importing it. ``src/scenario_engine``
may not depend on ``src.dataset`` (ADR-P6-04), exactly as
``src/dataset/release/gates.py`` duplicates ``GateResult`` rather than importing
from ``src/training``. The two are pinned byte-for-byte by a test, so drift is
caught rather than assumed away.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.config_helpers import get_class_names_from_data_yaml, load_data_config, load_yaml

DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_FLAGS_CONFIG = Path("configs/feature_flags.yaml")
DEFAULT_WET_FLOOR_DECISION = Path("data/qa_reports/wet_floor_decision.json")

#: How a class can be observed. Only ``bbox`` classes can satisfy a trigger.
DETECTION_LEVELS = ("bbox", "scene", "none")


class TaxonomyError(ValueError):
    """Raised when the taxonomy or capability inputs are unusable."""


def taxonomy_fingerprint(nc: int, names: dict[int, str]) -> str:
    """Digest that changes iff ``nc`` or any (id, name) pair changes.

    Mirrors ``src.dataset.completeness.taxonomy_fingerprint`` byte-for-byte;
    see the layering note in this module's docstring.
    """
    canonical = json.dumps(
        {"nc": nc, "names": [[i, names[i]] for i in sorted(names)]},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClassCapability:
    """What the deployed system can actually do with one class."""

    name: str
    class_id: int
    detection_level: str
    enabled: bool
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Can a trigger rely on this class producing boxes?"""
        return self.detection_level == "bbox" and self.enabled

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.class_id,
            "detection_level": self.detection_level,
            "enabled": self.enabled,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class CapabilityMap:
    """The full per-class capability picture, plus the taxonomy fingerprint."""

    fingerprint: str
    classes: dict[str, ClassCapability]

    def names(self) -> frozenset[str]:
        return frozenset(self.classes)

    def unusable(self, class_names: frozenset[str] | set[str]) -> list[ClassCapability]:
        """Referenced classes that cannot satisfy a trigger, worst first."""
        found = [self.classes[n] for n in sorted(class_names) if n in self.classes]
        return [c for c in found if not c.usable]

    def unknown(self, class_names: frozenset[str] | set[str]) -> list[str]:
        """Referenced names that are not in the taxonomy at all."""
        return sorted(n for n in class_names if n not in self.classes)

    def to_dict(self) -> dict[str, Any]:
        return {name: cap.to_dict() for name, cap in sorted(self.classes.items())}


def _read_wet_floor_decision(path: Path) -> tuple[str, str] | None:
    """Read the R24 pilot decision artifact, if one has been recorded.

    Persisted as a small git-tracked artifact rather than re-derived by globbing
    ``data/qa_reports/iaa_*.json``: the compiler must not re-implement
    ``src.dataset.release.gates.read_wet_floor_pilot_decision``, and it may not
    import it either.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaxonomyError(f"{path} exists but is not readable JSON: {exc}") from exc

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in ("keep", "demote"):
        raise TaxonomyError(
            f"{path}: 'decision' must be 'keep' or 'demote', got {decision!r}. "
            f"See docs/04_dataset_engineering/capture_annotation_runbook.md §8."
        )
    return decision, str(payload.get("reason", ""))


def build_capability_map(
    data_config_path: Path | str = DEFAULT_DATA_CONFIG,
    flags_path: Path | str = DEFAULT_FLAGS_CONFIG,
    wet_floor_decision_path: Path | str = DEFAULT_WET_FLOOR_DECISION,
) -> CapabilityMap:
    """Derive the capability map from the live configuration.

    Args:
        data_config_path:        ``configs/data.yaml`` — the only source of class
            names and ids. The repo already carries eight partial duplicates of
            the 23-class list; this adds no ninth.
        flags_path:              ``configs/feature_flags.yaml`` — per-class enable.
        wet_floor_decision_path: The recorded R24 decision, if any.

    Raises:
        TaxonomyError: If the taxonomy cannot be loaded or the decision artifact
            is malformed.
    """
    try:
        data_config = load_data_config(data_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise TaxonomyError(f"Cannot load taxonomy from {data_config_path}: {exc}") from exc

    names = get_class_names_from_data_yaml(data_config)
    if not names:
        raise TaxonomyError(f"{data_config_path} defines no class names")

    nc = int(data_config.get("nc", len(names)))

    enabled_flags: dict[str, bool] = {}
    flags_file = Path(flags_path)
    if flags_file.exists():
        flags = load_yaml(flags_file)
        raw_classes = flags.get("classes", {}) or {}
        if isinstance(raw_classes, dict):
            enabled_flags = {str(k): bool(v) for k, v in raw_classes.items()}

    demotions: dict[str, str] = {}
    decision = _read_wet_floor_decision(Path(wet_floor_decision_path))
    if decision is not None and decision[0] == "demote":
        demotions["wet_floor"] = decision[1] or "R24 pilot demoted wet_floor to scene-level"

    classes: dict[str, ClassCapability] = {}
    for class_id, name in sorted(names.items()):
        enabled = enabled_flags.get(name, True)
        if name in demotions:
            level, reason = "scene", demotions[name]
        else:
            level, reason = "bbox", ""
        if not enabled and not reason:
            reason = f"disabled in {flags_file.name}"
        classes[name] = ClassCapability(
            name=name,
            class_id=class_id,
            detection_level=level,
            enabled=enabled,
            reason=reason,
        )

    return CapabilityMap(fingerprint=taxonomy_fingerprint(nc, names), classes=classes)

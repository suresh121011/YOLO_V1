"""
src.dataset.annotation.base — Auto-Annotator Contract & Value Types
===================================================================

The provider contract every candidate-generating backend implements, plus the
frozen value types that flow through the candidate artifact. Mirrors the
house provider pattern (src/dataset/completeness_policies.py,
src/dataset/splitting/base.py): ABC + decorator registry (registry.py),
frozen dataclasses, fail-loud validation.

Backends must be deterministic for a fixed (weights, config, image) triple on
one machine: sorted image order, fixed seeds, FP32, deterministic NMS
(ADR-P5-02 — cross-machine bit-identity is NOT promised; candidates are
advisory and only human-verified output is exact).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


class AnnotationError(ValueError):
    """Raised when annotation config/artifacts cannot be resolved unambiguously.

    Every ambiguity is a hard error by design (house failure philosophy): a
    wrong candidate silently wastes human verification time, and a wrong
    verified label corrupts supervision.
    """


@dataclass(frozen=True)
class Detection:
    """One candidate detection in normalized YOLO geometry.

    Attributes:
        class_id:   Taxonomy class id (0-based, validated against nc).
        conf:       Backend confidence in [0, 1].
        bbox_xywhn: Normalized (x_center, y_center, width, height), all [0, 1].
        refined:    True when a SAM refinement pass tightened the box.
        origin:     Producing backend name (or source slug for cross_dataset).
    """

    class_id: int
    conf: float
    bbox_xywhn: tuple[float, float, float, float]
    refined: bool = False
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form (artifact convention)."""
        return {
            "class_id": self.class_id,
            "conf": self.conf,
            "bbox_xywhn": list(self.bbox_xywhn),
            "refined": self.refined,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class ModelFingerprint:
    """Reproducibility record for one backend load (embedded in the artifact).

    Attributes:
        backend:            Registered backend name.
        weights_path:       Repo-relative weights path ("" for model-free
                            backends like cross_dataset).
        weights_sha256:     Digest of the weight file ("" when model-free).
        library_versions:   Library → version (e.g. ultralytics, torch).
        device:             Torch device string the model ran on.
        prompt_fingerprint: sha256 over canonical prompts+thresholds JSON —
                            candidates regenerate when prompts change.
    """

    backend: str
    weights_path: str
    weights_sha256: str
    library_versions: Mapping[str, str]
    device: str
    prompt_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form (artifact convention)."""
        return {
            "backend": self.backend,
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "library_versions": dict(self.library_versions),
            "device": self.device,
            "prompt_fingerprint": self.prompt_fingerprint,
        }


@dataclass(frozen=True)
class BackendConfig:
    """Validated per-backend slice of configs/annotation.yaml.

    Attributes:
        name:           Backend name (must match a registered annotator).
        enabled:        Whether the backend participates in generation.
        weights:        Repo-relative weights path ("" for model-free).
        weights_sha256: Expected digest; backends hard-fail on mismatch at
                        load ("" skips the check — only valid pre-pinning,
                        validate() warns via problems list).
        imgsz:          Inference image size.
        conf_floor:     Record-everything floor (per-class thresholds decide
                        candidate status downstream).
        max_det:        Max detections per image.
        prompts:        Taxonomy class name → text prompts (empty tuple =
                        class never prompted; L2 scope honesty).
        thresholds:     Class name → candidate confidence threshold; must
                        contain "default".
        extra:          Backend-specific keys (hf_model, match mode, …).
    """

    name: str
    enabled: bool
    weights: str
    weights_sha256: str
    imgsz: int
    conf_floor: float
    max_det: int
    prompts: Mapping[str, tuple[str, ...]]
    thresholds: Mapping[str, float]
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_annotation_config(cls, name: str, raw: Mapping[str, Any]) -> BackendConfig:
        """Build from one ``auto_annotation.backends.<name>`` config section.

        Args:
            name: Backend name (config key).
            raw:  The backend's mapping from configs/annotation.yaml.

        Raises:
            AnnotationError: On malformed prompt/threshold structures.
        """
        known = {
            "enabled",
            "weights",
            "weights_sha256",
            "imgsz",
            "conf_floor",
            "max_det",
            "prompts",
            "thresholds",
        }
        raw_prompts = raw.get("prompts") or {}
        if not isinstance(raw_prompts, Mapping):
            raise AnnotationError(
                f"Backend '{name}': prompts must be a mapping of class → list, "
                f"got {type(raw_prompts).__name__}"
            )
        prompts: dict[str, tuple[str, ...]] = {}
        for cls_name, entries in raw_prompts.items():
            if entries is None:
                entries = []
            if not isinstance(entries, list) or not all(isinstance(p, str) for p in entries):
                raise AnnotationError(
                    f"Backend '{name}': prompts for class '{cls_name}' must be a "
                    f"list of strings, got {entries!r}"
                )
            prompts[str(cls_name)] = tuple(entries)

        raw_thresholds = raw.get("thresholds") or {"default": 0.25}
        if not isinstance(raw_thresholds, Mapping):
            raise AnnotationError(
                f"Backend '{name}': thresholds must be a mapping, "
                f"got {type(raw_thresholds).__name__}"
            )
        thresholds = {str(k): float(v) for k, v in raw_thresholds.items()}

        return cls(
            name=name,
            enabled=bool(raw.get("enabled", False)),
            weights=str(raw.get("weights", "")),
            weights_sha256=str(raw.get("weights_sha256", "")),
            imgsz=int(raw.get("imgsz", 640)),
            conf_floor=float(raw.get("conf_floor", 0.05)),
            max_det=int(raw.get("max_det", 100)),
            prompts=prompts,
            thresholds=thresholds,
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def validate(self, class_names: Mapping[str, int] | None = None) -> list[str]:
        """Return a list of problems (empty = valid).

        Args:
            class_names: Taxonomy name → id; when given, every prompt and
                         threshold class name must exist in it.
        """
        problems: list[str] = []
        if not 0.0 <= self.conf_floor <= 1.0:
            problems.append(f"conf_floor {self.conf_floor} outside [0, 1]")
        if self.imgsz <= 0:
            problems.append(f"imgsz {self.imgsz} must be positive")
        if self.max_det <= 0:
            problems.append(f"max_det {self.max_det} must be positive")
        if "default" not in self.thresholds:
            problems.append("thresholds must define a 'default' entry")
        for cls_name, value in self.thresholds.items():
            if not 0.0 <= value <= 1.0:
                problems.append(f"threshold for '{cls_name}' ({value}) outside [0, 1]")
        if class_names is not None:
            unknown = sorted(
                (set(self.prompts) | (set(self.thresholds) - {"default"})) - set(class_names)
            )
            if unknown:
                problems.append(
                    f"prompt/threshold class names not in the taxonomy: {unknown} "
                    f"(valid: {sorted(class_names)})"
                )
        return problems

    def threshold_for(self, class_name: str) -> float:
        """Candidate threshold for a class (falls back to 'default')."""
        return self.thresholds.get(class_name, self.thresholds["default"])


def prompt_fingerprint(
    prompts: Mapping[str, tuple[str, ...]], thresholds: Mapping[str, float]
) -> str:
    """Stable fingerprint over prompts + thresholds (canonical JSON).

    Changes iff any prompt string, class set, or threshold changes — recorded
    in every candidates artifact so prompt tuning invalidates stale runs.
    """
    canonical = json.dumps(
        {
            "prompts": {k: list(v) for k, v in sorted(prompts.items())},
            "thresholds": dict(sorted(thresholds.items())),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AutoAnnotator(ABC):
    """Contract for candidate-generating backends.

    Lifecycle: instantiate via the registry → ``load(config, device)`` once →
    ``annotate(image, target_class_ids)`` per image (sorted order for
    determinism) → ``fingerprint()`` recorded into the artifact.
    """

    #: Registry key (set by ``@register_annotator``); also the config section
    #: name under ``auto_annotation.backends``.
    name: ClassVar[str] = ""

    @abstractmethod
    def load(self, config: BackendConfig, device: str) -> None:
        """Load model weights / open resources.

        Args:
            config: Validated backend configuration.
            device: Torch device string (e.g. "cuda:0").

        Raises:
            AnnotationError: On weights-sha256 mismatch or unusable config.
        """

    @abstractmethod
    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        """Return candidate detections for the targeted classes of one image.

        Implementations must return detections ONLY for ``target_class_ids``
        and only above the configured ``conf_floor``.

        Args:
            image_path:       Image file to annotate.
            target_class_ids: Taxonomy class ids to look for (untrusted +
                              unverified cells; computed by targeting.py).
        """

    @abstractmethod
    def fingerprint(self) -> ModelFingerprint:
        """Reproducibility record for the loaded model (post-``load`` only)."""

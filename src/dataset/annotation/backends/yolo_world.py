"""
src.dataset.annotation.backends.yolo_world — YOLO-World Backend
===============================================================

Primary L2 backend (ADR-P5-02): ultralytics-native open-vocabulary detection
via per-class text prompts — zero dependencies beyond the core install (plus
the pinned CLIP text encoder from requirements-annotation.txt, which
``set_classes()`` needs).

Prompt→class mapping: every class with non-empty prompts contributes ALL its
prompt strings to ``set_classes()`` (a class detected under any of its
phrasings counts); the model's predicted class index maps back to the
taxonomy id through ``_prompt_class_ids``. ``annotate()`` filters to the
image's targeted ids — the model is configured once per run, never per image.

Weight pinning (determinism contract): ``load()`` computes the weight file's
sha256 and hard-fails on mismatch; an EMPTY pin also hard-fails, with the
computed digest in the message so the operator copies it into
configs/annotation.yaml (pin-bootstrap flow).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import (
    AnnotationError,
    AutoAnnotator,
    BackendConfig,
    Detection,
    ModelFingerprint,
    prompt_fingerprint,
)
from src.dataset.annotation.registry import register_annotator
from src.utils.dataset_utils import compute_file_hash

logger = logging.getLogger(__name__)


def verify_weights(weights: Path, expected_sha256: str, backend: str) -> str:
    """Verify a weight file exists and matches its pinned digest.

    Args:
        weights:         Weight file path.
        expected_sha256: Pinned digest from configs/annotation.yaml.
        backend:         Backend name (error messages).

    Returns:
        The computed sha256 hex digest.

    Raises:
        AnnotationError: Missing file, empty pin (message carries the
                         computed digest to copy into the config), or
                         digest mismatch.
    """
    if not weights.exists():
        raise AnnotationError(
            f"Backend '{backend}': weights not found at {weights}. Download the "
            f"model into models/annotators/ (see docs/07 auto_annotation_runbook) "
            f"and pin its sha256 in configs/annotation.yaml."
        )
    actual = compute_file_hash(weights)
    if not expected_sha256:
        raise AnnotationError(
            f"Backend '{backend}': weights_sha256 is not pinned in "
            f"configs/annotation.yaml. Computed digest for {weights.name}: "
            f"{actual} — record it under auto_annotation.backends.{backend}."
            f"weights_sha256 (determinism contract, ADR-P5-02)."
        )
    if actual != expected_sha256:
        raise AnnotationError(
            f"Backend '{backend}': weights digest mismatch for {weights} — "
            f"expected {expected_sha256}, got {actual}. The file changed or the "
            f"pin is stale; re-download or re-pin deliberately."
        )
    return actual


@register_annotator("yolo_world")
class YoloWorldBackend(AutoAnnotator):
    """Open-vocabulary candidates via ultralytics YOLO-World."""

    def __init__(self) -> None:
        self._model: Any = None
        self._config: BackendConfig | None = None
        self._device = ""
        self._weights_sha256 = ""
        #: Prompt index (model class) → taxonomy class id.
        self._prompt_class_ids: list[int] = []

    def load(self, config: BackendConfig, device: str, ids_by_name: Mapping[str, int]) -> None:
        """Verify weights, load the model, and configure the prompt classes.

        Args:
            config:      Backend configuration (prompts drive ``set_classes``).
            device:      Torch device string.
            ids_by_name: Taxonomy class name → id.

        Raises:
            AnnotationError: On pin problems, a promptless configuration, or
                             prompted classes missing from the taxonomy.
        """
        weights = Path(config.weights)
        self._weights_sha256 = verify_weights(weights, config.weights_sha256, "yolo_world")

        prompt_strings: list[str] = []
        prompt_class_ids: list[int] = []
        unknown = sorted(n for n, p in config.prompts.items() if p and n not in ids_by_name)
        if unknown:
            raise AnnotationError(
                f"Backend 'yolo_world': prompted classes not in the taxonomy: {unknown}"
            )
        for class_name in sorted(config.prompts):
            for prompt in config.prompts[class_name]:
                prompt_strings.append(prompt)
                prompt_class_ids.append(ids_by_name[class_name])
        if not prompt_strings:
            raise AnnotationError(
                "Backend 'yolo_world': every prompt list is empty — nothing to "
                "detect. Configure prompts in configs/annotation.yaml."
            )

        import torch  # heavy import stays inside load() (house pattern)
        from ultralytics import YOLO

        torch.manual_seed(0)
        model = YOLO(str(weights))
        model.set_classes(prompt_strings)

        self._model = model
        self._config = config
        self._device = device
        self._prompt_class_ids = prompt_class_ids
        logger.info(
            f"yolo_world loaded: {weights.name} ({len(prompt_strings)} prompts, "
            f"device={device})"
        )

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        """Run one deterministic prediction and map/filter detections.

        Args:
            image_path:       Image file.
            target_class_ids: Taxonomy ids to keep (targeting.py output).

        Raises:
            RuntimeError: If called before ``load``.
        """
        if self._model is None or self._config is None or not self._prompt_class_ids:
            raise RuntimeError("YoloWorldBackend.load() must be called before annotate()")

        results = self._model.predict(
            source=str(image_path),
            imgsz=self._config.imgsz,
            conf=self._config.conf_floor,
            max_det=self._config.max_det,
            device=self._device,
            verbose=False,
        )
        targets = set(target_class_ids)
        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for prompt_idx, conf, xywhn in zip(
            boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xywhn.tolist(), strict=True
        ):
            if not 0 <= prompt_idx < len(self._prompt_class_ids):
                raise AnnotationError(
                    f"yolo_world returned class index {prompt_idx} outside the "
                    f"configured prompt list ({len(self._prompt_class_ids)} prompts) "
                    f"for {image_path.name} — model/prompt configuration drift."
                )
            class_id = self._prompt_class_ids[prompt_idx]
            if class_id not in targets:
                continue
            x, y, w, h = (float(v) for v in xywhn)
            detections.append(
                Detection(
                    class_id=class_id,
                    conf=float(conf),
                    bbox_xywhn=(x, y, w, h),
                    refined=False,
                    origin="yolo_world",
                )
            )
        return detections

    def fingerprint(self) -> ModelFingerprint:
        """Reproducibility record (post-load only)."""
        if self._config is None:
            raise RuntimeError("YoloWorldBackend.load() must be called before fingerprint()")
        import torch
        import ultralytics

        return ModelFingerprint(
            backend="yolo_world",
            weights_path=self._config.weights,
            weights_sha256=self._weights_sha256,
            library_versions={
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
            },
            device=self._device,
            prompt_fingerprint=prompt_fingerprint(self._config.prompts, self._config.thresholds),
        )

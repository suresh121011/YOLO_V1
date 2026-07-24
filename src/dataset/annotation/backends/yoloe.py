"""
src.dataset.annotation.backends.yoloe — YOLOE Backend
=====================================================

Open-vocabulary seeder upgrade over ``yolo_world`` (P5, annotation V2 plan).
YOLOE ("Real-Time Seeing Anything", 2025) is ultralytics-native and improves
on YOLO-Worldv2 (+3.5 AP on LVIS at 1.4× speed) while keeping the same
prompt→detect flow, so it slots in as a drop-in alternate backend behind the
same :class:`AutoAnnotator` contract.

Key API difference vs YOLO-World: YOLOE binds text prompts through
``set_classes(names, model.get_text_pe(names))`` — the class embeddings are
computed once at load, not per image. Everything else (per-prompt→class
mapping, target filtering, weight-pin determinism contract, NMS knobs) is
identical to :mod:`.yolo_world`.

Disabled by default and unpinned until the operator downloads a YOLOE weight
into ``models/annotators/`` and records its sha256 (the same pin-bootstrap
flow ``load()`` enforces for every backend).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.dataset.annotation.backends.yolo_world import verify_weights
from src.dataset.annotation.base import (
    AnnotationError,
    AutoAnnotator,
    BackendConfig,
    Detection,
    ModelFingerprint,
    prompt_fingerprint,
)
from src.dataset.annotation.registry import register_annotator

logger = logging.getLogger(__name__)


@register_annotator("yoloe")
class YoloeBackend(AutoAnnotator):
    """Open-vocabulary candidates via ultralytics YOLOE."""

    def __init__(self) -> None:
        self._model: Any = None
        self._config: BackendConfig | None = None
        self._device = ""
        self._weights_sha256 = ""
        #: Prompt index (model class) → taxonomy class id.
        self._prompt_class_ids: list[int] = []

    def load(self, config: BackendConfig, device: str, ids_by_name: Mapping[str, int]) -> None:
        """Verify weights, load YOLOE, and bind the text prompt embeddings.

        Args:
            config:      Backend configuration (prompts drive ``set_classes``).
            device:      Torch device string.
            ids_by_name: Taxonomy class name → id.

        Raises:
            AnnotationError: On pin problems, a promptless configuration, or
                             prompted classes missing from the taxonomy.
        """
        weights = Path(config.weights)
        self._weights_sha256 = verify_weights(weights, config.weights_sha256, "yoloe")

        prompt_strings: list[str] = []
        prompt_class_ids: list[int] = []
        unknown = sorted(n for n, p in config.prompts.items() if p and n not in ids_by_name)
        if unknown:
            raise AnnotationError(
                f"Backend 'yoloe': prompted classes not in the taxonomy: {unknown}"
            )
        for class_name in sorted(config.prompts):
            for prompt in config.prompts[class_name]:
                prompt_strings.append(prompt)
                prompt_class_ids.append(ids_by_name[class_name])
        if not prompt_strings:
            raise AnnotationError(
                "Backend 'yoloe': every prompt list is empty — nothing to detect. "
                "Configure prompts in configs/annotation.yaml."
            )

        import torch  # heavy import stays inside load() (house pattern)
        from ultralytics import YOLOE

        torch.manual_seed(0)
        model = YOLOE(str(weights))
        # YOLOE binds text prompts via precomputed text embeddings (get_text_pe).
        model.set_classes(prompt_strings, model.get_text_pe(prompt_strings))

        self._model = model
        self._config = config
        self._device = device
        self._prompt_class_ids = prompt_class_ids
        logger.info(
            f"yoloe loaded: {weights.name} ({len(prompt_strings)} prompts, device={device})"
        )

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        """Run one deterministic prediction and map/filter detections."""
        if self._model is None or self._config is None or not self._prompt_class_ids:
            raise RuntimeError("YoloeBackend.load() must be called before annotate()")

        results = self._model.predict(
            source=str(image_path),
            imgsz=self._config.imgsz,
            conf=self._config.conf_floor,
            iou=self._config.iou,
            agnostic_nms=self._config.agnostic_nms,
            max_det=self._config.max_det,
            device=self._device,
            verbose=False,
        )
        return self._detections_from_result(results[0], image_path, set(target_class_ids))

    def _detections_from_result(
        self, result: Any, image_path: Path, targets: set[int]
    ) -> list[Detection]:
        """Map one ultralytics result's boxes → targeted :class:`Detection`s.

        Shared by :meth:`annotate` and :meth:`annotate_batch` so single and
        batched paths produce byte-identical detections.
        """
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for prompt_idx, conf, xywhn in zip(
            boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xywhn.tolist(), strict=True
        ):
            if not 0 <= prompt_idx < len(self._prompt_class_ids):
                raise AnnotationError(
                    f"yoloe returned class index {prompt_idx} outside the configured "
                    f"prompt list ({len(self._prompt_class_ids)} prompts) for "
                    f"{image_path.name} — model/prompt configuration drift."
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
                    origin="yoloe",
                )
            )
        return detections

    def annotate_batch(
        self, image_paths: list[Path], target_class_ids: tuple[int, ...]
    ) -> list[list[Detection]]:
        """Batched inference — one ``predict`` over many images.

        Returns one detection list per input path, in input order. Semantically
        identical to calling :meth:`annotate` per image, but a single batched
        forward pass is far faster at dataset scale (P5 motivation).
        """
        if self._model is None or self._config is None or not self._prompt_class_ids:
            raise RuntimeError("YoloeBackend.load() must be called before annotate_batch()")
        if not image_paths:
            return []
        results = self._model.predict(
            source=[str(p) for p in image_paths],
            imgsz=self._config.imgsz,
            conf=self._config.conf_floor,
            iou=self._config.iou,
            agnostic_nms=self._config.agnostic_nms,
            max_det=self._config.max_det,
            device=self._device,
            verbose=False,
        )
        targets = set(target_class_ids)
        return [
            self._detections_from_result(res, path, targets)
            for res, path in zip(results, image_paths, strict=True)
        ]

    def fingerprint(self) -> ModelFingerprint:
        """Reproducibility record (post-load only)."""
        if self._config is None:
            raise RuntimeError("YoloeBackend.load() must be called before fingerprint()")
        import torch
        import ultralytics

        return ModelFingerprint(
            backend="yoloe",
            weights_path=self._config.weights,
            weights_sha256=self._weights_sha256,
            library_versions={
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
            },
            device=self._device,
            prompt_fingerprint=prompt_fingerprint(
                self._config.prompts,
                self._config.thresholds,
                {"iou": self._config.iou, "agnostic_nms": self._config.agnostic_nms},
            ),
        )

"""
src.dataset.annotation.backends.grounding_dino — GroundingDINO Backend (ADR-P5-02)
======================================================================================

Optional second-opinion open-vocabulary backend (HF transformers,
``IDEA-Research/grounding-dino-base`` by default). Disabled by default; M8's
``grounding_dino_decision()`` (src/dataset/annotation/coverage.py) recommends
enabling it only once a priority class's CALIBRATED yolo_world precision
falls below ``configs/annotation.yaml``'s ``enable_below_precision``
threshold — this backend is the thing that decision recommends turning on.

Pin (determinism contract, ADR-P5-02): ``hf_revision`` (a commit sha on the
HF hub) is required non-empty, mirroring yolo_world's ``weights_sha256``
hard-fail — an unpinned revision resolves to "latest", which silently
drifts underneath a fixed config.

Prompt→class mapping mirrors yolo_world: every class with non-empty prompts
contributes all its prompt strings to one text query; GroundingDINO returns
the matched phrase itself (not an index), so the reverse map here is
text -> class id rather than yolo_world's index -> class id.
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

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "IDEA-Research/grounding-dino-base"


def verify_revision(hf_revision: str, backend: str = "grounding_dino") -> None:
    """Hard-fail on an unpinned HF revision.

    Args:
        hf_revision: ``auto_annotation.backends.grounding_dino.hf_revision``.
        backend:     Backend name (error messages).

    Raises:
        AnnotationError: If ``hf_revision`` is empty.
    """
    if not hf_revision:
        raise AnnotationError(
            f"Backend '{backend}': hf_revision is not pinned in "
            f"configs/annotation.yaml. An empty revision resolves to the HF hub's "
            f"'latest', which silently drifts underneath a fixed config (ADR-P5-02 "
            f"determinism contract) — pin a commit sha under "
            f"auto_annotation.backends.{backend}.hf_revision."
        )


@register_annotator("grounding_dino")
class GroundingDinoBackend(AutoAnnotator):
    """Second-opinion candidates via HF GroundingDINO zero-shot detection."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._config: BackendConfig | None = None
        self._device = ""
        self._hf_model = ""
        self._hf_revision = ""
        #: class_id -> its own prompt strings (a class may have >1 phrasing).
        self._prompts_by_class_id: dict[int, tuple[str, ...]] = {}
        #: Matched phrase (lowercased, as the processor returns it) -> class_id.
        self._class_id_by_prompt: dict[str, int] = {}

    def load(self, config: BackendConfig, device: str, ids_by_name: Mapping[str, int]) -> None:
        """Verify the HF revision pin, load the model, and configure prompt classes.

        Args:
            config:      Backend configuration (``extra["hf_model"/"hf_revision"]``).
            device:      Torch device string.
            ids_by_name: Taxonomy class name → id.

        Raises:
            AnnotationError: On an unpinned revision, a promptless
                             configuration, or prompted classes missing
                             from the taxonomy.
        """
        hf_model = str(config.extra.get("hf_model", DEFAULT_HF_MODEL))
        hf_revision = str(config.extra.get("hf_revision", ""))
        verify_revision(hf_revision, "grounding_dino")

        unknown = sorted(n for n, p in config.prompts.items() if p and n not in ids_by_name)
        if unknown:
            raise AnnotationError(
                f"Backend 'grounding_dino': prompted classes not in the taxonomy: {unknown}"
            )
        prompts_by_class_id: dict[int, tuple[str, ...]] = {}
        class_id_by_prompt: dict[str, int] = {}
        for class_name in sorted(config.prompts):
            prompts = config.prompts[class_name]
            if not prompts:
                continue
            class_id = ids_by_name[class_name]
            prompts_by_class_id[class_id] = prompts
            for prompt in prompts:
                class_id_by_prompt[prompt.lower().strip()] = class_id
        if not class_id_by_prompt:
            raise AnnotationError(
                "Backend 'grounding_dino': every prompt list is empty — nothing to "
                "detect. Configure prompts in configs/annotation.yaml."
            )

        import torch  # heavy import stays inside load() (house pattern)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        torch.manual_seed(0)
        self._processor = AutoProcessor.from_pretrained(hf_model, revision=hf_revision)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            hf_model, revision=hf_revision
        ).to(device)
        self._model.eval()

        self._config = config
        self._device = device
        self._hf_model = hf_model
        self._hf_revision = hf_revision
        self._prompts_by_class_id = prompts_by_class_id
        self._class_id_by_prompt = class_id_by_prompt
        logger.info(
            f"grounding_dino loaded: {hf_model}@{hf_revision} "
            f"({len(class_id_by_prompt)} prompts, device={device})"
        )

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        """Run one deterministic prediction and map/filter detections.

        Args:
            image_path:       Image file.
            target_class_ids: Taxonomy ids to keep (targeting.py output);
                              also narrows the text query to just these
                              classes' prompts (shorter, more precise query).

        Raises:
            RuntimeError: If called before ``load``.
        """
        if self._model is None or self._processor is None or self._config is None:
            raise RuntimeError("GroundingDinoBackend.load() must be called before annotate()")

        targets = set(target_class_ids)
        text_prompts = sorted(
            {p for cid in targets for p in self._prompts_by_class_id.get(cid, ())}
        )
        if not text_prompts:
            return []

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        text_query = ". ".join(text_prompts) + "."
        box_threshold = float(self._config.extra.get("box_threshold", 0.30))
        text_threshold = float(self._config.extra.get("text_threshold", 0.25))

        inputs = self._processor(images=image, text=text_query, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        width, height = image.size
        detections: list[Detection] = []
        for box, label, score in zip(
            results["boxes"].tolist(), results["labels"], results["scores"].tolist(), strict=True
        ):
            class_id = self._class_id_by_prompt.get(str(label).lower().strip())
            if class_id is None or class_id not in targets:
                continue
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    class_id=class_id,
                    conf=float(score),
                    bbox_xywhn=(
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ),
                    refined=False,
                    origin="grounding_dino",
                )
            )
        return detections

    def fingerprint(self) -> ModelFingerprint:
        """Reproducibility record (post-``load`` only). Pin is hf_revision, not a file hash."""
        if self._config is None:
            raise RuntimeError("GroundingDinoBackend.load() must be called before fingerprint()")
        import torch
        import transformers

        return ModelFingerprint(
            backend="grounding_dino",
            weights_path=f"{self._hf_model}@{self._hf_revision}",
            weights_sha256="",
            library_versions={
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
            device=self._device,
            prompt_fingerprint=prompt_fingerprint(self._config.prompts, self._config.thresholds),
        )

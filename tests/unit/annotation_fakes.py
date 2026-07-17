"""Test-only fake auto-annotator (ADR-P5-11: CI never loads real backends).

Import as ``from unit.annotation_fakes import FakeAnnotator`` (pytest maps
tests/unit to package ``unit`` — tests/ itself is not a package). Import the
module, never copy the class, so registration happens exactly once per
process; the registry rejects duplicate names by design.
"""

from __future__ import annotations

from pathlib import Path

from src.dataset.annotation.base import (
    AutoAnnotator,
    BackendConfig,
    Detection,
    ModelFingerprint,
    prompt_fingerprint,
)
from src.dataset.annotation.registry import register_annotator


@register_annotator("fake")
class FakeAnnotator(AutoAnnotator):
    """Deterministic model-free backend for tests and the M6 pipeline drill.

    Emits one centered candidate per targeted class id with
    ``conf = 0.9 - 0.1 * (index of class in the targeted tuple)``, floored at
    the configured ``conf_floor`` — deterministic, image-content-independent.
    """

    def __init__(self) -> None:
        self._config: BackendConfig | None = None
        self._device = ""

    def load(self, config: BackendConfig, device: str) -> None:
        self._config = config
        self._device = device

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        if self._config is None:
            raise RuntimeError("FakeAnnotator.load() must be called before annotate()")
        floor = self._config.conf_floor
        return [
            Detection(
                class_id=class_id,
                conf=max(floor, round(0.9 - 0.1 * i, 2)),
                bbox_xywhn=(0.5, 0.5, 0.2, 0.2),
                refined=False,
                origin="fake",
            )
            for i, class_id in enumerate(target_class_ids)
        ]

    def fingerprint(self) -> ModelFingerprint:
        if self._config is None:
            raise RuntimeError("FakeAnnotator.load() must be called before fingerprint()")
        return ModelFingerprint(
            backend="fake",
            weights_path="",
            weights_sha256="",
            library_versions={},
            device=self._device,
            prompt_fingerprint=prompt_fingerprint(self._config.prompts, self._config.thresholds),
        )

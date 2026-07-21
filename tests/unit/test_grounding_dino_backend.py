"""Unit tests for the grounding_dino backend (mocked model + processor).

The real model never loads here — transformers.AutoProcessor /
AutoModelForZeroShotObjectDetection are monkeypatched — so these act as
contract tests for revision pinning, prompt->class mapping, and
target-class filtering. importorskip: transformers is an optional extra
(ADR-P5-11), absent from the default CI environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")

from src.dataset.annotation.backends.grounding_dino import (  # noqa: E402
    GroundingDinoBackend,
    verify_revision,
)
from src.dataset.annotation.base import AnnotationError, BackendConfig  # noqa: E402

pytestmark = pytest.mark.unit

_IDS = {"charger": 10, "wire": 11, "cupboard": 14}


def _config(
    hf_revision: str = "abc123",
    prompts: dict[str, list[str]] | None = None,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "grounding_dino",
        {
            "enabled": True,
            "weights": "",
            "weights_sha256": "",
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": (
                prompts
                if prompts is not None
                else {"charger": ["phone charger", "power adapter"], "wire": ["cable"]}
            ),
            "thresholds": {"default": 0.25},
            "hf_model": "IDEA-Research/grounding-dino-base",
            "hf_revision": hf_revision,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        },
    )


class _FakeProcessor:
    instances: list[_FakeProcessor] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.post_process_result: dict[str, Any] = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]),
            "labels": ["phone charger", "cable"],
            "scores": torch.tensor([0.9, 0.4]),
        }
        self.call_kwargs: dict[str, Any] | None = None
        self.post_process_kwargs: dict[str, Any] | None = None
        _FakeProcessor.instances.append(self)

    @classmethod
    def from_pretrained(cls, hf_model: str, revision: str) -> _FakeProcessor:
        return cls()

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.call_kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]]), "pixel_values": torch.zeros(1, 3, 4, 4)}

    def post_process_grounded_object_detection(
        self, outputs: Any, input_ids: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.post_process_kwargs = kwargs
        return [self.post_process_result]


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self) -> None:
        _FakeModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, hf_model: str, revision: str) -> _FakeModel:
        return cls()

    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> None:
        pass

    def __call__(self, **kwargs: Any) -> Any:
        return SimpleNamespace()


@pytest.fixture(autouse=True)
def _patch_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the WHOLE `transformers` module in sys.modules, not just
    setattr individual names on it — transformers' real __init__.py is a
    _LazyModule with a custom __getattr__/caching scheme that silently
    ignores plain monkeypatch.setattr overrides and still resolves
    `from transformers import AutoProcessor` to the REAL class (confirmed:
    a naive monkeypatch.setattr(transformers, "AutoProcessor", Fake) here
    made real HTTP calls to huggingface.co instead of using the fake).
    Swapping the sys.modules entry sidesteps _LazyModule entirely — `from
    transformers import X` then just does a plain getattr on our fake."""
    _FakeProcessor.instances = []
    _FakeModel.instances = []
    fake_transformers = SimpleNamespace(
        AutoProcessor=_FakeProcessor,
        AutoModelForZeroShotObjectDetection=_FakeModel,
        __version__=transformers.__version__,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


class TestVerifyRevision:
    def test_empty_revision_raises(self) -> None:
        with pytest.raises(AnnotationError, match="not pinned"):
            verify_revision("")

    def test_pinned_revision_passes(self) -> None:
        verify_revision("abc123")  # no raise


class TestLoad:
    def test_unpinned_revision_blocks_load(self) -> None:
        backend = GroundingDinoBackend()
        with pytest.raises(AnnotationError, match="not pinned"):
            backend.load(_config(hf_revision=""), "cpu", _IDS)

    def test_unknown_prompt_class_raises(self) -> None:
        backend = GroundingDinoBackend()
        with pytest.raises(AnnotationError, match="ghost"):
            backend.load(_config(prompts={"ghost": ["x"]}), "cpu", _IDS)

    def test_all_empty_prompts_raise(self) -> None:
        backend = GroundingDinoBackend()
        with pytest.raises(AnnotationError, match="empty"):
            backend.load(_config(prompts={"charger": []}), "cpu", _IDS)

    def test_successful_load_registers_prompt_map(self) -> None:
        backend = GroundingDinoBackend()
        backend.load(_config(), "cpu", _IDS)
        assert backend._class_id_by_prompt == {
            "phone charger": 10,
            "power adapter": 10,
            "cable": 11,
        }


class TestAnnotate:
    def _loaded(self) -> GroundingDinoBackend:
        backend = GroundingDinoBackend()
        backend.load(_config(), "cpu", _IDS)
        return backend

    def test_annotate_before_load_raises(self) -> None:
        backend = GroundingDinoBackend()
        with pytest.raises(RuntimeError):
            backend.annotate(Path("x.jpg"), target_class_ids=(10,))

    def test_no_targeted_prompts_returns_empty_without_calling_model(self, tmp_path: Path) -> None:
        # load() itself instantiates the processor — assert INFERENCE never
        # ran (processor.__call__/post_process), not that it was never built.
        backend = self._loaded()
        # cupboard (14) has no configured prompts.
        result = backend.annotate(tmp_path / "x.jpg", target_class_ids=(14,))
        assert result == []
        assert _FakeProcessor.instances[-1].call_kwargs is None
        assert _FakeProcessor.instances[-1].post_process_kwargs is None

    def test_matched_detections_mapped_and_filtered_to_targets(self, tmp_path: Path) -> None:
        from PIL import Image

        img_path = tmp_path / "x.jpg"
        Image.new("RGB", (100, 100)).save(img_path)

        backend = self._loaded()
        detections = backend.annotate(img_path, target_class_ids=(10, 11))
        assert len(detections) == 2
        charger = next(d for d in detections if d.class_id == 10)
        assert charger.origin == "grounding_dino"
        assert charger.conf == pytest.approx(0.9)
        # box (10,10,30,30) on a 100x100 image -> cx=0.2, cy=0.2, w=0.2, h=0.2
        assert charger.bbox_xywhn == pytest.approx((0.2, 0.2, 0.2, 0.2))

    def test_detection_outside_targets_is_filtered_out(self, tmp_path: Path) -> None:
        from PIL import Image

        img_path = tmp_path / "x.jpg"
        Image.new("RGB", (100, 100)).save(img_path)

        backend = self._loaded()
        detections = backend.annotate(img_path, target_class_ids=(10,))  # wire (11) not targeted
        assert {d.class_id for d in detections} == {10}

    def test_thresholds_passed_through_to_post_process(self, tmp_path: Path) -> None:
        from PIL import Image

        img_path = tmp_path / "x.jpg"
        Image.new("RGB", (100, 100)).save(img_path)

        backend = GroundingDinoBackend()
        backend.load(_config(box_threshold=0.5, text_threshold=0.4), "cpu", _IDS)
        backend.annotate(img_path, target_class_ids=(10, 11))
        kwargs = _FakeProcessor.instances[-1].post_process_kwargs
        assert kwargs is not None
        assert kwargs["box_threshold"] == 0.5
        assert kwargs["text_threshold"] == 0.4


class TestFingerprint:
    def test_fingerprint_before_load_raises(self) -> None:
        backend = GroundingDinoBackend()
        with pytest.raises(RuntimeError):
            backend.fingerprint()

    def test_fingerprint_has_no_local_weights_hash(self) -> None:
        backend = GroundingDinoBackend()
        backend.load(_config(hf_revision="deadbeef"), "cpu", _IDS)
        fp = backend.fingerprint()
        assert fp.backend == "grounding_dino"
        assert fp.weights_sha256 == ""
        assert fp.weights_path == "IDEA-Research/grounding-dino-base@deadbeef"

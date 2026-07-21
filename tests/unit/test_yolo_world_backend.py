"""Unit tests for the yolo_world backend + weight pinning (mocked model).

The real model never loads here — ultralytics.YOLO is monkeypatched — so
these run in CI (torch/ultralytics installed) and act as contract tests for
prompt→class mapping, target filtering, and the pin-bootstrap flow.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from src.dataset.annotation.backends.yolo_world import (  # noqa: E402
    YoloWorldBackend,
    verify_weights,
)
from src.dataset.annotation.base import AnnotationError, BackendConfig  # noqa: E402

pytestmark = pytest.mark.unit

_IDS = {"charger": 10, "wire": 11, "cupboard": 14}


def _weights(tmp_path: Path, content: bytes = b"fake-weights") -> tuple[Path, str]:
    path = tmp_path / "yolov8x-worldv2.pt"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _config(weights: Path, sha: str, prompts: dict[str, list[str]] | None = None) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "yolo_world",
        {
            "enabled": True,
            "weights": str(weights),
            "weights_sha256": sha,
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": (
                prompts
                if prompts is not None
                else {"charger": ["phone charger", "power adapter"], "wire": ["cable"]}
            ),
            "thresholds": {"default": 0.25},
        },
    )


class _FakeYolo:
    """Records set_classes/predict calls; returns configured boxes."""

    instances: list[_FakeYolo] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.classes: list[str] | None = None
        self.predict_kwargs: dict[str, Any] | None = None
        self.boxes: Any = SimpleNamespace(
            cls=torch.tensor([0, 2]),  # prompt indices
            conf=torch.tensor([0.9, 0.4]),
            xywhn=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        )
        _FakeYolo.instances.append(self)

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes

    def predict(self, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.fixture(autouse=True)
def _patch_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYolo.instances = []
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYolo)


class TestVerifyWeights:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AnnotationError, match="not found"):
            verify_weights(tmp_path / "absent.pt", "x", "yolo_world")

    def test_empty_pin_raises_with_computed_digest(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        with pytest.raises(AnnotationError, match=sha):
            verify_weights(path, "", "yolo_world")

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        path, _ = _weights(tmp_path)
        with pytest.raises(AnnotationError, match="digest mismatch"):
            verify_weights(path, "0" * 64, "yolo_world")

    def test_match_returns_digest(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        assert verify_weights(path, sha, "yolo_world") == sha


class TestLoad:
    def test_prompts_flattened_in_sorted_class_order(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        backend = YoloWorldBackend()
        backend.load(_config(path, sha), "cpu", _IDS)
        model = _FakeYolo.instances[-1]
        # charger sorts before wire; charger has two prompts
        assert model.classes == ["phone charger", "power adapter", "cable"]

    def test_unknown_prompt_class_raises(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        backend = YoloWorldBackend()
        with pytest.raises(AnnotationError, match="ghost"):
            backend.load(_config(path, sha, {"ghost": ["x"]}), "cpu", _IDS)

    def test_all_empty_prompts_raise(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        backend = YoloWorldBackend()
        with pytest.raises(AnnotationError, match="empty"):
            backend.load(_config(path, sha, {"charger": []}), "cpu", _IDS)

    def test_bad_pin_blocks_load(self, tmp_path: Path) -> None:
        path, _ = _weights(tmp_path)
        backend = YoloWorldBackend()
        with pytest.raises(AnnotationError, match="digest mismatch"):
            backend.load(_config(path, "0" * 64), "cpu", _IDS)


class TestAnnotate:
    def _loaded(self, tmp_path: Path) -> tuple[YoloWorldBackend, _FakeYolo]:
        path, sha = _weights(tmp_path)
        backend = YoloWorldBackend()
        backend.load(_config(path, sha), "cpu", _IDS)
        return backend, _FakeYolo.instances[-1]

    def test_prompt_indices_map_to_taxonomy_ids(self, tmp_path: Path) -> None:
        backend, _ = self._loaded(tmp_path)
        # prompt idx 0/1 → charger (10), idx 2 → wire (11)
        detections = backend.annotate(tmp_path / "img.jpg", (10, 11))
        assert [d.class_id for d in detections] == [10, 11]
        assert [round(d.conf, 2) for d in detections] == [0.9, 0.4]
        assert all(d.origin == "yolo_world" for d in detections)

    def test_untargeted_classes_filtered(self, tmp_path: Path) -> None:
        backend, _ = self._loaded(tmp_path)
        detections = backend.annotate(tmp_path / "img.jpg", (11,))
        assert [d.class_id for d in detections] == [11]

    def test_predict_receives_config_values(self, tmp_path: Path) -> None:
        backend, model = self._loaded(tmp_path)
        backend.annotate(tmp_path / "img.jpg", (10,))
        assert model.predict_kwargs is not None
        assert model.predict_kwargs["imgsz"] == 640
        assert model.predict_kwargs["conf"] == 0.05
        assert model.predict_kwargs["max_det"] == 100
        assert model.predict_kwargs["verbose"] is False

    def test_out_of_range_prompt_index_raises(self, tmp_path: Path) -> None:
        backend, model = self._loaded(tmp_path)
        model.boxes = SimpleNamespace(
            cls=torch.tensor([99]),
            conf=torch.tensor([0.9]),
            xywhn=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        )
        with pytest.raises(AnnotationError, match="outside the"):
            backend.annotate(tmp_path / "img.jpg", (10,))

    def test_no_boxes_returns_empty(self, tmp_path: Path) -> None:
        backend, model = self._loaded(tmp_path)
        model.boxes = None
        assert backend.annotate(tmp_path / "img.jpg", (10,)) == []

    def test_requires_load(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="load"):
            YoloWorldBackend().annotate(tmp_path / "img.jpg", (10,))


class TestFingerprint:
    def test_records_versions_and_pin(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        backend = YoloWorldBackend()
        backend.load(_config(path, sha), "cuda:0", _IDS)
        fp = backend.fingerprint()
        assert fp.backend == "yolo_world"
        assert fp.weights_sha256 == sha
        assert fp.device == "cuda:0"
        assert fp.library_versions["ultralytics"] == ultralytics.__version__
        assert fp.library_versions["torch"] == torch.__version__

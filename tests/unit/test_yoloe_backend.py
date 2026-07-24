"""Unit tests for the yoloe backend (mocked model).

ultralytics.YOLOE is monkeypatched so these run in CI without real weights —
contract tests for prompt→class mapping, NMS plumbing, batched inference
equivalence, and the pin-bootstrap flow (shared with yolo_world).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from src.dataset.annotation.backends.yoloe import YoloeBackend  # noqa: E402
from src.dataset.annotation.base import AnnotationError, BackendConfig  # noqa: E402

pytestmark = pytest.mark.unit

_IDS = {"charger": 10, "wire": 11, "cupboard": 14, "face": 1, "medicine_bottle": 3}


def _weights(tmp_path: Path, content: bytes = b"fake-yoloe-weights") -> tuple[Path, str]:
    path = tmp_path / "yoloe-11l-seg.pt"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _config(
    weights: Path,
    sha: str,
    prompts: dict[str, list[str]] | None = None,
    iou: float = 0.7,
    agnostic_nms: bool = False,
) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "yoloe",
        {
            "enabled": True,
            "weights": str(weights),
            "weights_sha256": sha,
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "iou": iou,
            "agnostic_nms": agnostic_nms,
            "prompts": (
                prompts
                if prompts is not None
                else {"charger": ["phone charger", "power adapter"], "wire": ["cable"]}
            ),
            "thresholds": {"default": 0.25},
        },
    )


def _boxes() -> Any:
    # Two detections: prompt idx 0 (charger) and 2 (wire) for the default prompts.
    return SimpleNamespace(
        cls=torch.tensor([0, 2]),
        conf=torch.tensor([0.9, 0.4]),
        xywhn=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
    )


class _FakeYoloe:
    """Records set_classes/get_text_pe/predict; returns configured boxes."""

    instances: list[_FakeYoloe] = []

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.classes: list[str] | None = None
        self.text_pe_calls: list[list[str]] = []
        self.predict_kwargs: dict[str, Any] | None = None
        self.boxes: Any = _boxes()
        _FakeYoloe.instances.append(self)

    def get_text_pe(self, names: list[str]) -> str:
        self.text_pe_calls.append(list(names))
        return f"text_pe[{len(names)}]"

    def set_classes(self, classes: list[str], embeddings: Any) -> None:
        self.classes = classes
        self.embeddings = embeddings

    def predict(self, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        source = kwargs["source"]
        if isinstance(source, list):  # batched
            return [SimpleNamespace(boxes=self.boxes) for _ in source]
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.fixture(autouse=True)
def _patch_yoloe(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYoloe.instances = []
    monkeypatch.setattr(ultralytics, "YOLOE", _FakeYoloe, raising=False)


def _loaded(tmp_path: Path, **cfg_kw: Any) -> tuple[YoloeBackend, _FakeYoloe]:
    path, sha = _weights(tmp_path)
    backend = YoloeBackend()
    backend.load(_config(path, sha, **cfg_kw), "cpu", _IDS)
    return backend, _FakeYoloe.instances[-1]


class TestLoad:
    def test_binds_text_prompt_embeddings(self, tmp_path: Path) -> None:
        _backend, model = _loaded(tmp_path)
        # set_classes gets the flattened prompt strings + their embeddings.
        assert model.classes == ["phone charger", "power adapter", "cable"]
        assert model.text_pe_calls == [["phone charger", "power adapter", "cable"]]

    def test_empty_prompts_raise(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        with pytest.raises(AnnotationError, match="every prompt list is empty"):
            YoloeBackend().load(_config(path, sha, prompts={"charger": []}), "cpu", _IDS)

    def test_unpinned_weights_raise(self, tmp_path: Path) -> None:
        path, _ = _weights(tmp_path)
        with pytest.raises(AnnotationError, match="not pinned"):
            YoloeBackend().load(_config(path, ""), "cpu", _IDS)

    def test_unknown_prompt_class_raises(self, tmp_path: Path) -> None:
        path, sha = _weights(tmp_path)
        with pytest.raises(AnnotationError, match="not in the taxonomy"):
            YoloeBackend().load(_config(path, sha, prompts={"nope": ["x"]}), "cpu", _IDS)


class TestAnnotate:
    def test_maps_and_filters_to_targets(self, tmp_path: Path) -> None:
        backend, _model = _loaded(tmp_path)
        dets = backend.annotate(tmp_path / "img.jpg", (10,))  # charger only
        assert [d.class_id for d in dets] == [10]
        assert dets[0].origin == "yoloe"

    def test_predict_receives_nms_config(self, tmp_path: Path) -> None:
        backend, model = _loaded(tmp_path, iou=0.45, agnostic_nms=True)
        backend.annotate(tmp_path / "img.jpg", (10, 11))
        assert model.predict_kwargs["iou"] == 0.45
        assert model.predict_kwargs["agnostic_nms"] is True
        assert model.predict_kwargs["imgsz"] == 640
        assert model.predict_kwargs["conf"] == 0.05

    def test_out_of_range_prompt_index_raises(self, tmp_path: Path) -> None:
        backend, model = _loaded(tmp_path)
        model.boxes = SimpleNamespace(
            cls=torch.tensor([99]),
            conf=torch.tensor([0.9]),
            xywhn=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        )
        with pytest.raises(AnnotationError, match="outside the configured"):
            backend.annotate(tmp_path / "img.jpg", (10,))

    def test_requires_load(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="load"):
            YoloeBackend().annotate(tmp_path / "img.jpg", (10,))


class TestAnnotateBatch:
    def test_batch_matches_per_image(self, tmp_path: Path) -> None:
        backend, _model = _loaded(tmp_path)
        paths = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]
        batched = backend.annotate_batch(paths, (10, 11))
        per_image = [backend.annotate(p, (10, 11)) for p in paths]
        assert batched == per_image
        assert len(batched) == 3

    def test_batch_uses_single_predict_call(self, tmp_path: Path) -> None:
        backend, model = _loaded(tmp_path)
        paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
        backend.annotate_batch(paths, (10,))
        assert isinstance(model.predict_kwargs["source"], list)
        assert len(model.predict_kwargs["source"]) == 2

    def test_empty_batch_returns_empty(self, tmp_path: Path) -> None:
        backend, _model = _loaded(tmp_path)
        assert backend.annotate_batch([], (10,)) == []

    def test_requires_load(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="load"):
            YoloeBackend().annotate_batch([tmp_path / "a.jpg"], (10,))


class TestFingerprint:
    def test_records_backend_and_nms(self, tmp_path: Path) -> None:
        backend, _model = _loaded(tmp_path)
        fp = backend.fingerprint()
        assert fp.backend == "yoloe"
        base = fp.prompt_fingerprint
        # NMS change must alter the fingerprint (stale-run invalidation).
        backend2, _ = _loaded(tmp_path, iou=0.45)
        assert backend2.fingerprint().prompt_fingerprint != base

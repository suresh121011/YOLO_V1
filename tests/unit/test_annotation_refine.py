"""Unit tests for the SAM refinement pass (mocked SAM; real tiny image)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from src.dataset.annotation.base import AnnotationError, Detection  # noqa: E402
from src.dataset.annotation.refine import RefinementPass, _mask_bounds  # noqa: E402

pytestmark = pytest.mark.unit

_IMG_SIZE = 64  # square test image side


def _weights(tmp_path: Path) -> tuple[Path, str]:
    content = b"fake-sam"
    path = tmp_path / "mobile_sam.pt"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _image(tmp_path: Path) -> Path:
    import cv2

    path = tmp_path / "img.jpg"
    cv2.imwrite(str(path), np.full((_IMG_SIZE, _IMG_SIZE, 3), 128, dtype=np.uint8))
    return path


def _detection(bbox: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)) -> Detection:
    return Detection(class_id=10, conf=0.7, bbox_xywhn=bbox, origin="yolo_world")


class _FakeSam:
    """Returns a configurable mask stack for the prompted boxes."""

    next_masks: Any = None
    raise_on_predict: bool = False

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def predict(self, **kwargs: Any) -> list[Any]:
        if _FakeSam.raise_on_predict:
            raise RuntimeError("sam exploded")
        masks = _FakeSam.next_masks
        return [SimpleNamespace(masks=None if masks is None else SimpleNamespace(data=masks))]


@pytest.fixture(autouse=True)
def _patch_sam(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSam.next_masks = None
    _FakeSam.raise_on_predict = False
    monkeypatch.setattr(ultralytics, "SAM", _FakeSam)


def _pass(tmp_path: Path) -> RefinementPass:
    weights, sha = _weights(tmp_path)
    return RefinementPass(weights, sha, device="cpu")


def _mask_with_box(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    mask = torch.zeros(_IMG_SIZE, _IMG_SIZE, dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return mask


class TestRefinementPass:
    def test_pin_enforced(self, tmp_path: Path) -> None:
        weights, _ = _weights(tmp_path)
        with pytest.raises(AnnotationError, match="digest mismatch"):
            RefinementPass(weights, "0" * 64, device="cpu")

    def test_mask_tightens_box(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.next_masks = torch.stack([_mask_with_box(16, 16, 48, 48)])
        [refined] = refiner.refine(_image(tmp_path), [_detection()])
        assert refined.refined is True
        x, y, w, h = refined.bbox_xywhn
        assert (x, y) == (0.5, 0.5)
        assert w == pytest.approx(32 / _IMG_SIZE)
        assert h == pytest.approx(32 / _IMG_SIZE)
        # class/conf/origin preserved
        assert refined.class_id == 10 and refined.conf == 0.7 and refined.origin == "yolo_world"

    def test_empty_mask_keeps_original(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.next_masks = torch.stack([torch.zeros(_IMG_SIZE, _IMG_SIZE, dtype=torch.bool)])
        original = _detection()
        [kept] = refiner.refine(_image(tmp_path), [original])
        assert kept == original
        assert kept.refined is False

    def test_degenerate_tiny_mask_keeps_original(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.next_masks = torch.stack([_mask_with_box(30, 30, 32, 32)])  # 2px < min extent
        original = _detection()
        [kept] = refiner.refine(_image(tmp_path), [original])
        assert kept == original

    def test_mask_count_mismatch_keeps_all_originals(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.next_masks = torch.stack([_mask_with_box(0, 0, 32, 32)])  # 1 mask, 2 dets
        originals = [_detection(), _detection((0.2, 0.2, 0.1, 0.1))]
        assert refiner.refine(_image(tmp_path), originals) == originals

    def test_sam_failure_keeps_all_originals(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.raise_on_predict = True
        originals = [_detection()]
        assert refiner.refine(_image(tmp_path), originals) == originals

    def test_never_creates_or_deletes(self, tmp_path: Path) -> None:
        refiner = _pass(tmp_path)
        _FakeSam.next_masks = torch.stack(
            [_mask_with_box(0, 0, 32, 32), _mask_with_box(32, 32, 64, 64)]
        )
        originals = [_detection(), _detection((0.2, 0.2, 0.1, 0.1))]
        refined = refiner.refine(_image(tmp_path), originals)
        assert len(refined) == len(originals)
        assert [d.class_id for d in refined] == [d.class_id for d in originals]

    def test_empty_input(self, tmp_path: Path) -> None:
        assert _pass(tmp_path).refine(_image(tmp_path), []) == []


class TestMaskBounds:
    def test_bounds_of_rectangular_mask(self) -> None:
        assert _mask_bounds(_mask_with_box(4, 8, 10, 20)) == (4.0, 8.0, 10.0, 20.0)

    def test_empty_mask_is_none(self) -> None:
        assert _mask_bounds(torch.zeros(8, 8, dtype=torch.bool)) is None

"""Unit tests for sliced (SAHI-style) inference (P6): planning, remap, NMS, orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.base import Detection
from src.dataset.annotation.sliced import (
    SliceConfig,
    SliceRect,
    annotate_sliced,
    nms_per_class,
    plan_slices,
    remap_box_to_full,
)

pytestmark = pytest.mark.unit


class TestPlanSlices:
    def test_subtile_image_is_single_full_frame(self) -> None:
        slices = plan_slices(400, 300, 640, 640, 0.2)
        assert slices == [SliceRect(0, 0, 400, 300)]

    def test_covers_edges_with_overlap(self) -> None:
        slices = plan_slices(1000, 800, 640, 640, 0.2)
        # xs = [0, 360] (step 512, plus flush far edge), ys = [0, 160] → 4 tiles.
        assert len(slices) == 4
        assert SliceRect(0, 0, 640, 640) in slices
        assert SliceRect(360, 160, 1000, 800) in slices
        # far edges reached
        assert max(s.x1 for s in slices) == 1000
        assert max(s.y1 for s in slices) == 800

    def test_every_pixel_covered(self) -> None:
        w, h = 1500, 900
        slices = plan_slices(w, h, 640, 640, 0.25)
        assert min(s.x0 for s in slices) == 0
        assert min(s.y0 for s in slices) == 0
        assert max(s.x1 for s in slices) == w
        assert max(s.y1 for s in slices) == h

    def test_invalid_overlap_raises(self) -> None:
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match="overlap_ratio"):
                plan_slices(1000, 800, 640, 640, bad)


class TestRemapBoxToFull:
    def test_tile_center_maps_to_tile_location(self) -> None:
        tile = SliceRect(0, 0, 640, 640)
        # centered box in a 640-tile of a 1000x800 image
        cx, cy, w, h = remap_box_to_full((0.5, 0.5, 0.2, 0.2), tile, 1000, 800)
        assert cx == pytest.approx(0.32)  # 320/1000
        assert cy == pytest.approx(0.40)  # 320/800
        assert w == pytest.approx(0.128)  # 128/1000
        assert h == pytest.approx(0.16)  # 128/800

    def test_offset_tile_adds_origin(self) -> None:
        tile = SliceRect(360, 160, 1000, 800)  # 640x640
        cx, cy, _, _ = remap_box_to_full((0.5, 0.5, 0.1, 0.1), tile, 1000, 800)
        assert cx == pytest.approx((360 + 320) / 1000)  # 0.68
        assert cy == pytest.approx((160 + 320) / 800)  # 0.60


class TestNmsPerClass:
    def _det(self, cid: int, conf: float, box: tuple[float, float, float, float]) -> Detection:
        return Detection(class_id=cid, conf=conf, bbox_xywhn=box, origin="t")

    def test_overlapping_same_class_suppressed_keeps_higher_conf(self) -> None:
        a = self._det(0, 0.9, (0.5, 0.5, 0.2, 0.2))
        b = self._det(0, 0.6, (0.51, 0.5, 0.2, 0.2))  # heavy overlap, lower conf
        kept = nms_per_class([b, a], iou_threshold=0.5)
        assert kept == [a]

    def test_different_classes_not_suppressed(self) -> None:
        a = self._det(0, 0.9, (0.5, 0.5, 0.2, 0.2))
        b = self._det(1, 0.6, (0.5, 0.5, 0.2, 0.2))  # same box, different class
        kept = nms_per_class([a, b], iou_threshold=0.5)
        assert len(kept) == 2

    def test_distinct_boxes_all_kept(self) -> None:
        a = self._det(0, 0.9, (0.2, 0.2, 0.1, 0.1))
        b = self._det(0, 0.8, (0.8, 0.8, 0.1, 0.1))
        kept = nms_per_class([a, b], iou_threshold=0.5)
        assert len(kept) == 2


class _TileFake:
    """Mock backend: returns a center detection ONLY for the named tile index.

    annotate_sliced crops tiles to ``<stem>__tile<i>.png``; this lets a test
    assert the remapped coordinate for exactly one known tile.
    """

    def __init__(self, active_tile: int, cid: int = 10) -> None:
        self.active = active_tile
        self.cid = cid

    def annotate(self, image_path: Path, target_class_ids: tuple[int, ...]) -> list[Detection]:
        if f"__tile{self.active}" in image_path.name:
            return [Detection(self.cid, 0.9, (0.5, 0.5, 0.2, 0.2), origin="fake")]
        return []

    def annotate_batch(
        self, image_paths: list[Path], target_class_ids: tuple[int, ...]
    ) -> list[list[Detection]]:
        return [self.annotate(p, target_class_ids) for p in image_paths]


def _make_image(path: Path, size: tuple[int, int]) -> Path:
    from PIL import Image

    Image.new("RGB", size, (120, 120, 120)).save(path)
    return path


class TestAnnotateSliced:
    def test_single_tile_detection_remapped(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path / "img.png", (1000, 800))
        backend = _TileFake(active_tile=0)
        cfg = SliceConfig(enabled=True, overlap_ratio=0.2, include_full_frame=False)
        dets = annotate_sliced(backend, img, (10,), cfg)
        assert len(dets) == 1
        cx, cy, w, h = dets[0].bbox_xywhn
        assert cx == pytest.approx(0.32)  # tile0 center in full frame
        assert cy == pytest.approx(0.40)
        assert dets[0].class_id == 10

    def test_subtile_image_uses_single_full_pass(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path / "small.png", (400, 300))

        calls: dict[str, int] = {"annotate": 0, "batch": 0}

        class _Counter(_TileFake):
            def annotate(self, image_path: Path, t: tuple[int, ...]) -> list[Detection]:
                calls["annotate"] += 1
                return [Detection(10, 0.9, (0.5, 0.5, 0.2, 0.2), origin="fake")]

            def annotate_batch(self, ps: list[Path], t: tuple[int, ...]) -> list[list[Detection]]:
                calls["batch"] += 1
                return [self.annotate(p, t) for p in ps]

        cfg = SliceConfig(enabled=True)
        dets = annotate_sliced(_Counter(0), img, (10,), cfg)
        assert len(dets) == 1
        assert calls["annotate"] == 1  # one plain full-frame pass
        assert calls["batch"] == 0  # never sliced

    def test_full_frame_pass_included(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path / "img.png", (1000, 800))

        class _AllFake:
            """Returns a center box for EVERY call (each tile + the full frame)."""

            def annotate(self, p: Path, t: tuple[int, ...]) -> list[Detection]:
                return [Detection(10, 0.9, (0.5, 0.5, 0.2, 0.2), origin="fake")]

            def annotate_batch(self, ps: list[Path], t: tuple[int, ...]) -> list[list[Detection]]:
                return [self.annotate(p, t) for p in ps]

        cfg = SliceConfig(enabled=True, include_full_frame=True, nms_iou=0.5)
        dets = annotate_sliced(_AllFake(), img, (10,), cfg)
        centers = [(round(d.bbox_xywhn[0], 2), round(d.bbox_xywhn[1], 2)) for d in dets]
        # 4 remapped tile centers + the full-frame center (all distinct → kept).
        assert (0.5, 0.5) in centers  # full-frame pass
        assert (0.32, 0.4) in centers  # tile0
        assert len(dets) == 5

    def test_full_frame_disabled_skips_whole_image_pass(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path / "img.png", (1000, 800))

        class _AllFake:
            def annotate(self, p: Path, t: tuple[int, ...]) -> list[Detection]:
                return [Detection(10, 0.9, (0.5, 0.5, 0.2, 0.2), origin="fake")]

            def annotate_batch(self, ps: list[Path], t: tuple[int, ...]) -> list[list[Detection]]:
                return [self.annotate(p, t) for p in ps]

        cfg = SliceConfig(enabled=True, include_full_frame=False, nms_iou=0.5)
        dets = annotate_sliced(_AllFake(), img, (10,), cfg)
        centers = [(round(d.bbox_xywhn[0], 2), round(d.bbox_xywhn[1], 2)) for d in dets]
        assert (0.5, 0.5) not in centers  # no full-frame pass
        assert len(dets) == 4  # 4 tiles only

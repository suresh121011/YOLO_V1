"""
src.dataset.annotation.sliced — Sliced (SAHI-style) Inference
=============================================================

P6 (annotation V2 plan). Tiny objects (charger, wire, monitor — tens of
instances each) are missed by a single 640px forward pass because they occupy a
handful of pixels once the frame is downscaled. This module runs the annotator
over overlapping tiles of the full image ("Slicing-Aided Hyper Inference"),
remaps each tile's detections back to full-frame coordinates, and merges them
with a cross-tile NMS.

Dependency-free by design: the geometry (slice planning, tile→full remap,
cross-class NMS) is pure and fully unit-tested, and the orchestrator reuses the
existing :meth:`AutoAnnotator.annotate_batch` contract — no new heavy package,
consistent with the codebase's lazy-import backends (ADR-P5-11). The ``sahi``
package could be swapped in behind :func:`annotate_sliced` without changing
callers.

Selective by intent: slicing is slow, so callers should gate it to the small/
priority classes (``targeting.priority_classes``), not the whole taxonomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.dataset.annotation.base import AutoAnnotator, Detection
from src.dataset.annotation.coverage import iou_xywhn

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SliceConfig:
    """Sliced-inference settings (read from configs/annotation.yaml)."""

    enabled: bool = False
    slice_width: int = 640
    slice_height: int = 640
    overlap_ratio: float = 0.2  # fraction of tile that overlaps its neighbour
    nms_iou: float = 0.5  # cross-tile duplicate suppression threshold
    include_full_frame: bool = True  # also do one whole-image pass (large objects)


@dataclass(frozen=True)
class SliceRect:
    """A tile's pixel box in the full image: [x0, x1) × [y0, y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _axis_starts(length: int, tile: int, overlap_ratio: float) -> list[int]:
    """Tile start offsets covering ``length`` with the given tile size/overlap.

    Always covers the far edge (a final clamped tile is added when the stride
    doesn't land flush), and collapses to ``[0]`` when the image is no larger
    than the tile.
    """
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap_ratio))))
    starts = list(range(0, length - tile + 1, step))
    if not starts or starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def plan_slices(
    width: int, height: int, slice_w: int, slice_h: int, overlap_ratio: float
) -> list[SliceRect]:
    """Plan overlapping tiles covering a ``width``×``height`` image.

    Returns row-major tiles; every pixel is covered by ≥1 tile and edges are
    always included. A sub-tile-sized image yields a single full-frame rect.
    """
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio {overlap_ratio} must be in [0, 1)")
    xs = _axis_starts(width, slice_w, overlap_ratio)
    ys = _axis_starts(height, slice_h, overlap_ratio)
    return [
        SliceRect(x, y, min(x + slice_w, width), min(y + slice_h, height)) for y in ys for x in xs
    ]


def remap_box_to_full(box: Box, tile: SliceRect, full_w: int, full_h: int) -> Box:
    """Map a tile-normalized xywhn box to full-image-normalized xywhn.

    The box is normalized within the tile; convert to pixels in the tile, add
    the tile's offset, then renormalize by the full image size.
    """
    cx, cy, w, h = box
    cx_px = cx * tile.width + tile.x0
    cy_px = cy * tile.height + tile.y0
    w_px = w * tile.width
    h_px = h * tile.height
    return (cx_px / full_w, cy_px / full_h, w_px / full_w, h_px / full_h)


def nms_per_class(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy per-class NMS: keep highest-conf boxes, drop overlapping twins.

    Overlapping detections of the SAME class from adjacent tiles are the
    duplicates to remove; boxes of different classes never suppress each other.
    Ordering within the kept set follows descending confidence (stable).
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.conf, reverse=True):
        if any(
            other.class_id == det.class_id
            and iou_xywhn(det.bbox_xywhn, other.bbox_xywhn) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(det)
    return kept


def annotate_sliced(
    backend: AutoAnnotator,
    image_path: Path,
    target_class_ids: tuple[int, ...],
    config: SliceConfig,
    *,
    _tmp_dir: Path | None = None,
) -> list[Detection]:
    """Run ``backend`` over overlapping tiles and merge to full-frame detections.

    Crops each tile to a temp image, batches them through
    :meth:`AutoAnnotator.annotate_batch`, remaps every detection to full-image
    coordinates, optionally adds a whole-image pass, then de-duplicates across
    tiles with :func:`nms_per_class`. Falls back to a single full-frame
    ``annotate`` when the image is no larger than one tile.

    ``_tmp_dir`` is an injection point for tests; production uses a
    :class:`tempfile.TemporaryDirectory`.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        full_w, full_h = img.size
        rgb = img.convert("RGB")

        slices = plan_slices(
            full_w, full_h, config.slice_width, config.slice_height, config.overlap_ratio
        )
        # Sub-tile image: nothing to slice — one plain pass.
        if len(slices) == 1 and slices[0] == SliceRect(0, 0, full_w, full_h):
            return backend.annotate(image_path, target_class_ids)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = _tmp_dir or Path(td)
            tmp.mkdir(parents=True, exist_ok=True)
            tile_paths: list[Path] = []
            for i, rect in enumerate(slices):
                crop = rgb.crop((rect.x0, rect.y0, rect.x1, rect.y1))
                p = tmp / f"{image_path.stem}__tile{i}.png"
                crop.save(p)
                tile_paths.append(p)

            tile_dets = backend.annotate_batch(tile_paths, target_class_ids)

    merged: list[Detection] = []
    for rect, dets in zip(slices, tile_dets, strict=True):
        for det in dets:
            merged.append(
                Detection(
                    class_id=det.class_id,
                    conf=det.conf,
                    bbox_xywhn=remap_box_to_full(det.bbox_xywhn, rect, full_w, full_h),
                    refined=det.refined,
                    origin=det.origin,
                )
            )

    if config.include_full_frame:
        merged.extend(backend.annotate(image_path, target_class_ids))

    return nms_per_class(merged, config.nms_iou)

"""
src.dataset.annotation.refine — SAM Box-Refinement Post-Pass
============================================================

Optional refinement (ADR-P5-02): prompts an ultralytics-native SAM variant
(MobileSAM default) with each candidate box and tightens the box to the
resulting mask's extent. STRICTLY box-adjusting — it never creates, deletes,
or reclassifies detections, and any degenerate/failed mask falls back to the
original box, so the refinement pass can only improve geometry, never lose a
candidate a human should have seen.

Weight pinning follows the backend contract (verify_weights — hard-fail on
missing/mismatched/unpinned digests).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.dataset.annotation.backends.yolo_world import verify_weights
from src.dataset.annotation.base import Detection

logger = logging.getLogger(__name__)

#: Minimum mask extent (pixels) below which refinement falls back to the
#: original box — a 1-2 px mask is a SAM failure mode, not a real object.
MIN_MASK_EXTENT_PX = 4


class RefinementPass:
    """Reusable SAM refinement across many images (model loaded once)."""

    def __init__(self, weights: Path, weights_sha256: str, device: str) -> None:
        """Verify the pin and load the SAM model.

        Args:
            weights:        SAM weight file (e.g. models/annotators/mobile_sam.pt).
            weights_sha256: Pinned digest from configs/annotation.yaml.
            device:         Torch device string.

        Raises:
            AnnotationError: On pin problems (verify_weights contract).
        """
        self.weights_sha256 = verify_weights(weights, weights_sha256, "refinement")
        from ultralytics import SAM  # heavy import stays inside (house pattern)

        self._model: Any = SAM(str(weights))
        self._device = device
        logger.info(f"Refinement loaded: {weights.name} (device={device})")

    def refine(self, image_path: Path, detections: list[Detection]) -> list[Detection]:
        """Tighten candidate boxes via mask extents (same order, same count).

        Args:
            image_path: Image the detections belong to.
            detections: Candidate detections (normalized xywhn geometry).

        Returns:
            Detections with ``refined=True`` where a usable mask tightened the
            box; originals (``refined=False``) wherever SAM failed or the mask
            was degenerate.
        """
        if not detections:
            return []

        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Refinement: cannot read {image_path} — keeping original boxes")
            return list(detections)
        img_h, img_w = image.shape[:2]

        boxes_xyxy = []
        for det in detections:
            x, y, w, h = det.bbox_xywhn
            boxes_xyxy.append(
                [
                    max(0.0, (x - w / 2) * img_w),
                    max(0.0, (y - h / 2) * img_h),
                    min(float(img_w), (x + w / 2) * img_w),
                    min(float(img_h), (y + h / 2) * img_h),
                ]
            )

        try:
            results = self._model.predict(
                source=str(image_path),
                bboxes=boxes_xyxy,
                device=self._device,
                verbose=False,
            )
            masks = results[0].masks
        except Exception as exc:  # SAM failure must never drop candidates
            logger.warning(f"Refinement failed for {image_path.name} ({exc}) — keeping originals")
            return list(detections)
        if masks is None or len(masks.data) != len(detections):
            logger.warning(
                f"Refinement: mask count mismatch for {image_path.name} — keeping originals"
            )
            return list(detections)

        refined: list[Detection] = []
        for det, mask in zip(detections, masks.data, strict=True):  # counts checked above
            bounds = _mask_bounds(mask)
            if bounds is None:
                refined.append(det)
                continue
            x0, y0, x1, y1 = bounds
            if (x1 - x0) < MIN_MASK_EXTENT_PX or (y1 - y0) < MIN_MASK_EXTENT_PX:
                refined.append(det)
                continue
            mask_h, mask_w = mask.shape[-2], mask.shape[-1]
            refined.append(
                Detection(
                    class_id=det.class_id,
                    conf=det.conf,
                    bbox_xywhn=(
                        ((x0 + x1) / 2) / mask_w,
                        ((y0 + y1) / 2) / mask_h,
                        (x1 - x0) / mask_w,
                        (y1 - y0) / mask_h,
                    ),
                    refined=True,
                    origin=det.origin,
                )
            )
        return refined


def _mask_bounds(mask: Any) -> tuple[float, float, float, float] | None:
    """Tight (x0, y0, x1, y1) pixel bounds of a boolean mask tensor, or None."""
    nonzero = mask.nonzero()
    if nonzero.numel() == 0:
        return None
    ys = nonzero[:, -2]
    xs = nonzero[:, -1]
    return (
        float(xs.min().item()),
        float(ys.min().item()),
        float(xs.max().item()) + 1.0,
        float(ys.max().item()) + 1.0,
    )

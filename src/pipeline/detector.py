"""
YOLO11n Object Detector
========================
Wraps Ultralytics YOLO11n with per-class confidence thresholds
and optional model hash verification.

Supports: .pt (PyTorch), .onnx (ONNX Runtime), .tflite (TFLite / Android).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import BoundingBox, Detection

logger = logging.getLogger(__name__)


# Safety-critical classes get lower confidence thresholds to prefer recall.
# Values override the global default confidence threshold from configs.
DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "knife": 0.20,
    "stove": 0.25,
    "gas_cylinder": 0.22,
    "wire": 0.22,
    "wet_floor": 0.20,
    "medicine_strip": 0.25,
    "medicine_bottle": 0.25,
    "passport": 0.30,
    "person": 0.30,
    "face": 0.35,
}


class YOLODetector:
    """YOLO11n object detector with per-class confidence threshold support.

    Responsibilities:
    - Load model (PT / ONNX / TFLite) and verify integrity
    - Run inference on BGR frames
    - Apply global and per-class confidence thresholds
    - Return typed list[Detection]
    """

    DEFAULT_CONF = 0.25
    DEFAULT_IOU = 0.45

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONF,
        class_thresholds: dict[str, float] | None = None,
        expected_hash: str | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path: Path to YOLO model weights (.pt / .onnx / .tflite).
            conf_threshold: Global minimum confidence for detections.
            class_thresholds: Per-class overrides (safety classes use lower values).
            expected_hash: Optional SHA-256 hash to verify model integrity on load.
            device: Inference device ('cpu', 'cuda', 'mps').
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.class_thresholds = {**DEFAULT_CLASS_THRESHOLDS, **(class_thresholds or {})}
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        if expected_hash:
            self._verify_hash(expected_hash)

        self.model: Any = self._load_model()
        logger.info(f"YOLODetector loaded: {self.model_path.name} on {device}")

    def _verify_hash(self, expected: str) -> None:
        """Verify model file integrity against expected SHA-256 hash."""
        sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if sha256 != expected:
            raise ValueError(
                f"Model integrity check failed.\n"
                f"  Expected: {expected}\n"
                f"  Got:      {sha256}"
            )
        logger.info("Model hash verified OK")

    def _load_model(self) -> Any:
        """Load model using appropriate backend for the file format."""
        suffix = self.model_path.suffix.lower()
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            logger.info(f"Model loaded via Ultralytics (format: {suffix})")
            return model
        except ImportError as e:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            ) from e

    def warmup(self, n: int = 3) -> None:
        """Run inference on dummy frames to initialize GPU/NPU kernels.

        Call once after loading, before real-time inference begins.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, verbose=False)
        logger.info(f"YOLO warmup complete ({n} frames)")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame: BGR image array from OpenCV.
            frame_id: Current frame index for traceability.

        Returns:
            List of Detection objects passing confidence thresholds.
        """
        timestamp_ms = time.time() * 1000

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.DEFAULT_IOU,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self.model.names[class_id]
                conf = float(box.conf.item())

                # Apply per-class threshold (may be stricter or more lenient than default)
                min_conf = self.class_thresholds.get(class_name, self.conf_threshold)
                if conf < min_conf:
                    continue

                # Normalized [x1, y1, x2, y2] → YOLO center format
                xyxy = box.xyxyn[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=BoundingBox(cx=cx, cy=cy, w=w, h=h),
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                    )
                )

        return detections

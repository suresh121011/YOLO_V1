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


class TaxonomyMismatchError(ValueError):
    """The loaded weights do not carry the class list the system reasons over.

    Raised at construction, never at inference. Every downstream layer keys off
    ``Detection.class_name``: the scenario engine's inverted index, its
    per-class confidence floors, the capability map, and the privacy suppression
    of ``passport``. If the weights disagree with ``configs/data.yaml``, all of
    those silently address the wrong class and the pipeline looks healthy while
    reasoning about the wrong world.
    """


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
        disabled_classes: set[str] | frozenset[str] | None = None,
        expected_classes: dict[int, str] | None = None,
    ) -> None:
        """
        Args:
            model_path: Path to YOLO model weights (.pt / .onnx / .tflite).
            conf_threshold: Global minimum confidence for detections.
            class_thresholds: Per-class overrides (safety classes use lower values).
            expected_hash: Optional SHA-256 hash to verify model integrity on load.
            device: Inference device ('cpu', 'cuda', 'mps').
            disabled_classes: Class names to suppress entirely. Suppression happens
                here, so a disabled class never becomes a Detection and therefore
                never reaches a rule, a log, or an alert. This is what makes
                ``passport: false  # privacy`` in configs/feature_flags.yaml a real
                guarantee rather than a comment.
            expected_classes: The ``id -> name`` taxonomy from configs/data.yaml.
                When given, the loaded weights must carry exactly it. Optional so
                that tests and experiments can load arbitrary weights; the
                composition root always supplies it.

        Raises:
            FileNotFoundError:      If the weights file does not exist.
            TaxonomyMismatchError:  If the weights disagree with expected_classes.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.class_thresholds = {**DEFAULT_CLASS_THRESHOLDS, **(class_thresholds or {})}
        self.device = device
        self.disabled_classes = frozenset(disabled_classes or ())

        # Ultralytics filters by `conf` BEFORE our per-class pass runs, so
        # handing it the global threshold made every per-class value that is
        # LOWER than the global unreachable — silently defeating the
        # recall-preferring safety thresholds documented in
        # configs/class_thresholds.yaml. Predict at the loosest threshold any
        # class asks for; the per-class filter in detect() then applies the
        # real, stricter-or-equal cut.
        self._predict_conf = min([conf_threshold, *self.class_thresholds.values()])

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        if expected_hash:
            self._verify_hash(expected_hash)

        self.model: Any = self._load_model()
        self._verify_taxonomy(expected_classes)
        logger.info(f"YOLODetector loaded: {self.model_path.name} on {device}")

    # ─────────────────────────────────────────
    # Taxonomy verification
    # ─────────────────────────────────────────

    @staticmethod
    def model_class_names(model: Any) -> dict[int, str]:
        """Normalise Ultralytics' ``names`` (dict or list) to ``id -> name``."""
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def _verify_taxonomy(self, expected: dict[int, str] | None) -> None:
        """Refuse weights whose class list is not the one the system reasons over.

        This is the check that makes "the model lands and nothing needs changing"
        falsifiable. Class **ids** matter as much as names: the whole taxonomy is
        addressed by id in ``configs/data.yaml`` and the R24 decision deliberately
        holds id 20 reserved for ``wet_floor`` without renumbering, so a model
        that renumbers while keeping the same name set is exactly the failure this
        exists to catch.
        """
        actual = self.model_class_names(self.model)
        self.class_names = actual

        if not expected:
            logger.warning(
                "No expected taxonomy supplied -- the weights' class list is UNVERIFIED. "
                "Build through src.app.factory.build_pipeline(), which passes "
                "configs/data.yaml, or run scripts/qa/model_landing_check.py."
            )
            return

        if actual == expected:
            logger.info(f"Taxonomy verified: {len(actual)} classes match configs/data.yaml")
            return

        missing = sorted(i for i in expected if i not in actual)
        extra = sorted(i for i in actual if i not in expected)
        renamed = sorted(i for i in expected.keys() & actual.keys() if expected[i] != actual[i])

        lines = [
            f"{self.model_path.name} was trained on a different taxonomy than "
            f"configs/data.yaml declares ({len(actual)} classes vs {len(expected)}).",
        ]
        if missing:
            lines.append(
                "  missing from the model: " + ", ".join(f"{i}={expected[i]}" for i in missing[:10])
            )
        if extra:
            lines.append(
                "  present only in the model: " + ", ".join(f"{i}={actual[i]}" for i in extra[:10])
            )
        if renamed:
            collisions = ", ".join(
                f"{i}: expected {expected[i]!r}, model has {actual[i]!r}" for i in renamed[:10]
            )
            lines.append(f"  id collisions: {collisions}")
        lines.append(
            "  Every scenario addresses classes by name and the detector suppresses "
            "'passport' by name, so this must not be papered over at inference time."
        )
        raise TaxonomyMismatchError("\n".join(lines))

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
            conf=self._predict_conf,
            iou=self.DEFAULT_IOU,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self.model.names[class_id]
                conf = float(box.conf.item())

                # Privacy/noise suppression — drop before anything else observes it.
                if class_name in self.disabled_classes:
                    continue

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

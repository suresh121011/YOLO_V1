"""
src.pipeline — Core Pipeline Data Contracts
============================================

This module defines the canonical data types shared across ALL pipeline components.
These interfaces are LOCKED in Stage 1. All future stages must conform to them.

Changing these dataclasses is a breaking change — update all downstream modules.

Data flow:
    Camera frame
        → Detection (output of YOLODetector)
        → MemoryEntry (maintained by EventMemory)
        → SceneContext (output of SmolVLM2Analyzer, optional)
        → Alert (output of RuleEngine)
        → PipelineMetrics (performance telemetry)
        → FrameResult (complete per-frame summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ─── Enumerations ─────────────────────────────────────────────────────────────


class Severity(IntEnum):
    """Alert severity levels, ordered from lowest to highest urgency.

    Used by:
        - RuleEngine to set alert priority
        - AlertQueue to sort pending alerts (CRITICAL spoken first)
        - StructuredLogger to tag log entries
        - TTS engine for priority queue ordering
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ─── Geometric Types ──────────────────────────────────────────────────────────


@dataclass
class BoundingBox:
    """Normalized bounding box in YOLO format.

    All coordinates are in [0, 1] range, normalized to frame dimensions.
    Origin is top-left corner.

    Args:
        cx: Center x coordinate [0, 1]
        cy: Center y coordinate [0, 1]
        w:  Width as fraction of frame width [0, 1]
        h:  Height as fraction of frame height [0, 1]
    """

    cx: float  # center x in [0, 1]
    cy: float  # center y in [0, 1]
    w: float  # width in [0, 1]
    h: float  # height in [0, 1]

    def area(self) -> float:
        """Return bounding box area as fraction of frame area."""
        return self.w * self.h

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return x1, y1, x2, y2

    def iou(self, other: BoundingBox) -> float:
        """Compute intersection-over-union with another bounding box."""
        ax1, ay1, ax2, ay2 = self.to_xyxy()
        bx1, by1, bx2, by2 = other.to_xyxy()

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union_area = self.area() + other.area() - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# ─── Detection ────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """Single object detection output from YOLO11n.

    Produced by: YOLODetector.detect()
    Consumed by: EventMemory.update(), ConfidenceFusion.fuse(), RuleEngine.evaluate()

    Args:
        class_id:     Integer class index matching configs/data.yaml names
        class_name:   String class name (e.g., "knife", "stove", "person")
        confidence:   Detection confidence score [0.0, 1.0]
        bbox:         Normalized bounding box in YOLO cx/cy/w/h format
        frame_id:     Sequential frame counter for this pipeline session
        timestamp_ms: Unix timestamp in milliseconds when detection was made
    """

    class_id: int
    class_name: str
    confidence: float  # [0.0, 1.0]
    bbox: BoundingBox
    frame_id: int
    timestamp_ms: float


# ─── Scene Context (VLM Output) ───────────────────────────────────────────────


@dataclass
class SceneContext:
    """Scene understanding output from SmolVLM2.

    Optional — only available when smolvlm_analysis is enabled in feature flags
    and hardware supports the VLM.

    Produced by: SmolVLM2Analyzer.analyze()
    Consumed by: ConfidenceFusion.fuse(), RuleEngine.evaluate()

    Safety constraint: VLM context is ADVISORY ONLY. The Rule Engine uses YOLO
    detections as ground truth. VLM can upgrade alert severity but cannot cancel
    a rule-engine-triggered alert.

    Args:
        activity:         Detected activity description (e.g., "cooking", "sleeping")
        risks:            List of detected risks with type and severity
        recommendations:  List of suggested safety actions
        raw_response:     Raw VLM JSON response (for debugging)
        inference_time_ms: VLM inference latency in milliseconds
        frame_id:         Frame this context was generated for
    """

    activity: str
    risks: list[dict]  # [{"risk_type": str, "severity": str, "description": str}]
    recommendations: list[str]
    raw_response: str
    inference_time_ms: float
    frame_id: int


# ─── Alert ────────────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """Safety alert generated by the Rule Engine.

    Produced by: RuleEngine.evaluate()
    Consumed by: AlertQueue, PiperTTS.speak(), StructuredLogger.log_alert()

    The message field is the ONLY field shown/spoken to the user.
    All other fields are for logging and debugging only.

    Args:
        rule_id:              Unique rule identifier from risk_rules.yaml
        severity:             Alert severity level (used for priority sorting)
        message:              User-facing spoken message (simple, clear English)
        message_hi:           Hindi translation (V2 — None in V1)
        triggering_detections: Detections that caused this alert
        timestamp_ms:         Unix timestamp when alert was generated
        cooldown_seconds:     Minimum seconds before this rule fires again
        frame_id:             Frame where the alert condition was detected
        explanation:          Debug dict — never shown to user, logged for debugging
    """

    rule_id: str
    severity: Severity
    message: str
    message_hi: str | None  # Hindi translation (V2)
    triggering_detections: list[Detection]
    timestamp_ms: float
    cooldown_seconds: int
    frame_id: int
    explanation: dict = field(default_factory=dict)  # Debug metadata only


# ─── Event Memory Entry ───────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    """Temporal tracking entry maintained by EventMemory.

    One entry per detected class, updated each frame.
    Enables temporal rules like "stove visible for >30 seconds".

    Args:
        class_id:         Integer class index
        class_name:       String class name
        last_seen_frame:  Most recent frame where this class was detected
        first_seen_frame: First frame where this class was detected this session
        detection_count:  Total number of frames this class has appeared in
        avg_confidence:   Exponential moving average of detection confidence
        last_bbox:        Most recent bounding box for this class
    """

    class_id: int
    class_name: str
    last_seen_frame: int
    first_seen_frame: int
    detection_count: int
    avg_confidence: float
    last_bbox: BoundingBox


# ─── Performance Metrics ──────────────────────────────────────────────────────


@dataclass
class PipelineMetrics:
    """Per-frame performance telemetry.

    Produced by: Orchestrator.process_frame()
    Consumed by: StructuredLogger.log_metrics()

    All timing values are in milliseconds.
    Used to validate performance budget compliance (see pyproject.toml comments).

    Args:
        frame_id:         Sequential frame counter
        capture_ms:       Camera capture + preprocessing time
        detection_ms:     YOLO inference time
        memory_update_ms: EventMemory.update() time
        vlm_ms:           SmolVLM2 inference time (None if not invoked this frame)
        rule_eval_ms:     RuleEngine.evaluate() time
        tts_queue_ms:     Time to enqueue alert to TTS queue (not TTS synthesis time)
        total_ms:         Total main-thread processing time for this frame
        fps:              Instantaneous FPS (1000 / total_ms)
        ram_mb:           Process RAM usage in MB at this frame
    """

    frame_id: int
    capture_ms: float
    detection_ms: float
    memory_update_ms: float
    vlm_ms: float | None  # None when VLM not invoked
    rule_eval_ms: float
    tts_queue_ms: float | None  # None when no alert generated
    total_ms: float
    fps: float
    ram_mb: float


# ─── Frame Result ─────────────────────────────────────────────────────────────


@dataclass
class FrameResult:
    """Complete result of processing one camera frame through the full pipeline.

    Produced by: Orchestrator.process_frame()
    Consumed by: StructuredLogger.log_frame()

    Args:
        frame_id:      Sequential frame counter for this session
        timestamp_ms:  Unix timestamp in milliseconds
        detections:    All YOLO detections (post confidence filter)
        context:       SmolVLM2 scene analysis (None if VLM disabled or not invoked)
        alerts:        Safety alerts generated by Rule Engine
        metrics:       Performance telemetry for this frame
        mode:          Pipeline operating mode
    """

    frame_id: int
    timestamp_ms: float
    detections: list[Detection]
    context: SceneContext | None
    alerts: list[Alert]
    metrics: PipelineMetrics
    mode: str  # "full" | "yolo_only" | "degraded"


# ─── Module Interfaces (Abstract Base Contracts) ──────────────────────────────
# These define the expected interface for each pipeline component.
# Concrete implementations live in their respective module files.


class BaseDetector:
    """Abstract interface for object detectors."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame."""
        raise NotImplementedError

    def warmup(self, n: int = 3) -> None:
        """Run N dummy inferences to initialize hardware accelerator."""
        raise NotImplementedError

    def get_class_names(self) -> dict[int, str]:
        """Return mapping of class_id → class_name."""
        raise NotImplementedError


class BaseMemory:
    """Abstract interface for event memory."""

    def update(self, detections: list[Detection]) -> None:
        """Advance memory state with this frame's detections."""
        raise NotImplementedError

    def is_present(self, class_id: int) -> bool:
        """Was this class detected in the most recent frame?"""
        raise NotImplementedError

    def frames_since_seen(self, class_id: int) -> int:
        """How many frames ago was this class last detected?"""
        raise NotImplementedError

    def seconds_since_seen(self, class_id: int, fps: float) -> float:
        """How many seconds ago was this class last detected?"""
        raise NotImplementedError

    def get_entry(self, class_id: int) -> MemoryEntry | None:
        """Return the MemoryEntry for a class, or None if never seen."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all memory state (called on shutdown for privacy)."""
        raise NotImplementedError


class BaseAnalyzer:
    """Abstract interface for scene analyzers (VLM)."""

    def analyze(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> SceneContext:
        """Analyze the scene and return structured context."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Return True if the analyzer is loaded and healthy."""
        raise NotImplementedError


class BaseRuleEngine:
    """Abstract interface for safety rule engines."""

    def evaluate(
        self,
        detections: list[Detection],
        memory: BaseMemory,
        context: SceneContext | None = None,
    ) -> list[Alert]:
        """Evaluate all rules and return triggered alerts."""
        raise NotImplementedError

    def reload_rules(self) -> None:
        """Hot-reload rules from YAML without pipeline restart."""
        raise NotImplementedError


class BaseTTS:
    """Abstract interface for TTS engines."""

    def speak(self, text: str, priority: bool = False) -> None:
        """Enqueue text for speech (non-blocking)."""
        raise NotImplementedError

    def is_speaking(self) -> bool:
        """Return True if TTS is currently synthesizing or playing."""
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return True if TTS engine is initialized and healthy."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Gracefully stop the TTS worker thread."""
        raise NotImplementedError

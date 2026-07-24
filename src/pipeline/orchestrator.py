"""
Main Pipeline Orchestrator
===========================
Top-level coordinator for the Elderly Assistant inference pipeline.

Threading model:
  Main thread   → frame capture, YOLO, Event Memory, Rule Engine, Alert enqueue
  TTS thread    → PiperTTS worker (daemon) — consumes alert queue
  VLM thread    — integrated inside SmolVLM2Analyzer (optional)

Graceful degradation:
  YOLO fails        → unrecoverable, raises on init
  SmolVLM2 fails    → transparent fallback to YOLO-only mode
  TTS fails         → silent mode (logs alerts, no speech)
  Storage full      → logger silently caps; pipeline continues

Feature flags loaded from: configs/feature_flags.yaml
Rules loaded from:         configs/risk_rules.yaml
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from ..logging.structured_logger import StructuredLogger
from . import PipelineMetrics, Severity
from .confidence_fusion import ConfidenceFusion
from .detector import YOLODetector
from .event_memory import EventMemory
from .rule_engine import RuleEngine
from .scene_analyzer import SmolVLM2Analyzer
from .tts_engine import PiperTTS

logger = logging.getLogger(__name__)


def _load_flags(path: str = "configs/feature_flags.yaml") -> dict[str, Any]:
    """Load feature flags from YAML. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Feature flags file not found: {path}. Using defaults.")
        return {}
    result = yaml.safe_load(p.read_text(encoding="utf-8")).get("feature_flags", {})
    return result if isinstance(result, dict) else {}


class ElderlyAssistantPipeline:
    """Main pipeline coordinator.

    Initialise once, then call process_frame() in your camera loop.
    Call shutdown() when done.

    Example::

        pipeline = ElderlyAssistantPipeline()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                result = pipeline.process_frame(frame)
        pipeline.shutdown()
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n/weights/best.pt",
        rules_path: str = "configs/risk_rules.yaml",
        tts_model_path: str = "models/tts/en_IN-medium.onnx",
        tts_config_path: str = "models/tts/en_IN-medium.onnx.json",
        flags_path: str = "configs/feature_flags.yaml",
        vlm_model: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        log_dir: str = "logs",
        target_fps: float = 15.0,
    ) -> None:
        self._flags = _load_flags(flags_path)
        self._target_fps = target_fps
        self._frame_count = 0
        self._mode = "initialising"

        # ── YOLO Detector (required) ─────────────────────────────────────
        self._detector = YOLODetector(model_path=model_path)
        self._detector.warmup()

        # ── Event Memory ─────────────────────────────────────────────────
        self._memory = EventMemory(window_size=150)

        # ── Confidence Fusion ────────────────────────────────────────────
        self._fusion = ConfidenceFusion(alpha=0.7, beta=0.3)

        # ── SmolVLM2 Analyzer (optional) ─────────────────────────────────
        vlm_enabled = self._flags.get("vlm_enabled", True)
        if vlm_enabled:
            self._analyzer: SmolVLM2Analyzer | None = SmolVLM2Analyzer(vlm_model)
        else:
            self._analyzer = None
            logger.info("SmolVLM2 disabled via feature flag")

        # ── Rule Engine ──────────────────────────────────────────────────
        self._rule_engine = RuleEngine(rules_path=rules_path, fps=target_fps)

        # ── Piper TTS (non-blocking) ─────────────────────────────────────
        self._tts: PiperTTS | None = None
        try:
            self._tts = PiperTTS(
                model_path=tts_model_path,
                config_path=tts_config_path,
            )
        except Exception as e:
            logger.warning(f"TTS init failed: {e}. Running in silent mode.")

        # ── Logger ───────────────────────────────────────────────────────
        self._logger = StructuredLogger(log_dir=log_dir)

        # ── Background health reporter (every 60s) ───────────────────────
        self._health_thread = threading.Thread(
            target=self._health_reporter, daemon=True, name="health-reporter"
        )
        self._shutdown_event = threading.Event()
        self._health_thread.start()

        # ── Plugins (empty V1 — reserved for V2 fall detection, OCR) ────
        self._plugins: list = []

        self._mode = "yolo_only" if self._analyzer is None else "full"
        logger.info(f"Pipeline ready — mode: {self._mode}")

    # ─────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────

    def process_frame(self, frame: Any) -> dict[str, Any]:
        """Process one camera frame through the full pipeline.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.read().

        Returns:
            Dict with keys: frame_id, detections, alerts, mode, metrics.
        """
        t0 = time.perf_counter()
        self._frame_count += 1
        frame_id = self._frame_count
        context = None

        # ── 1. YOLO detection ─────────────────────────────────────────
        t1 = time.perf_counter()
        try:
            detections = self._detector.detect(frame, frame_id=frame_id)
        except Exception as e:
            self._logger.log_error(e, context="yolo_detect")
            return self._empty_result(frame_id)
        detection_ms = (time.perf_counter() - t1) * 1000

        # ── 2. Update Event Memory ────────────────────────────────────
        t2 = time.perf_counter()
        self._memory.update(detections)
        memory_ms = (time.perf_counter() - t2) * 1000

        # ── 3. SmolVLM2 (every 5th frame, if enabled) ─────────────────
        vlm_ms = None
        t3 = time.perf_counter()
        vlm_interval = 5
        if (
            self._analyzer is not None
            and self._analyzer.is_available()
            and frame_id % vlm_interval == 0
            and detections
        ):
            try:
                context = self._analyzer.analyze(frame, detections, frame_id)
            except Exception as e:
                self._logger.log_error(e, context="vlm_analyze")
                context = None
            vlm_ms = (time.perf_counter() - t3) * 1000

        # ── 4. Confidence Fusion (YOLO + VLM) ────────────────────────
        if context is not None:
            detections = self._fusion.fuse(detections, context)

        # ── 5. Rule Engine ────────────────────────────────────────────
        t4 = time.perf_counter()
        try:
            alerts = self._rule_engine.evaluate(detections, self._memory, context, self._target_fps)
        except Exception as e:
            self._logger.log_error(e, context="rule_engine")
            alerts = []
        rule_ms = (time.perf_counter() - t4) * 1000

        # ── 6. V2+ Plugins (no-op in V1 — empty list) ─────────────────
        for plugin in self._plugins:
            try:
                plugin_alerts = plugin.on_frame(frame)
                alerts.extend(plugin_alerts or [])
            except Exception as e:
                self._logger.log_error(e, context=f"plugin_{type(plugin).__name__}")

        # ── 7. Speak highest-priority alert ───────────────────────────
        if alerts and self._tts is not None:
            top = max(alerts, key=lambda a: a.severity.value)
            self._tts.speak(top.message, priority=(top.severity == Severity.CRITICAL))
            for alert in alerts:
                self._logger.log_alert(alert)

        # ── 8. Assemble metrics & log ─────────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        metrics = PipelineMetrics(
            frame_id=frame_id,
            capture_ms=0.0,  # Measured outside this method
            detection_ms=detection_ms,
            memory_update_ms=memory_ms,
            vlm_ms=vlm_ms,
            rule_eval_ms=rule_ms,
            tts_queue_ms=None,
            total_ms=total_ms,
            fps=1000.0 / max(total_ms, 1.0),
            ram_mb=self._get_ram_mb(),
        )

        if self._flags.get("performance_logging", True):
            self._logger.log_frame(frame_id, detections, alerts, metrics, self._mode)

        return {
            "frame_id": frame_id,
            "detections": detections,
            "alerts": alerts,
            "mode": self._mode,
            "metrics": metrics,
        }

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: flush logs, stop background threads, clear memory."""
        logger.info("Pipeline shutting down...")
        self._shutdown_event.set()
        if self._tts is not None:
            self._tts.shutdown()
        self._memory.clear()
        self._logger.dump_health_summary()
        logger.info("Pipeline shutdown complete")

    def reload_rules(self) -> None:
        """Hot-reload risk rules from YAML without restarting."""
        self._rule_engine.reload_rules()

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _health_reporter(self) -> None:
        """Background thread: dump health snapshot every 60 seconds."""
        while not self._shutdown_event.wait(timeout=60.0):
            self._logger.dump_health_summary()

    def _get_ram_mb(self) -> float:
        try:
            import os

            import psutil

            rss: float = psutil.Process(os.getpid()).memory_info().rss / 1e6
            return rss
        except Exception:
            return 0.0

    def _empty_result(self, frame_id: int) -> dict:
        return {
            "frame_id": frame_id,
            "detections": [],
            "alerts": [],
            "mode": "error",
            "metrics": None,
        }

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
  YOLO taxonomy ≠ configs/data.yaml → unrecoverable, raises on init (ADR-P6-12)
  SmolVLM2 fails    → transparent fallback to YOLO-only mode
  TTS fails         → silent mode (logs alerts, no speech)
  Storage full      → logger silently caps; pipeline continues

Alert fan-out (M9). One alert reaches up to three places, and which ones is
authored per scenario rather than decided here:
  logs/events.jsonl   always, via StructuredLogger
  caregiver sink      when caregiver_channel != none
  speech              when patient_facing, subject to the per-minute cap
Seven of the nine scenarios are caregiver-only, so the second row is not an
afterthought — it is where most of this system's output goes.

Feature flags loaded from: configs/feature_flags.yaml
Rules:                     injected as a BaseRuleEngine (ADR-P6-04). The default
                           implementation is src.scenario_engine.runtime, built
                           by src.app.factory. configs/risk_rules.yaml and its
                           string DSL are retired (ADR-P6-03).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..config.config_loader import SystemConfig
from ..logging.structured_logger import StructuredLogger
from . import Alert, BaseCaregiverSink, BaseRuleEngine, PipelineMetrics, Severity
from .alert_queue import AlertQueue
from .caregiver import LocalCaregiverSink
from .confidence_fusion import ConfidenceFusion
from .detector import YOLODetector
from .event_memory import EventMemory
from .scene_analyzer import SmolVLM2Analyzer
from .tts_engine import PiperTTS

logger = logging.getLogger(__name__)


def resolve_speech_language(configured: str, voice_model_path: str) -> str:
    """Which language the device can actually speak, as a bare code ('en', 'hi').

    ``runtime.tts_language`` states the language the deployment *wants*; the
    Piper voice file determines what it can *pronounce*. Piper voices are named
    ``<lang>_<REGION>-<quality>.onnx``, so the two are checkable against each
    other — and they must be, because feeding Devanagari to an English voice does
    not fail, it produces confident gibberish at a resident who may be alone.

    On disagreement the voice wins and the mismatch is logged: speaking
    understandable English is a recoverable disappointment, speaking noise is not.
    """
    wanted = (configured or "en").split("_")[0].strip().lower() or "en"
    voice_tag = Path(voice_model_path).name.split("-")[0].split("_")[0].strip().lower()
    if not voice_tag:
        return wanted
    if voice_tag != wanted:
        logger.warning(
            f"runtime.tts_language is {configured!r} but the configured voice is "
            f"{Path(voice_model_path).name!r} ({voice_tag}). Speaking {voice_tag}: a voice "
            f"cannot pronounce a language it was not trained for."
        )
    return voice_tag


class ElderlyAssistantPipeline:
    """Main pipeline coordinator.

    Initialise once, then call process_frame() in your camera loop.
    Call shutdown() when done.

    Build it through the composition root, which injects the rule engine::

        from src.app import build_pipeline

        pipeline = build_pipeline(room="kitchen")
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
        scenario_dir: str = "configs/scenarios",
        tts_model_path: str = "models/tts/en_IN-medium.onnx",
        tts_config_path: str = "models/tts/en_IN-medium.onnx.json",
        flags_path: str = "configs/feature_flags.yaml",
        thresholds_path: str = "configs/class_thresholds.yaml",
        vlm_model: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        log_dir: str = "logs",
        target_fps: float | None = None,
        room: str | None = None,
        rule_engine: BaseRuleEngine | None = None,
        caregiver_sink: BaseCaregiverSink | None = None,
    ) -> None:
        """
        Args:
            scenario_dir: Authored scenario masters. Replaces the retired
                ``rules_path``/``configs/risk_rules.yaml`` (ADR-P6-03).
            room: Deployment room for this camera — a static per-camera
                constant that room-scoped scenarios are gated on. A pipeline
                with no room set never fires a room-scoped scenario.
            rule_engine: **Required.** The integration seam (ADR-P6-04): one
                constructor parameter, matching the trainer injection approved
                in Phase 4. Normally
                :class:`~src.scenario_engine.runtime.ScenarioRuleEngine`, built
                by :func:`src.app.factory.build_pipeline`. It is not constructed
                here — see the comment at the assignment for why.
            caregiver_sink: Where ``Alert.caregiver_channel`` alerts go (M9).
                Defaults to :class:`~src.pipeline.caregiver.LocalCaregiverSink`
                writing ``logs/caregiver.jsonl``. Unlike the rule engine this
                *does* default, because it is in this package and a missing sink
                would silently discard the caregiver stage.

        Raises:
            ValueError: If ``rule_engine`` is None.
        """
        # Every component is configured from this one object — no component
        # reads YAML directly (SystemConfig's own documented contract). Until
        # Phase 6 the orchestrator hand-rolled a loader that read a
        # `feature_flags:` root key which does not exist in the file, so every
        # flag silently evaluated to its default and SystemConfig was reachable
        # only from tests.
        self._config = SystemConfig.load(flags_path, thresholds_path)
        self._target_fps = (
            target_fps
            if target_fps is not None
            else float(self._config.get_runtime("target_fps", 15))
        )
        self._measured_fps = self._target_fps
        self._last_frame_ts: float | None = None
        self._frame_count = 0
        self._mode = "initialising"

        # ── YOLO Detector (required) ─────────────────────────────────────
        # Per-class thresholds come from configs/class_thresholds.yaml, which
        # was previously never loaded at runtime.
        # expected_classes makes the weights prove they carry the taxonomy every
        # other layer addresses by name. Without it a model trained on a
        # different class list runs happily and reasons about the wrong world.
        self._detector = YOLODetector(
            model_path=model_path,
            class_thresholds=dict(self._config.class_thresholds),
            disabled_classes={
                name for name, enabled in self._config.classes.items() if not enabled
            },
            expected_classes=dict(self._config.class_names),
        )
        self._detector.warmup()

        # ── Event Memory ─────────────────────────────────────────────────
        self._memory = EventMemory(
            window_size=int(self._config.get_runtime("memory_window_frames", 150))
        )

        # ── Confidence Fusion ────────────────────────────────────────────
        self._fusion = ConfidenceFusion(alpha=0.7, beta=0.3)

        # ── SmolVLM2 Analyzer (optional) ─────────────────────────────────
        # smolvlm_every_n_frames and smolvlm_timeout_ms are documented as the VLM
        # control surface; the sampling interval was hard-coded to 5 in
        # process_frame and the timeout was stored by the analyzer and never
        # applied, so neither key did anything.
        self._vlm_interval = max(1, int(self._config.get_runtime("smolvlm_every_n_frames", 5)))
        vlm_timeout_s = float(self._config.get_runtime("smolvlm_timeout_ms", 2000)) / 1000.0
        if self._config.is_component_enabled("smolvlm_analysis"):
            self._analyzer: SmolVLM2Analyzer | None = SmolVLM2Analyzer(
                vlm_model, timeout_seconds=vlm_timeout_s
            )
        else:
            self._analyzer = None
            logger.info("SmolVLM2 disabled via feature flag")

        # ── Rule Engine ──────────────────────────────────────────────────
        # Constructor injection, not the plugin seam (ADR-P6-04). BaseRuleEngine
        # already declared exactly this signature; the plugin seam had three
        # mutually incompatible contracts and no registration mechanism.
        #
        # configs/risk_rules.yaml and its string DSL are retired (ADR-P6-03) —
        # three of its six rules were deleted rather than migrated, on a
        # projected volume of ~370 alerts/day with ~0 actionable.
        #
        # There is deliberately NO default here. Constructing the scenario
        # engine in this file would make src.pipeline import
        # src.scenario_engine, which already imports src.pipeline — a cycle that
        # works today only because src/pipeline/__init__.py re-exports no
        # submodules, and that would detonate the first time one is added.
        # tests/unit/scenario_engine/test_layering.py enforces the direction.
        #
        # Build the pipeline through src.app.factory.build_pipeline(), the
        # composition root that sits above both packages.
        if rule_engine is None:
            raise ValueError(
                "ElderlyAssistantPipeline needs a rule_engine. configs/risk_rules.yaml "
                "and its string DSL are retired (ADR-P6-03); the replacement is "
                "src.scenario_engine.runtime.ScenarioRuleEngine, injected here rather "
                "than constructed here so the package dependency stays one-way "
                "(ADR-P6-04). Use src.app.factory.build_pipeline(), or pass an engine "
                "explicitly."
            )
        self._rule_engine: BaseRuleEngine = rule_engine

        # ── Alert arbitration ────────────────────────────────────────────
        # A severity-ordered, bounded queue sits between the rule engine and
        # TTS. It was fully implemented and unit-tested but never imported, so
        # the pipeline spoke max(alerts) each frame and discarded the rest.
        # Assigned before any background thread starts.
        self._alert_queue = AlertQueue(max_size=10)
        self._max_alerts_per_minute = int(self._config.get_runtime("max_alerts_per_minute", 6))
        self._spoken_at: deque[float] = deque()

        # ── Piper TTS (non-blocking) ─────────────────────────────────────
        # `tts_output: false` now genuinely silences the device. It was one of
        # eight component flags that nothing consumed — an operator could switch
        # off speech in the config and the device would keep talking.
        self._tts: PiperTTS | None = None
        if not self._config.is_component_enabled("tts_output"):
            logger.info("TTS disabled via feature flag — alerts are logged, not spoken")
        else:
            try:
                self._tts = PiperTTS(
                    model_path=tts_model_path,
                    config_path=tts_config_path,
                    speech_rate=float(self._config.get_runtime("tts_speed", 0.9)),
                )
            except Exception as e:
                logger.warning(f"TTS init failed: {e}. Running in silent mode.")

        self._speech_language = resolve_speech_language(
            str(self._config.get_runtime("tts_language", "en_IN")), tts_model_path
        )

        # ── Caregiver channel ────────────────────────────────────────────
        # The fourth product stage. Local-only by design — see caregiver.py.
        self._caregiver: BaseCaregiverSink = caregiver_sink or LocalCaregiverSink(log_dir=log_dir)

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

        # Track the REAL loop rate. Temporal rules convert Event Memory frame
        # counts into seconds, so feeding them the nominal target while the
        # device is throttling silently rescales every threshold — a 30s
        # "unattended" rule becomes 225s at 2 FPS.
        if self._last_frame_ts is not None:
            interval = t0 - self._last_frame_ts
            if interval > 0:
                self._measured_fps = 0.9 * self._measured_fps + 0.1 * (1.0 / interval)
        self._last_frame_ts = t0

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

        # ── 3. SmolVLM2 (every Nth frame, if enabled) ─────────────────
        vlm_ms = None
        t3 = time.perf_counter()
        if (
            self._analyzer is not None
            and self._analyzer.is_available()
            and frame_id % self._vlm_interval == 0
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
            alerts = self._rule_engine.evaluate(
                detections, self._memory, context, self._measured_fps
            )
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

        # ── 7. Arbitrate, notify the caregiver, and speak ─────────────
        for alert in alerts:
            self._alert_queue.put(alert)
            self._logger.log_alert(alert)
            try:
                self._caregiver.notify(alert)
            except Exception as e:
                self._logger.log_error(e, context="caregiver_sink")

        spoken = self._next_speakable_alert()
        if spoken is not None and self._tts is not None:
            text = self.spoken_text(spoken)
            if text:
                self._tts.speak(text, priority=(spoken.severity == Severity.CRITICAL))

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

        if self._config.is_component_enabled("performance_logging"):
            self._logger.log_frame(frame_id, detections, alerts, metrics, self._mode)

        return {
            "frame_id": frame_id,
            "detections": detections,
            "alerts": alerts,
            "mode": self._mode,
            "metrics": metrics,
        }

    # ─────────────────────────────────────────
    # Alert arbitration
    # ─────────────────────────────────────────

    def spoken_text(self, alert: Alert) -> str:
        """The string to speak for this alert, in the device's speech language.

        ``Alert.messages`` is the open-ended ``{lang: text}`` map added by
        ADR-P6-09; ``message`` is the English string every pre-Phase-6 alert
        carries. Falling back to ``message`` matters more than honouring the
        locale: a resident who hears the wrong language still hears a warning,
        while a resident who hears nothing does not.
        """
        messages = alert.messages or {}
        text = messages.get(self._speech_language, "").strip()
        if text:
            return text
        if messages and self._speech_language != "en":
            logger.warning(
                f"{alert.rule_id} has no {self._speech_language!r} message "
                f"(has {sorted(messages)}) -- speaking the English text."
            )
        return messages.get("en", "").strip() or alert.message

    def _next_speakable_alert(self) -> Alert | None:
        """Pop the highest-priority pending alert, subject to the rate limit.

        ``runtime.max_alerts_per_minute`` in configs/feature_flags.yaml carries
        the comment "Hard cap — prevents alert fatigue" and was referenced by no
        code at all. It is enforced here.

        CRITICAL alerts bypass the cap. A cap that can silence a genuine
        emergency is a worse failure than the fatigue it prevents, and the
        alert-fatigue evidence concerns routine chatter, not emergencies. The
        alert is still queued, logged, and counted either way — only speech is
        rate-limited.

        Alerts with ``patient_facing=False`` are skipped rather than returned.
        ``Alert`` documents that field as "when False, nothing is spoken to the
        resident" and until M9 nothing enforced it, so a caregiver-only finding
        (``SC-BTH-002``, no grab bar) was announced to the resident, and a
        quiet-hours ``caregiver_only`` scenario spoke at 3am — the exact
        behaviour that setting exists to prevent. They are skipped, not merely
        left in the queue, so a silent alert cannot block a speakable one behind
        it; the caregiver sink has already received them.
        """
        alert = self._pop_patient_facing()
        if alert is None:
            return None

        now = time.monotonic()
        while self._spoken_at and now - self._spoken_at[0] >= 60.0:
            self._spoken_at.popleft()

        if (
            alert.severity is not Severity.CRITICAL
            and len(self._spoken_at) >= self._max_alerts_per_minute
        ):
            logger.info(
                f"Rate limit reached ({self._max_alerts_per_minute}/min) — "
                f"not speaking {alert.rule_id} [{alert.severity.name}]"
            )
            return None

        self._spoken_at.append(now)
        return alert

    def _pop_patient_facing(self) -> Alert | None:
        """Drain silent alerts and return the first speakable one, if any."""
        while True:
            alert = self._alert_queue.get(timeout=0)
            if alert is None:
                return None
            if alert.patient_facing:
                return alert
            logger.debug(f"{alert.rule_id} is caregiver-only — not spoken")

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: flush logs, stop background threads, clear memory."""
        logger.info("Pipeline shutting down...")
        self._shutdown_event.set()
        try:
            # Before anything else: a buffered digest that is never flushed is a
            # caregiver notification the caregiver never receives.
            self._caregiver.flush()
        except Exception as e:
            logger.error(f"Caregiver digest flush failed: {e}")
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

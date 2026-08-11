"""
Unit tests for src.pipeline.orchestrator alert arbitration and fan-out.

`runtime.max_alerts_per_minute` in configs/feature_flags.yaml is commented
"Hard cap — prevents alert fatigue" and was referenced by no code whatsoever.
`AlertQueue` was likewise fully implemented, unit-tested, and never imported —
the pipeline spoke `max(alerts)` each frame and discarded the rest.

M9 added the fan-out tests. Step 7 of `process_frame` is where both of that
milestone's defects lived: `patient_facing` was enforced nowhere (7 of the 9
scenarios set it False, and all seven were spoken aloud at the resident) and
`caregiver_channel` had no consumer. The two masked each other — everything
spoken, nothing routed — which is why the seam is now exercised end to end
rather than only by its parts.

Every collaborator is faked at our own module boundary, so these construct the
orchestrator without loading a YOLO model, a TTS voice, or a VLM.
"""

from __future__ import annotations

from collections import deque

import pytest

from src.config.config_loader import SystemConfig
from src.pipeline import Alert, BoundingBox, Detection, Severity
from src.pipeline.alert_queue import AlertQueue
from src.pipeline.orchestrator import ElderlyAssistantPipeline, resolve_speech_language


def _alert(rule_id: str, severity: Severity, **kwargs: object) -> Alert:
    detection = Detection(
        class_id=5,
        class_name="knife",
        confidence=0.9,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1),
        frame_id=1,
        timestamp_ms=0.0,
    )
    fields: dict[str, object] = {
        "rule_id": rule_id,
        "severity": severity,
        "message": f"message for {rule_id}",
        "message_hi": None,
        "triggering_detections": [detection],
        "timestamp_ms": 0.0,
        "cooldown_seconds": 60,
        "frame_id": 1,
        "explanation": {},
    }
    fields.update(kwargs)
    return Alert(**fields)  # type: ignore[arg-type]


class _FakeDetector:
    def detect(self, frame: object, frame_id: int = 0) -> list[Detection]:
        return []


class _FakeMemory:
    def update(self, detections: list[Detection]) -> None:
        return None


class _FakeFusion:
    def fuse(self, detections: list[Detection], context: object) -> list[Detection]:
        return detections


class _FakeEngine:
    def __init__(self, alerts: list[Alert]) -> None:
        self._alerts = alerts

    def evaluate(self, *args: object, **kwargs: object) -> list[Alert]:
        return list(self._alerts)

    def reload_rules(self) -> None:
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def log_alert(self, alert: Alert) -> None:
        return None

    def log_frame(self, *args: object, **kwargs: object) -> None:
        return None

    def log_error(self, error: Exception, context: str = "") -> None:
        self.errors.append(context)


class _FakeSink:
    def __init__(self) -> None:
        self.notified: list[Alert] = []
        self.explode = False

    def notify(self, alert: Alert) -> bool:
        if self.explode:
            raise OSError("no space left on device")
        self.notified.append(alert)
        return True

    def flush(self) -> None:
        return None


class _FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str, priority: bool = False) -> None:
        self.spoken.append(text)


def _pipeline(max_per_minute: int = 6, language: str = "en") -> ElderlyAssistantPipeline:
    """An orchestrator with only the arbitration collaborators wired."""
    pipeline = object.__new__(ElderlyAssistantPipeline)
    pipeline._alert_queue = AlertQueue(max_size=10)
    pipeline._max_alerts_per_minute = max_per_minute
    pipeline._spoken_at = deque()
    pipeline._speech_language = language
    return pipeline


class TestOrdering:
    @pytest.mark.unit
    def test_empty_queue_returns_none(self) -> None:
        assert _pipeline()._next_speakable_alert() is None

    @pytest.mark.unit
    def test_highest_severity_is_spoken_first(self) -> None:
        pipeline = _pipeline()
        for severity in (Severity.INFO, Severity.CRITICAL, Severity.MEDIUM):
            pipeline._alert_queue.put(_alert(severity.name.lower(), severity))

        spoken = pipeline._next_speakable_alert()
        assert spoken is not None
        assert spoken.severity is Severity.CRITICAL

    @pytest.mark.unit
    def test_backlog_survives_across_frames(self) -> None:
        """The old path spoke one alert per frame and threw the rest away."""
        pipeline = _pipeline()
        pipeline._alert_queue.put(_alert("a", Severity.HIGH))
        pipeline._alert_queue.put(_alert("b", Severity.MEDIUM))

        first = pipeline._next_speakable_alert()
        second = pipeline._next_speakable_alert()

        assert first is not None and first.rule_id == "a"
        assert second is not None and second.rule_id == "b"


class TestRateLimit:
    @pytest.mark.unit
    def test_cap_is_enforced(self) -> None:
        pipeline = _pipeline(max_per_minute=3)
        for i in range(5):
            pipeline._alert_queue.put(_alert(f"rule_{i}", Severity.LOW))

        spoken = [pipeline._next_speakable_alert() for _ in range(5)]
        assert sum(1 for alert in spoken if alert is not None) == 3

    @pytest.mark.unit
    def test_critical_bypasses_the_cap(self) -> None:
        """A cap that can silence an emergency is worse than the fatigue it prevents."""
        pipeline = _pipeline(max_per_minute=2)
        for i in range(2):
            pipeline._alert_queue.put(_alert(f"routine_{i}", Severity.INFO))
        for _ in range(2):
            pipeline._next_speakable_alert()

        pipeline._alert_queue.put(_alert("stove_unattended", Severity.CRITICAL))
        spoken = pipeline._next_speakable_alert()

        assert spoken is not None
        assert spoken.rule_id == "stove_unattended"

    @pytest.mark.unit
    def test_non_critical_is_suppressed_once_the_cap_is_hit(self) -> None:
        pipeline = _pipeline(max_per_minute=1)
        pipeline._alert_queue.put(_alert("first", Severity.HIGH))
        pipeline._alert_queue.put(_alert("second", Severity.HIGH))

        assert pipeline._next_speakable_alert() is not None
        assert pipeline._next_speakable_alert() is None

    @pytest.mark.unit
    def test_window_expires_after_sixty_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _pipeline(max_per_minute=1)
        clock = {"now": 1_000.0}
        monkeypatch.setattr("src.pipeline.orchestrator.time.monotonic", lambda: clock["now"])

        pipeline._alert_queue.put(_alert("first", Severity.HIGH))
        assert pipeline._next_speakable_alert() is not None

        pipeline._alert_queue.put(_alert("second", Severity.HIGH))
        assert pipeline._next_speakable_alert() is None

        clock["now"] += 61.0
        pipeline._alert_queue.put(_alert("third", Severity.HIGH))
        assert pipeline._next_speakable_alert() is not None

    @pytest.mark.unit
    def test_suppressed_alert_is_not_requeued(self) -> None:
        """A rate-limited alert is consumed, not left to retry forever."""
        pipeline = _pipeline(max_per_minute=0)
        pipeline._alert_queue.put(_alert("only", Severity.HIGH))

        assert pipeline._next_speakable_alert() is None
        assert pipeline._alert_queue.is_empty()


class TestPatientFacing:
    """`Alert.patient_facing` documents "when False, nothing is spoken to the
    resident". Until M9 nothing enforced it, so a caregiver-only finding was
    announced to the resident and a quiet-hours `caregiver_only` scenario spoke
    at 3am — the exact behaviour that setting exists to prevent."""

    @pytest.mark.unit
    def test_a_caregiver_only_alert_is_never_spoken(self) -> None:
        pipeline = _pipeline()
        pipeline._alert_queue.put(_alert("SC-BTH-002", Severity.MEDIUM, patient_facing=False))

        assert pipeline._next_speakable_alert() is None

    @pytest.mark.unit
    def test_a_silent_alert_does_not_block_a_speakable_one(self) -> None:
        """Skipped, not left in place: otherwise one silent alert gags the queue."""
        pipeline = _pipeline()
        pipeline._alert_queue.put(_alert("silent", Severity.HIGH, patient_facing=False))
        pipeline._alert_queue.put(_alert("audible", Severity.MEDIUM))

        spoken = pipeline._next_speakable_alert()
        assert spoken is not None and spoken.rule_id == "audible"

    @pytest.mark.unit
    def test_silent_alerts_do_not_consume_the_speech_budget(self) -> None:
        pipeline = _pipeline(max_per_minute=1)
        for i in range(3):
            pipeline._alert_queue.put(_alert(f"silent_{i}", Severity.HIGH, patient_facing=False))
        pipeline._alert_queue.put(_alert("audible", Severity.HIGH))

        spoken = pipeline._next_speakable_alert()
        assert spoken is not None and spoken.rule_id == "audible"

    @pytest.mark.unit
    def test_a_critical_caregiver_only_alert_is_still_not_spoken(self) -> None:
        """Severity does not override the setting; the caregiver sink carries it."""
        pipeline = _pipeline()
        pipeline._alert_queue.put(_alert("critical", Severity.CRITICAL, patient_facing=False))
        assert pipeline._next_speakable_alert() is None


class TestSpokenLanguage:
    @pytest.mark.unit
    def test_the_configured_language_is_chosen(self) -> None:
        pipeline = _pipeline(language="hi")
        alert = _alert("SC-KIT-001", Severity.HIGH, messages={"en": "English", "hi": "हिंदी"})
        assert pipeline.spoken_text(alert) == "हिंदी"

    @pytest.mark.unit
    def test_missing_translation_falls_back_to_english(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hearing the wrong language beats hearing nothing."""
        pipeline = _pipeline(language="ta")
        alert = _alert("SC-KIT-001", Severity.HIGH, messages={"en": "Careful"})
        with caplog.at_level("WARNING"):
            assert pipeline.spoken_text(alert) == "Careful"
        assert "no 'ta' message" in caplog.text

    @pytest.mark.unit
    def test_a_pre_phase_six_alert_still_speaks(self) -> None:
        """`messages` defaults to None on every alert built before ADR-P6-09."""
        pipeline = _pipeline()
        assert pipeline.spoken_text(_alert("legacy", Severity.LOW)) == "message for legacy"


class TestFrameFanOut:
    """One alert reaches up to three places — log, caregiver sink, speech — and
    which ones is authored per scenario. Step 7 of `process_frame` is where both
    M9 defects lived, so it is exercised end to end rather than by its parts."""

    @staticmethod
    def _wired(alerts: list[Alert]) -> tuple[ElderlyAssistantPipeline, _FakeSink, _FakeTTS]:
        pipeline = _pipeline()
        sink, tts = _FakeSink(), _FakeTTS()
        pipeline._detector = _FakeDetector()
        pipeline._memory = _FakeMemory()
        pipeline._fusion = _FakeFusion()
        pipeline._analyzer = None
        pipeline._vlm_interval = 5
        pipeline._rule_engine = _FakeEngine(alerts)
        pipeline._logger = _FakeLogger()
        pipeline._caregiver = sink
        pipeline._tts = tts
        pipeline._config = SystemConfig(components={"performance_logging": False})
        pipeline._plugins = []
        pipeline._frame_count = 0
        pipeline._last_frame_ts = None
        pipeline._measured_fps = 15.0
        pipeline._mode = "yolo_only"
        return pipeline, sink, tts

    @pytest.mark.unit
    def test_a_caregiver_only_alert_is_routed_but_not_spoken(self) -> None:
        alert = _alert("SC-BTH-002", Severity.LOW, patient_facing=False, caregiver_channel="digest")
        pipeline, sink, tts = self._wired([alert])

        pipeline.process_frame(object())

        assert [a.rule_id for a in sink.notified] == ["SC-BTH-002"]
        assert tts.spoken == []

    @pytest.mark.unit
    def test_a_spoken_alert_also_reaches_the_caregiver(self) -> None:
        alert = _alert("SC-KIT-001", Severity.CRITICAL, caregiver_channel="push")
        pipeline, sink, tts = self._wired([alert])

        pipeline.process_frame(object())

        assert [a.rule_id for a in sink.notified] == ["SC-KIT-001"]
        assert tts.spoken == ["message for SC-KIT-001"]

    @pytest.mark.unit
    def test_a_failing_sink_does_not_stop_the_frame(self) -> None:
        """A caregiver channel is not allowed to take the safety pipeline down."""
        pipeline, sink, tts = self._wired([_alert("SC-KIT-001", Severity.HIGH)])
        sink.explode = True

        result = pipeline.process_frame(object())

        assert result["alerts"]
        assert tts.spoken == ["message for SC-KIT-001"]


class TestSpeechLanguageResolution:
    @pytest.mark.unit
    def test_matching_config_and_voice(self) -> None:
        assert resolve_speech_language("en_IN", "models/tts/en_IN-medium.onnx") == "en"

    @pytest.mark.unit
    def test_the_voice_wins_a_disagreement(self, caplog: pytest.LogCaptureFixture) -> None:
        """Devanagari through an English voice is confident gibberish, not an accent."""
        with caplog.at_level("WARNING"):
            resolved = resolve_speech_language("hi_IN", "models/tts/en_IN-medium.onnx")
        assert resolved == "en"
        assert "cannot pronounce" in caplog.text

    @pytest.mark.unit
    def test_a_hindi_voice_enables_hindi(self) -> None:
        assert resolve_speech_language("hi_IN", "models/tts/hi_IN-medium.onnx") == "hi"

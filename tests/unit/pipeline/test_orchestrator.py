"""
Unit tests for src.pipeline.orchestrator alert arbitration.

`runtime.max_alerts_per_minute` in configs/feature_flags.yaml is commented
"Hard cap — prevents alert fatigue" and was referenced by no code whatsoever.
`AlertQueue` was likewise fully implemented, unit-tested, and never imported —
the pipeline spoke `max(alerts)` each frame and discarded the rest.

`_next_speakable_alert` depends only on the alert queue, the spoken-timestamp
deque, and the cap, so these construct the orchestrator without loading a YOLO
model, a TTS voice, or a VLM.
"""

from __future__ import annotations

from collections import deque

import pytest

from src.pipeline import Alert, BoundingBox, Detection, Severity
from src.pipeline.alert_queue import AlertQueue
from src.pipeline.orchestrator import ElderlyAssistantPipeline


def _alert(rule_id: str, severity: Severity) -> Alert:
    detection = Detection(
        class_id=5,
        class_name="knife",
        confidence=0.9,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1),
        frame_id=1,
        timestamp_ms=0.0,
    )
    return Alert(
        rule_id=rule_id,
        severity=severity,
        message=f"message for {rule_id}",
        message_hi=None,
        triggering_detections=[detection],
        timestamp_ms=0.0,
        cooldown_seconds=60,
        frame_id=1,
        explanation={},
    )


def _pipeline(max_per_minute: int = 6) -> ElderlyAssistantPipeline:
    """An orchestrator with only the arbitration collaborators wired."""
    pipeline = object.__new__(ElderlyAssistantPipeline)
    pipeline._alert_queue = AlertQueue(max_size=10)
    pipeline._max_alerts_per_minute = max_per_minute
    pipeline._spoken_at = deque()
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

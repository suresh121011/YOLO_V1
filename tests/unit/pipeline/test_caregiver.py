"""
Unit tests for src.pipeline.caregiver — the caregiver notification sink (M9).

`Alert.caregiver_channel` was added in ADR-P6-09 with no consumer at all, and
that ADR recorded "M9 specifies the sink". These tests pin the two behaviours
that make the sink worth having rather than reassuring:

* a `none` channel is distinguishable from a failed delivery, and
* `push_and_call` cannot quietly look delivered when no telephony exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline import Alert, BaseCaregiverSink, BoundingBox, Detection, Severity
from src.pipeline.caregiver import LocalCaregiverSink


def _alert(channel: str = "push", **kwargs: object) -> Alert:
    detection = Detection(
        class_id=18,
        class_name="toilet",
        confidence=0.8,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.2),
        frame_id=7,
        timestamp_ms=0.0,
    )
    defaults: dict[str, object] = {
        "rule_id": "SC-BTH-002",
        "severity": Severity.MEDIUM,
        "message": "message",
        "message_hi": None,
        "triggering_detections": [detection],
        "timestamp_ms": 0.0,
        "cooldown_seconds": 60,
        "frame_id": 7,
        "explanation": {},
        "scenario_id": "SC-BTH-002",
        "next_best_action": "Fit a grab bar beside the toilet.",
        "caregiver_channel": channel,
    }
    defaults.update(kwargs)
    return Alert(**defaults)  # type: ignore[arg-type]


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestChannelRouting:
    @pytest.mark.unit
    def test_channel_none_is_not_delivered(self, tmp_path: Path) -> None:
        sink = LocalCaregiverSink(log_dir=tmp_path)
        assert sink.notify(_alert(channel="none")) is False
        assert _records(sink.path) == []

    @pytest.mark.unit
    def test_push_is_written_immediately(self, tmp_path: Path) -> None:
        sink = LocalCaregiverSink(log_dir=tmp_path)
        assert sink.notify(_alert(channel="push")) is True

        records = _records(sink.path)
        assert len(records) == 1
        assert records[0]["scenario_id"] == "SC-BTH-002"
        assert records[0]["next_best_action"].startswith("Fit a grab bar")

    @pytest.mark.unit
    def test_digest_is_buffered_until_flush(self, tmp_path: Path) -> None:
        sink = LocalCaregiverSink(log_dir=tmp_path)
        for _ in range(3):
            sink.notify(_alert(channel="digest"))

        assert sink.pending_digest == 3
        assert _records(sink.path) == []

        sink.flush()
        records = _records(sink.path)
        assert len(records) == 1
        assert records[0]["type"] == "caregiver_digest"
        assert records[0]["count"] == 3

    @pytest.mark.unit
    def test_flush_with_nothing_buffered_writes_nothing(self, tmp_path: Path) -> None:
        sink = LocalCaregiverSink(log_dir=tmp_path)
        sink.flush()
        assert _records(sink.path) == []

    @pytest.mark.unit
    def test_digest_is_bounded_and_reports_what_it_dropped(self, tmp_path: Path) -> None:
        """A device runs for months; an unbounded digest is a memory leak."""
        sink = LocalCaregiverSink(log_dir=tmp_path, max_digest=2)
        for _ in range(5):
            sink.notify(_alert(channel="digest"))
        sink.flush()

        record = _records(sink.path)[0]
        assert record["count"] == 2
        assert record["dropped"] == 3

    @pytest.mark.unit
    def test_unknown_channel_is_delivered_rather_than_dropped(self, tmp_path: Path) -> None:
        """Over-notifying is recoverable; a silently withheld alert is not."""
        sink = LocalCaregiverSink(log_dir=tmp_path)
        assert sink.notify(_alert(channel="carrier-pigeon")) is True

        record = _records(sink.path)[0]
        assert record["unknown_channel"] is True


class TestEscalationHonesty:
    @pytest.mark.unit
    def test_push_and_call_is_marked_pending_because_nobody_is_called(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No telephony exists. The record must not read as a completed escalation."""
        sink = LocalCaregiverSink(log_dir=tmp_path)
        with caplog.at_level("WARNING"):
            sink.notify(_alert(channel="push_and_call", severity=Severity.CRITICAL))

        record = _records(sink.path)[0]
        assert record["escalation_pending"] is True
        assert "NOBODY HAS BEEN CALLED" in caplog.text

    @pytest.mark.unit
    def test_plain_push_is_not_marked_pending(self, tmp_path: Path) -> None:
        sink = LocalCaregiverSink(log_dir=tmp_path)
        sink.notify(_alert(channel="push"))
        assert "escalation_pending" not in _records(sink.path)[0]


class TestPrivacy:
    @pytest.mark.unit
    def test_no_bounding_boxes_reach_the_caregiver_log(self, tmp_path: Path) -> None:
        """Class names only. A second alert log leaking geometry reopens the
        hole StructuredLogger's redaction exists to close."""
        person = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(cx=0.41, cy=0.62, w=0.2, h=0.5),
            frame_id=7,
            timestamp_ms=0.0,
        )
        sink = LocalCaregiverSink(log_dir=tmp_path)
        sink.notify(_alert(channel="push", triggering_detections=[person]))

        raw = sink.path.read_text(encoding="utf-8")
        assert "person" in raw
        for coordinate in ("0.41", "0.62", "bbox", "cx"):
            assert coordinate not in raw


class TestContract:
    @pytest.mark.unit
    def test_sink_satisfies_the_published_protocol(self, tmp_path: Path) -> None:
        assert isinstance(LocalCaregiverSink(log_dir=tmp_path), BaseCaregiverSink)

    @pytest.mark.unit
    def test_write_failure_does_not_take_the_pipeline_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full disk must not stop a safety pipeline from detecting hazards."""
        sink = LocalCaregiverSink(log_dir=tmp_path)

        def explode(*args: object, **kwargs: object) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr("builtins.open", explode)
        assert sink.notify(_alert(channel="push")) is True
        assert sink.delivered == 0

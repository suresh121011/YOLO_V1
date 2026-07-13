"""
Unit tests for src.pipeline.alert_queue.
"""
from __future__ import annotations

import time
import threading

import pytest

from src.pipeline import Alert, BoundingBox, Detection, Severity
from src.pipeline.alert_queue import AlertQueue


def _make_alert(rule_id: str, severity: Severity) -> Alert:
    """Helper to create a minimal Alert for testing."""
    return Alert(
        rule_id=rule_id,
        severity=severity,
        message=f"Test alert: {rule_id}",
        message_hi=None,
        triggering_detections=[],
        timestamp_ms=time.time() * 1000,
        cooldown_seconds=60,
        frame_id=1,
    )


class TestAlertQueue:
    """Test AlertQueue priority and thread safety."""

    @pytest.mark.unit
    def test_higher_severity_dequeued_first(self) -> None:
        """CRITICAL alert should be dequeued before HIGH alert."""
        queue = AlertQueue()
        queue.put(_make_alert("wet_floor", Severity.HIGH))
        queue.put(_make_alert("stove_unattended", Severity.CRITICAL))
        first = queue.get(timeout=0)
        assert first is not None
        assert first.rule_id == "stove_unattended"

    @pytest.mark.unit
    def test_put_and_get_single_alert(self) -> None:
        """Single alert should be retrievable."""
        queue = AlertQueue()
        alert = _make_alert("knife_near_person", Severity.HIGH)
        queue.put(alert)
        result = queue.get(timeout=0)
        assert result is not None
        assert result.rule_id == "knife_near_person"

    @pytest.mark.unit
    def test_get_returns_none_on_empty_with_timeout(self) -> None:
        """get() with timeout=0 should return None when queue is empty."""
        queue = AlertQueue()
        result = queue.get(timeout=0)
        assert result is None

    @pytest.mark.unit
    def test_queue_bounded(self) -> None:
        """Queue should not grow beyond max_size."""
        queue = AlertQueue(max_size=3)
        for i in range(5):
            queue.put(_make_alert(f"rule_{i}", Severity.INFO))
        assert queue.size() <= 3

    @pytest.mark.unit
    def test_critical_survives_overflow(self) -> None:
        """CRITICAL alert should survive when queue is full of INFO alerts."""
        queue = AlertQueue(max_size=3)
        for i in range(3):
            queue.put(_make_alert(f"info_{i}", Severity.INFO))
        critical = _make_alert("critical_rule", Severity.CRITICAL)
        result = queue.put(critical)
        # Critical should have been enqueued (INFO was dropped)
        assert result is True

    @pytest.mark.unit
    def test_is_empty(self) -> None:
        """is_empty() should reflect queue state."""
        queue = AlertQueue()
        assert queue.is_empty()
        queue.put(_make_alert("test", Severity.INFO))
        assert not queue.is_empty()

    @pytest.mark.unit
    def test_clear(self) -> None:
        """clear() should empty the queue."""
        queue = AlertQueue()
        queue.put(_make_alert("test", Severity.HIGH))
        queue.clear()
        assert queue.is_empty()

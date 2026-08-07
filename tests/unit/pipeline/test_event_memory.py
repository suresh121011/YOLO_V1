"""
Unit tests for src.pipeline.event_memory.

Pins the never-seen absence semantics fixed in Phase 6 M1
(docs/08_scenario_engineering/architecture_review.md §3, D4) — the defect that
made ``stove_unattended``, the only CRITICAL rule in the system, unfireable in
a room nobody had yet walked into.
"""

from __future__ import annotations

import pytest

from src.pipeline import BoundingBox, Detection
from src.pipeline.event_memory import EventMemory


def _detection(class_id: int, class_name: str, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.2),
        frame_id=1,
        timestamp_ms=0.0,
    )


PERSON = _detection(0, "person")
STOVE = _detection(6, "stove")


@pytest.fixture
def memory() -> EventMemory:
    return EventMemory(window_size=150)


class TestPresence:
    @pytest.mark.unit
    def test_is_present_reflects_latest_frame(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        assert memory.is_present(0) is True
        memory.update([STOVE])
        assert memory.is_present(0) is False
        assert memory.is_present(6) is True

    @pytest.mark.unit
    def test_empty_memory_has_nothing_present(self, memory: EventMemory) -> None:
        assert memory.is_present(0) is False
        assert memory.is_present_by_name("person") is False

    @pytest.mark.unit
    def test_has_ever_seen(self, memory: EventMemory) -> None:
        assert memory.has_ever_seen("person") is False
        memory.update([PERSON])
        memory.update([STOVE])
        assert memory.has_ever_seen("person") is True
        assert memory.has_ever_seen("knife") is False


class TestNeverSeenAbsence:
    """D4 — the cold-start defect."""

    @pytest.mark.unit
    def test_never_seen_absence_grows_with_session(self, memory: EventMemory) -> None:
        """It used to saturate at window_size (150 frames = 10.0s at 15 FPS)."""
        for _ in range(600):
            memory.update([STOVE])
        assert memory.frames_since_seen_by_name("person") == 600
        assert memory.frames_since_seen(0) == 600

    @pytest.mark.unit
    def test_stove_unattended_can_fire_without_a_person_ever_seen(
        self, memory: EventMemory
    ) -> None:
        """The exact scenario the old semantics made impossible."""
        for _ in range(449):
            memory.update([STOVE])
        # 449 frames at 15 FPS is 29.9s — just under the rule's 30s threshold.
        assert memory.is_absent_for_by_name("person", 30.0, 15.0) is False

        memory.update([STOVE])  # 450 frames = 30.0s
        assert memory.is_absent_for_by_name("person", 30.0, 15.0) is True

    @pytest.mark.unit
    def test_old_saturation_point_is_gone(self, memory: EventMemory) -> None:
        """Absence past the 10s ceiling used to be unrepresentable."""
        for _ in range(1000):
            memory.update([STOVE])
        seconds_absent = memory.frames_since_seen_by_name("person") / 15.0
        assert seconds_absent > 10.0


class TestAbsenceAfterSighting:
    @pytest.mark.unit
    def test_absence_measured_from_last_sighting(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        for _ in range(300):
            memory.update([STOVE])
        assert memory.frames_since_seen_by_name("person") == 300

    @pytest.mark.unit
    def test_present_class_has_zero_absence(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        assert memory.frames_since_seen_by_name("person") == 0
        assert memory.is_absent_for_by_name("person", 0.1, 15.0) is False

    @pytest.mark.unit
    def test_absence_uses_supplied_fps(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        for _ in range(120):
            memory.update([STOVE])
        # 120 frames: 8s at 15 FPS, 60s at a throttled 2 FPS.
        assert memory.is_absent_for_by_name("person", 30.0, 15.0) is False
        assert memory.is_absent_for_by_name("person", 30.0, 2.0) is True


class TestEntries:
    @pytest.mark.unit
    def test_detection_count_and_bbox_tracked(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        memory.update([PERSON])
        entry = memory.get_entry(0)
        assert entry is not None
        assert entry.detection_count == 2
        assert entry.first_seen_frame == 1
        assert entry.last_seen_frame == 2
        assert entry.last_bbox is not None

    @pytest.mark.unit
    def test_consecutive_frames_counts_trailing_run(self, memory: EventMemory) -> None:
        memory.update([PERSON])
        memory.update([STOVE])
        memory.update([PERSON])
        memory.update([PERSON])
        assert memory.consecutive_frames(0) == 2
        assert memory.consecutive_frames(6) == 0

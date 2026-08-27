"""
Unit tests for src.scenario_engine.context.

EvalContext exists because the legacy evaluator received a bare ``set[str]``,
destroying geometry and multiplicity before any predicate ran. These tests pin
the two capabilities that set could not express — boxes and counts — since
losing either would silently degrade spatial predicates into plain presence
checks (ADR-P6-03).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.pipeline import BoundingBox, Detection
from src.scenario_engine import EvalContext, MemoryView
from src.scenario_engine.context import MemoryView as MemoryViewProtocol


class _StubMemory:
    def __init__(self, absent: set[str] | None = None, ever_seen: set[str] | None = None) -> None:
        self._absent = absent or set()
        self._ever_seen = ever_seen or set()

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        return class_name in self._absent

    def consecutive_frames(self, class_id: int) -> int:
        return 0

    def has_ever_seen(self, class_name: str) -> bool:
        return class_name in self._ever_seen


def _detection(class_name: str, cx: float = 0.5, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(cx=cx, cy=0.5, w=0.1, h=0.1),
        frame_id=1,
        timestamp_ms=0.0,
    )


def _context(*detections: Detection, **kwargs: object) -> EvalContext:
    return EvalContext.from_frame(list(detections), _StubMemory(), 15.0, **kwargs)  # type: ignore[arg-type]


class TestPresence:
    @pytest.mark.unit
    def test_present_and_detected_names(self) -> None:
        ctx = _context(_detection("person"), _detection("stove"))
        assert ctx.present("person") is True
        assert ctx.present("knife") is False
        assert ctx.detected_names == frozenset({"person", "stove"})

    @pytest.mark.unit
    def test_empty_frame(self) -> None:
        ctx = _context()
        assert ctx.detected_names == frozenset()
        assert ctx.present("person") is False
        assert ctx.count("person") == 0
        assert ctx.boxes("person") == ()
        assert ctx.best("person") is None


class TestMultiplicity:
    @pytest.mark.unit
    def test_count_distinguishes_one_person_from_three(self) -> None:
        """A `set[str]` could not express this — and multi-person homes need it."""
        ctx = _context(
            _detection("person", cx=0.2),
            _detection("person", cx=0.5),
            _detection("person", cx=0.8),
            _detection("stove"),
        )
        assert ctx.count("person") == 3
        assert ctx.count("stove") == 1
        assert ctx.count("knife") == 0

    @pytest.mark.unit
    def test_instances_returns_each_detection(self) -> None:
        ctx = _context(_detection("person", cx=0.2), _detection("person", cx=0.8))
        assert len(ctx.instances("person")) == 2


class TestGeometry:
    @pytest.mark.unit
    def test_boxes_survive_into_the_context(self) -> None:
        """Geometry was destroyed before predicates ran in the legacy engine."""
        ctx = _context(_detection("person", cx=0.2), _detection("stove", cx=0.9))
        person_box = ctx.boxes("person")[0]
        stove_box = ctx.boxes("stove")[0]
        assert person_box.cx == pytest.approx(0.2)
        assert abs(person_box.cx - stove_box.cx) == pytest.approx(0.7)

    @pytest.mark.unit
    def test_best_returns_highest_confidence_instance(self) -> None:
        ctx = _context(
            _detection("person", cx=0.2, conf=0.55),
            _detection("person", cx=0.8, conf=0.91),
        )
        best = ctx.best("person")
        assert best is not None
        assert best.confidence == pytest.approx(0.91)
        assert best.bbox.cx == pytest.approx(0.8)


class TestContract:
    @pytest.mark.unit
    def test_room_defaults_to_none_and_is_carried(self) -> None:
        assert _context().room is None
        assert _context(room="kitchen").room == "kitchen"

    @pytest.mark.unit
    def test_measured_fps_is_carried(self) -> None:
        ctx = EvalContext.from_frame([], _StubMemory(), 2.4)
        assert ctx.fps == pytest.approx(2.4)

    @pytest.mark.unit
    def test_detections_are_stored_immutably(self) -> None:
        source = [_detection("person")]
        ctx = EvalContext.from_frame(source, _StubMemory(), 15.0)
        source.append(_detection("knife"))
        assert ctx.detected_names == frozenset({"person"})

    @pytest.mark.unit
    def test_is_frozen(self) -> None:
        ctx = _context()
        with pytest.raises(FrozenInstanceError):
            ctx.fps = 30.0  # type: ignore[misc]

    @pytest.mark.unit
    def test_event_memory_satisfies_the_protocol(self) -> None:
        """The real EventMemory must remain structurally compatible."""
        from src.pipeline.event_memory import EventMemory

        assert isinstance(EventMemory(window_size=10), MemoryViewProtocol)
        assert MemoryView is MemoryViewProtocol

"""
Unit tests for src.scenario_engine.runtime.

The state machine is where alarm fatigue is either designed out or let in, so
each transition gets a test with the clock injected rather than slept through.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipeline import BoundingBox, Detection
from src.pipeline.event_memory import EventMemory
from src.scenario_engine.runtime import ScenarioRuleEngine, ScenarioRuntimeError

_SCENARIO: dict[str, Any] = {
    "scenario_id": "SC-BTH-001",
    "schema_version": 1,
    "scenario_name": "Wet floor",
    "category": "bathroom",
    "trigger": {
        "all": [
            {"op": "detected", "args": {"class_name": "wet_floor"}},
            {"op": "present_for", "args": {"class_name": "wet_floor", "seconds": 3}},
            {"op": "detected", "args": {"class_name": "person"}},
        ]
    },
    "risk_level": "HIGH",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "active",
    "reviewed_by": "test",
    "reviewed_on": "2026-08-11",
    "caregiver_channel": "digest",
    "next_best_action": "Ask someone to dry the floor",
    "messages": {"en": "The floor looks wet. Please walk carefully."},
    "min_confidence": 0.30,
    "min_dwell_seconds": 3.0,
    "cooldown_seconds": 120,
    "max_repeats": 3,
    "max_per_day": 6,
    "priority": 4,
    "clear_after_seconds": 10.0,
    "room_context": "bathroom",
    "quiet_hours": {"start": "21:00", "end": "06:00", "behaviour": "suppress"},
    "evidence": [
        {
            "source_id": "CDC-STEADI-CheckForSafety-2017",
            "section": "Bathroom",
            "item": "Non-slip surfaces and grab bars",
            "strength": "guideline",
            "supports": "hazard",
        }
    ],
    "escalation": [],
}


def _write(directory: Path, **overrides: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = {**_SCENARIO, **overrides}
    path = directory / f"{raw['scenario_id']}.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return directory


def _det(class_name: str, frame_id: int = 0, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.2),
        frame_id=frame_id,
        timestamp_ms=0.0,
    )


def _noon() -> datetime:
    return datetime(2026, 8, 11, 12, 0, 0)


def _midnight() -> datetime:
    return datetime(2026, 8, 11, 23, 30, 0)


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _engine(directory: Path, clock: _Clock, wall: Any = _noon, **kwargs: Any) -> ScenarioRuleEngine:
    return ScenarioRuleEngine(
        scenario_dir=directory,
        room=kwargs.pop("room", "bathroom"),
        fps=15.0,
        clock=clock,
        wall_clock=wall,
        **kwargs,
    )


def _run(
    engine: ScenarioRuleEngine, memory: EventMemory, clock: _Clock, seconds: float
) -> list[str]:
    """Feed a wet floor + person for N seconds at 15 fps. Returns fired ids."""
    fired: list[str] = []
    for i in range(int(seconds * 15)):
        frame = [_det("wet_floor", i), _det("person", i)]
        memory.update(frame)
        clock.advance(1.0 / 15.0)
        fired.extend(a.rule_id for a in engine.evaluate(frame, memory, current_fps=15.0))
    return fired


class TestLoading:
    @pytest.mark.unit
    def test_draft_scenarios_do_not_run(self, tmp_path: Path) -> None:
        """Promotion needs a clinical reviewer, not an engineer."""
        directory = _write(tmp_path / "s", status="draft", reviewed_by="", reviewed_on="")
        with pytest.raises(ScenarioRuntimeError, match="draft"):
            ScenarioRuleEngine(scenario_dir=directory)

    @pytest.mark.unit
    def test_an_empty_active_set_is_an_error_not_a_silent_start(self, tmp_path: Path) -> None:
        directory = tmp_path / "empty"
        directory.mkdir()
        with pytest.raises(ScenarioRuntimeError):
            ScenarioRuleEngine(scenario_dir=directory)

    @pytest.mark.unit
    def test_feature_flags_can_disable_a_scenario(self, tmp_path: Path) -> None:
        directory = _write(tmp_path / "s")
        with pytest.raises(ScenarioRuntimeError):
            ScenarioRuleEngine(scenario_dir=directory, scenario_enabled=lambda _id: False)

    @pytest.mark.unit
    def test_active_scenarios_load(self, tmp_path: Path) -> None:
        engine = ScenarioRuleEngine(scenario_dir=_write(tmp_path / "s"))
        assert [s.scenario_id for s in engine.scenarios] == ["SC-BTH-001"]


class TestDwell:
    @pytest.mark.unit
    def test_a_flicker_does_not_alert(self, tmp_path: Path) -> None:
        """One frame of a wet floor is a detector blip, not a hazard."""
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=1.0) == []

    @pytest.mark.unit
    def test_it_alerts_once_the_dwell_elapses(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=8.0) == ["SC-BTH-001"]

    @pytest.mark.unit
    def test_it_alerts_exactly_once_within_the_cooldown(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=60.0).count("SC-BTH-001") == 1


class TestRoomGating:
    @pytest.mark.unit
    def test_a_bathroom_scenario_is_inert_in_the_kitchen(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock, room="kitchen")
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=30.0) == []

    @pytest.mark.unit
    def test_an_unset_room_never_fires_a_room_scoped_scenario(self, tmp_path: Path) -> None:
        """Worth pinning: a camera with no configured room is silently inert."""
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock, room=None)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=30.0) == []


class TestConfidenceFloor:
    @pytest.mark.unit
    def test_a_low_confidence_detection_does_not_alert(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s", min_confidence=0.85), clock)
        memory = EventMemory(window_size=2700)

        fired: list[str] = []
        for i in range(300):
            frame = [_det("wet_floor", i, confidence=0.40), _det("person", i, confidence=0.9)]
            memory.update(frame)
            clock.advance(1.0 / 15.0)
            fired.extend(a.rule_id for a in engine.evaluate(frame, memory, current_fps=15.0))
        assert fired == []


class TestQuietHours:
    @pytest.mark.unit
    def test_suppress_silences_the_alert_overnight(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock, wall=_midnight)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=30.0) == []

    @pytest.mark.unit
    def test_caregiver_only_still_alerts_but_does_not_speak(self, tmp_path: Path) -> None:
        clock = _Clock()
        directory = _write(
            tmp_path / "s",
            quiet_hours={"start": "21:00", "end": "06:00", "behaviour": "caregiver_only"},
        )
        engine = _engine(directory, clock, wall=_midnight)
        memory = EventMemory(window_size=2700)

        alerts = []
        for i in range(300):
            frame = [_det("wet_floor", i), _det("person", i)]
            memory.update(frame)
            clock.advance(1.0 / 15.0)
            alerts.extend(engine.evaluate(frame, memory, current_fps=15.0))

        assert alerts, "caregiver_only must still produce an alert"
        assert alerts[0].patient_facing is False, "nothing is spoken to the resident at night"


class TestVolumeCaps:
    @pytest.mark.unit
    def test_max_repeats_bounds_a_persistent_hazard(self, tmp_path: Path) -> None:
        """The mechanism that took the projected volume from ~370/day to 5."""
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s", max_repeats=2, cooldown_seconds=10), clock)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=600.0).count("SC-BTH-001") == 2

    @pytest.mark.unit
    def test_max_repeats_above_three_is_refused_by_the_schema(self, tmp_path: Path) -> None:
        """Repetition is not escalation. The cap is a schema invariant, so the
        runtime cannot be handed a scenario that nags."""
        directory = _write(tmp_path / "s", max_repeats=99)
        with pytest.raises(ScenarioRuntimeError, match="max_repeats"):
            ScenarioRuleEngine(scenario_dir=directory)

    @pytest.mark.unit
    def test_max_per_day_binds_before_max_repeats(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(
            _write(tmp_path / "s", max_repeats=3, max_per_day=2, cooldown_seconds=5), clock
        )
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=300.0).count("SC-BTH-001") == 2


class TestBootTimeCooldown:
    @pytest.mark.unit
    def test_a_freshly_booted_device_is_not_silenced(self, tmp_path: Path) -> None:
        """Regression for the 0.0-sentinel defect fixed in the retired engine.

        time.monotonic() counts from boot, so a 0.0 "never fired" sentinel makes
        a device suppress every scenario for a full cooldown after a power cut.
        """
        clock = _Clock(start=3.0)  # 3 seconds of uptime
        engine = _engine(_write(tmp_path / "s", cooldown_seconds=1800), clock)
        memory = EventMemory(window_size=2700)
        assert _run(engine, memory, clock, seconds=10.0) == ["SC-BTH-001"]


class TestAlertContents:
    @pytest.mark.unit
    def test_the_alert_carries_the_four_stage_output(self, tmp_path: Path) -> None:
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock)
        memory = EventMemory(window_size=2700)

        alert = None
        for i in range(300):
            frame = [_det("wet_floor", i), _det("person", i)]
            memory.update(frame)
            clock.advance(1.0 / 15.0)
            for a in engine.evaluate(frame, memory, current_fps=15.0):
                alert = alert or a

        assert alert is not None
        assert alert.scenario_id == "SC-BTH-001"
        assert alert.next_best_action == "Ask someone to dry the floor"
        assert alert.caregiver_channel == "digest"
        assert alert.messages == {"en": "The floor looks wet. Please walk carefully."}
        assert alert.severity.name == "HIGH"


class TestDiagnostics:
    @pytest.mark.unit
    def test_decisions_explain_silence(self, tmp_path: Path) -> None:
        """ "It said nothing" and "there was no hazard" must be distinguishable."""
        clock = _Clock()
        engine = _engine(_write(tmp_path / "s"), clock)
        memory = EventMemory(window_size=2700)

        frame = [_det("wet_floor"), _det("person")]
        memory.update(frame)
        engine.evaluate(frame, memory, current_fps=15.0)

        reasons = {d.scenario_id: d.reason for d in engine.last_decisions()}
        assert reasons["SC-BTH-001"] in {"dwelling", "condition_false"}

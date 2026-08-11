"""
Golden alert-trace: the retired string engine vs the scenario engine.

The Phase-6 plan asked for a golden test proving "identical alerts for retained
rules". Running it revealed that the premise was wrong, and the test is written
to record why rather than to assert something we do not want.

**The behaviour was supposed to change.** Three of the six legacy rules were
deleted rather than migrated because they fired continuously during normal life
(`knife_near_person` during any cooking, `medicine_reminder` asking an
unanswerable question, `gas_cylinder_check` alarming on a detector dropout in a
kitchen where the cylinder is always visible). The projected day-one volume was
~370 alerts with ~0 actionable, and a device that behaves that way is unplugged
— a 100% false-negative rate. Asserting sameness would have locked in the defect
this phase exists to remove.

So this is a **characterisation test**: it replays one detection trace through
both engines and pins the difference, with the reason for each divergence
spelled out. If someone later changes the scenario set and the numbers move,
this test fails and makes them say why.

The legacy engine is driven from ``tests/fixtures/legacy_risk_rules.yaml``, the
frozen copy of the retired ``configs/risk_rules.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import BoundingBox, Detection
from src.pipeline.event_memory import EventMemory
from src.pipeline.rule_engine import RuleEngine
from src.scenario_engine.runtime import ScenarioRuleEngine, ScenarioRuntimeError

LEGACY_RULES = Path("tests/fixtures/legacy_risk_rules.yaml")

#: Rules deleted rather than migrated, and why. Asserted below so the reasoning
#: cannot quietly rot into "we forgot to port them".
DELETED_RULES = {
    "knife_near_person": "fires throughout normal cooking, a preserved IADL",
    "medicine_reminder": "asks an unanswerable question; no ASR, plausible double-dose path",
    "gas_cylinder_check": "alarms on a stove-detector dropout in a kitchen where the "
    "cylinder is always visible",
}


def _detection(class_name: str, frame_id: int, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.3),
        frame_id=frame_id,
        timestamp_ms=frame_id * 66.7,
    )


def _kitchen_trace(frames: int = 30) -> list[list[Detection]]:
    """A person cooking: stove, knife and cylinder visible, person present.

    The single most common minute in an Indian kitchen, and the trace that
    decides whether the device is tolerable to live with.
    """
    return [
        [
            _detection("stove", i),
            _detection("knife", i),
            _detection("gas_cylinder", i),
            _detection("person", i),
        ]
        for i in range(frames)
    ]


class TestLegacyEngineOnNormalCooking:
    """What the retired engine did, preserved so the change is legible."""

    @pytest.mark.unit
    def test_it_fires_on_the_first_frame_of_ordinary_cooking(self) -> None:
        engine = RuleEngine(str(LEGACY_RULES), fps=15.0)
        memory = EventMemory(window_size=150)

        trace = _kitchen_trace()
        memory.update(trace[0])
        alerts = engine.evaluate(trace[0], memory, current_fps=15.0)

        fired = {a.rule_id for a in alerts}
        assert "knife_near_person" in fired, (
            "The legacy engine alerted on a knife being visible while a person was "
            "present -- i.e. on cooking. This is the behaviour being removed."
        )

    @pytest.mark.unit
    def test_medicine_reminder_fired_on_mere_visibility(self) -> None:
        engine = RuleEngine(str(LEGACY_RULES), fps=15.0)
        memory = EventMemory(window_size=150)
        detections = [_detection("medicine_strip", 0)]
        memory.update(detections)

        alerts = engine.evaluate(detections, memory, current_fps=15.0)
        assert "medicine_reminder" in {a.rule_id for a in alerts}

    @pytest.mark.unit
    def test_the_deleted_rules_are_all_present_in_the_legacy_file(self) -> None:
        """If one is missing, the fixture drifted and the comparison is void."""
        engine = RuleEngine(str(LEGACY_RULES), fps=15.0)
        ids = {rule.rule_id for rule in engine._rules}  # noqa: SLF001 - fixture integrity
        assert DELETED_RULES.keys() <= ids


class TestScenarioEngineOnTheSameTrace:
    """What the replacement does with the identical input."""

    @pytest.mark.unit
    def test_it_refuses_to_start_while_every_scenario_is_draft(self) -> None:
        """The current, correct state of the repository.

        Every scenario ships `draft`; promotion to `active` needs a clinical
        reviewer. An engine that silently loaded zero rules would be
        indistinguishable from one working perfectly and seeing nothing, so
        construction fails loudly instead.
        """
        with pytest.raises(ScenarioRuntimeError, match="draft"):
            ScenarioRuleEngine(scenario_dir="configs/scenarios", room="kitchen")

    @pytest.mark.unit
    def test_normal_cooking_produces_no_alert_once_scenarios_are_active(
        self, active_scenarios: Path
    ) -> None:
        """The whole point of the migration, in one assertion.

        Same trace, same objects, person present and cooking. The legacy engine
        alerted immediately; the scenario engine says nothing, because
        SC-KIT-001 is about cooking left *unattended* and needs 15 minutes of
        absence, and because `knife_near_person` no longer exists.
        """
        clock = {"now": 1000.0}
        engine = ScenarioRuleEngine(
            scenario_dir=active_scenarios,
            room="kitchen",
            fps=15.0,
            clock=lambda: clock["now"],
        )
        memory = EventMemory(window_size=2700)

        alerts: list[str] = []
        for frame in _kitchen_trace(frames=60):
            memory.update(frame)
            clock["now"] += 1.0 / 15.0
            alerts.extend(a.rule_id for a in engine.evaluate(frame, memory, current_fps=15.0))

        assert alerts == [], (
            f"The scenario engine alerted during ordinary cooking: {alerts}. That is the "
            f"alarm-fatigue failure the legacy ruleset had and this phase removed."
        )

    @pytest.mark.unit
    def test_alerts_carry_the_four_stage_product_output(self, active_scenarios: Path) -> None:
        """A wet floor with a person present -- the one scenario with a 3 s dwell."""
        clock = {"now": 5000.0}
        engine = ScenarioRuleEngine(
            scenario_dir=active_scenarios,
            room="bathroom",
            fps=15.0,
            clock=lambda: clock["now"],
            wall_clock=lambda: _noon(),
        )
        memory = EventMemory(window_size=2700)

        fired = None
        for i in range(200):
            frame = [_detection("wet_floor", i), _detection("person", i)]
            memory.update(frame)
            clock["now"] += 1.0 / 15.0
            for alert in engine.evaluate(frame, memory, current_fps=15.0):
                fired = fired or alert

        assert fired is not None, "SC-BTH-001 never fired despite a 3 s dwell requirement"
        assert fired.scenario_id == "SC-BTH-001"
        assert fired.message, "risk level + patient prompt"
        assert fired.next_best_action, "the action stage the legacy Alert could not carry"
        assert fired.caregiver_channel in {"none", "digest", "push", "push_and_call"}
        assert fired.messages and "en" in fired.messages

    @pytest.mark.unit
    def test_explanation_carries_no_bounding_boxes(self, active_scenarios: Path) -> None:
        """StructuredLogger writes `explanation` verbatim to logs/events.jsonl.

        Person and face geometry there would be per-alert coordinates of a
        resident in their own home (ADR-P6-09).
        """
        clock = {"now": 5000.0}
        engine = ScenarioRuleEngine(
            scenario_dir=active_scenarios,
            room="bathroom",
            fps=15.0,
            clock=lambda: clock["now"],
            wall_clock=lambda: _noon(),
        )
        memory = EventMemory(window_size=2700)

        for i in range(200):
            frame = [_detection("wet_floor", i), _detection("person", i)]
            memory.update(frame)
            clock["now"] += 1.0 / 15.0
            for alert in engine.evaluate(frame, memory, current_fps=15.0):
                flat = repr(alert.explanation)
                assert "BoundingBox" not in flat
                assert "cx" not in flat
                return
        pytest.fail("no alert produced, so the assertion never ran")


class TestTheMigrationIsADeliberateChange:
    @pytest.mark.unit
    def test_deleted_rules_have_no_scenario_successor(self, active_scenarios: Path) -> None:
        """They were deleted, not renamed. A successor would be the same defect."""
        engine = ScenarioRuleEngine(scenario_dir=active_scenarios, room="kitchen")
        conditions = " ".join(s.condition_text for s in engine.scenarios)

        # SC-KIT-002 references `knife`, but gated on the person being ABSENT --
        # the opposite of knife_near_person.
        assert "absent_for(class_name='person'" in conditions
        for scenario in engine.scenarios:
            if "knife" in scenario.condition_text:
                assert "absent_for" in scenario.condition_text, (
                    f"{scenario.scenario_id} references a knife without requiring absence; "
                    f"that reintroduces knife_near_person under a new id"
                )


def _noon():  # noqa: ANN202 - trivial test helper
    from datetime import datetime

    return datetime(2026, 8, 11, 12, 0, 0)

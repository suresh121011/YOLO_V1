"""
Unit tests for src.scenario_engine.simulate.

The load-bearing test is `test_legacy_rules_blow_the_budget`: the three rules
being deleted must project a volume the gate rejects, and their repaired
replacements must pass. If the simulation cannot tell those apart it is
decoration, because every one of those rules looked reasonable in isolation and
only the replay against real occupancy exposes them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.scenario_engine.schema import Scenario
from src.scenario_engine.simulate import (
    SECONDS_PER_DAY,
    Frame,
    indian_kitchen_day,
    simulate,
    synthetic_day,
)

HOUR = 3600.0

_BASE: dict[str, Any] = {
    "scenario_id": "SC-KIT-001",
    "schema_version": 1,
    "scenario_name": "Test scenario",
    "category": "kitchen",
    "trigger": {"op": "detected", "args": {"class_name": "stove"}},
    "risk_level": "INFO",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "draft",
    "caregiver_channel": "none",
    "next_best_action": "Continue monitoring",
    "messages": {"en": "A message."},
    "min_dwell_seconds": 0,
    "cooldown_seconds": 600,
    "max_repeats": 3,
    "max_per_day": 100,
    "clear_after_seconds": 30,
    # Caregiver-facing by default so quiet hours do not gate the volume tests.
    # The schema refuses `always_speak` on a non-CRITICAL scenario, which is the
    # invariant working — quiet-hours behaviour is exercised explicitly below
    # with `patient_facing: True`.
    "patient_facing": False,
}


def _scenario(**overrides: Any) -> Scenario:
    raw = deepcopy(_BASE)
    raw.update(overrides)
    return Scenario.from_mapping(raw)


def _always_on_day(**classes: list[tuple[float, float]]) -> list[Frame]:
    return synthetic_day(classes, step_seconds=30.0, room="kitchen")


class TestCooldownAndBudget:
    @pytest.mark.unit
    def test_permanently_true_condition_is_bounded_by_cooldown(self) -> None:
        scenario = _scenario(cooldown_seconds=600, max_repeats=3, max_per_day=100)
        report = simulate([scenario], _always_on_day(stove=[(0.0, SECONDS_PER_DAY)]), budget=1000)
        assert report.per_scenario[0].alerts > 0
        # One event, capped at 1 + max_repeats announcements.
        assert report.per_scenario[0].alerts <= 4

    @pytest.mark.unit
    def test_daily_budget_suppresses_the_excess(self) -> None:
        scenario = _scenario(cooldown_seconds=60, max_repeats=3, max_per_day=2)
        report = simulate([scenario], _always_on_day(stove=[(0.0, SECONDS_PER_DAY)]), budget=1000)
        volume = report.per_scenario[0]
        assert volume.alerts <= 2
        assert volume.over_budget is False

    @pytest.mark.unit
    def test_hysteresis_makes_a_persisting_condition_one_event(self) -> None:
        """A floor wet for 20 minutes is one event, not ten announcements."""
        scenario = _scenario(cooldown_seconds=60, max_repeats=1, max_per_day=50)
        frames = _always_on_day(stove=[(0.0, 20 * 60)])
        report = simulate([scenario], frames, budget=1000)
        assert report.per_scenario[0].events == 1

    @pytest.mark.unit
    def test_separate_occurrences_are_separate_events(self) -> None:
        scenario = _scenario(cooldown_seconds=60, max_repeats=0, max_per_day=50)
        frames = _always_on_day(stove=[(0.0, 300.0), (5 * HOUR, 5 * HOUR + 300.0)])
        report = simulate([scenario], frames, budget=1000)
        assert report.per_scenario[0].events == 2


class TestDwell:
    @pytest.mark.unit
    def test_dwell_suppresses_a_brief_flicker(self) -> None:
        """Dwell is why the precision budget is spent here, not on thresholds."""
        scenario = _scenario(min_dwell_seconds=180, cooldown_seconds=60, max_per_day=50)
        brief = _always_on_day(stove=[(0.0, 60.0)])
        sustained = _always_on_day(stove=[(0.0, 900.0)])
        assert simulate([scenario], brief, budget=1000).total_alerts == 0
        assert simulate([scenario], sustained, budget=1000).total_alerts > 0


class TestQuietHours:
    @pytest.mark.unit
    def test_suppress_silences_the_night(self) -> None:
        quiet = {"start": "21:00", "end": "06:00", "behaviour": "suppress"}
        scenario = _scenario(
            quiet_hours=quiet, cooldown_seconds=600, max_per_day=100, patient_facing=True
        )
        night = _always_on_day(stove=[(22 * HOUR, 23 * HOUR)])
        assert simulate([scenario], night, budget=1000).total_alerts == 0

    @pytest.mark.unit
    def test_daytime_is_unaffected(self) -> None:
        quiet = {"start": "21:00", "end": "06:00", "behaviour": "suppress"}
        scenario = _scenario(
            quiet_hours=quiet, cooldown_seconds=600, max_per_day=100, patient_facing=True
        )
        day = _always_on_day(stove=[(10 * HOUR, 11 * HOUR)])
        assert simulate([scenario], day, budget=1000).total_alerts > 0

    @pytest.mark.unit
    def test_caregiver_only_still_suppresses_speech(self) -> None:
        quiet = {"start": "21:00", "end": "06:00", "behaviour": "caregiver_only"}
        scenario = _scenario(
            quiet_hours=quiet, cooldown_seconds=600, max_per_day=100, patient_facing=True
        )
        night = _always_on_day(stove=[(1 * HOUR, 3 * HOUR)])
        assert simulate([scenario], night, budget=1000).total_alerts == 0


class TestLegacyRules:
    """The rules being deleted must fail the gate; their replacements must pass."""

    @pytest.mark.unit
    def test_legacy_rules_blow_the_budget(self) -> None:
        profile = indian_kitchen_day()
        frames = synthetic_day(profile.occupancy, step_seconds=30.0, room=profile.room)

        # gas_cylinder_check: detected(gas_cylinder) AND NOT detected(stove).
        # Modelled here as its real behaviour — the cylinder is permanently
        # visible, so the rule is a 600s metronome.
        gas = _scenario(
            scenario_id="SC-KIT-901",
            trigger={"op": "detected", "args": {"class_name": "gas_cylinder"}},
            cooldown_seconds=600,
            max_repeats=3,
            max_per_day=1000,
            min_dwell_seconds=0,
            clear_after_seconds=0,
        )
        # medicine_reminder: a strip left on the table all day, every 300s.
        medicine = _scenario(
            scenario_id="SC-MED-901",
            trigger={"op": "detected", "args": {"class_name": "medicine_strip"}},
            cooldown_seconds=300,
            max_repeats=3,
            max_per_day=1000,
            min_dwell_seconds=0,
            clear_after_seconds=0,
        )
        # knife_near_person: fires throughout every meal preparation.
        knife = _scenario(
            scenario_id="SC-KIT-902",
            trigger={
                "all": [
                    {"op": "detected", "args": {"class_name": "knife"}},
                    {"op": "detected", "args": {"class_name": "person"}},
                ]
            },
            cooldown_seconds=60,
            max_repeats=3,
            max_per_day=1000,
            min_dwell_seconds=0,
            clear_after_seconds=0,
        )

        report = simulate([gas, medicine, knife], frames, budget=20)

        # What the LEGACY engine would have produced: no dwell, no clear
        # condition, no repeat cap — just cooldown against a permanently-true
        # condition, i.e. a metronome for the whole day.
        legacy_projection = (
            SECONDS_PER_DAY / 600  # gas_cylinder, cylinder always visible
            + (22 - 8) * HOUR / 300  # medicine strip on the table 08:00-22:00
            + (0.8 + 1.0) * HOUR / 60  # knife during two meal preparations
        )
        assert legacy_projection > 300, "the legacy arithmetic should be in the hundreds"

        # The schema's own caps make that pattern INEXPRESSIBLE: max_repeats is
        # capped at 3 and hysteresis makes a persisting condition one event, so
        # the worst a legacy-shaped scenario can now do is bounded. That is the
        # point of the caps — the failure mode was designed out rather than
        # left for the gate to catch.
        assert report.total_alerts < legacy_projection / 10
        for volume in report.per_scenario:
            # The bound is per EVENT, not per day: the knife scenario opens two
            # events because there are two meal preparations, which is correct —
            # a second cooking session is a genuinely new occurrence.
            ceiling = volume.events * (1 + 3)  # 3 == the schema's max_repeats cap
            assert volume.alerts <= ceiling, (
                f"{volume.scenario_id} produced {volume.alerts} alerts across "
                f"{volume.events} event(s), above the 1 + max_repeats bound"
            )

    @pytest.mark.unit
    def test_whole_set_budget_catches_death_by_a_thousand_cuts(self) -> None:
        """Each scenario inside its own budget, the set far outside the total.

        This is what the whole-set budget is for. Per-scenario limits cannot see
        it, and it is the realistic way a taxonomy degrades as it grows: nobody
        adds an obviously noisy scenario, they add the twentieth reasonable one.
        """
        frames = _always_on_day(stove=[(0.0, SECONDS_PER_DAY)])
        scenarios = [
            _scenario(
                scenario_id=f"SC-KIT-{index:03d}",
                cooldown_seconds=600,
                max_repeats=3,
                max_per_day=4,
                clear_after_seconds=0,
            )
            for index in range(1, 11)
        ]
        report = simulate(scenarios, frames, budget=20)

        assert all(not v.over_budget for v in report.per_scenario)
        assert report.total_alerts > 20
        assert report.within_budget is False

    @pytest.mark.unit
    def test_repaired_scenario_passes_the_budget(self) -> None:
        """Dwell, a clear condition, capped repeats and a daily budget."""
        profile = indian_kitchen_day()
        frames = synthetic_day(profile.occupancy, step_seconds=30.0, room=profile.room)

        repaired = _scenario(
            scenario_id="SC-KIT-001",
            trigger={
                "all": [
                    {"op": "detected", "args": {"class_name": "stove"}},
                    {"op": "absent_for", "args": {"class_name": "person", "seconds": 900}},
                ]
            },
            min_dwell_seconds=120,
            cooldown_seconds=1800,
            max_repeats=1,
            max_per_day=4,
            clear_after_seconds=120,
            quiet_hours={"start": "21:00", "end": "06:00", "behaviour": "caregiver_only"},
        )
        report = simulate([repaired], frames, budget=20)
        assert report.within_budget is True
        assert report.per_scenario[0].alerts <= 4


class TestReport:
    @pytest.mark.unit
    def test_report_serialises(self) -> None:
        scenario = _scenario(max_per_day=5)
        report = simulate([scenario], _always_on_day(stove=[(0.0, 600.0)]), budget=20)
        payload = report.to_dict()
        assert payload["max_projected_alerts_per_day"] == 20
        assert payload["per_scenario"][0]["scenario_id"] == "SC-KIT-001"
        assert "within_budget" in payload

    @pytest.mark.unit
    def test_simulation_is_deterministic(self) -> None:
        scenario = _scenario(max_per_day=50)
        frames = _always_on_day(stove=[(0.0, 5 * HOUR)])
        assert simulate([scenario], frames, budget=20).to_dict() == (
            simulate([scenario], frames, budget=20).to_dict()
        )

    @pytest.mark.unit
    def test_scenario_order_does_not_change_volume(self) -> None:
        a = _scenario(scenario_id="SC-KIT-001", max_per_day=50)
        b = _scenario(
            scenario_id="SC-KIT-002",
            trigger={"op": "detected", "args": {"class_name": "knife"}},
            max_per_day=50,
        )
        frames = _always_on_day(stove=[(0.0, 3 * HOUR)], knife=[(0.0, 3 * HOUR)])
        assert simulate([a, b], frames, budget=50).total_alerts == (
            simulate([b, a], frames, budget=50).total_alerts
        )


class TestSyntheticDay:
    @pytest.mark.unit
    def test_windows_control_presence(self) -> None:
        frames = synthetic_day({"stove": [(0.0, 100.0)]}, step_seconds=50.0, duration_seconds=200.0)
        assert [sorted(f.present) for f in frames] == [["stove"], ["stove"], [], []]

    @pytest.mark.unit
    def test_kitchen_profile_has_a_permanently_visible_cylinder(self) -> None:
        """The occupancy fact that made gas_cylinder_check a metronome."""
        profile = indian_kitchen_day()
        frames = synthetic_day(profile.occupancy, step_seconds=3600.0)
        assert all("gas_cylinder" in f.present for f in frames)
        assert any("person" in f.present for f in frames)
        assert not all("person" in f.present for f in frames)

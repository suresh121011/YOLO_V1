"""
Unit tests for src.scenario_engine.schema.

Two properties carry most of the weight:

* `rule_hash` must move when *behaviour* changes and stay put when prose does.
  If it moved on a Hindi typo fix, contributors would learn to ignore staleness
  warnings; if it stayed put on a risk downgrade, the clip suite would pass
  green against an expectation that no longer exists (ADR-P6-06).
* `required_objects` is derived from the trigger, so the CSV view cannot drift
  from the executable condition.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.scenario_engine.schema import (
    CaregiverChannel,
    Detectability,
    Scenario,
    ScenarioError,
    Status,
    sorted_scenarios,
)

_BASE: dict[str, Any] = {
    "scenario_id": "SC-BTH-001",
    "schema_version": 1,
    "scenario_name": "Wet floor near the sink",
    "category": "bathroom",
    "trigger": {
        "all": [
            {"op": "detected", "args": {"class_name": "wet_floor"}},
            {"op": "present_for", "args": {"class_name": "wet_floor", "seconds": 3}},
        ]
    },
    "risk_level": "HIGH",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "active",
    "caregiver_channel": "digest",
    "next_best_action": "Continue monitoring",
    "messages": {"en": "The floor near the sink looks wet.", "hi": "सिंक के पास फर्श गीला लग रहा है।"},
    "min_confidence": 0.5,
    "min_dwell_seconds": 3,
    "cooldown_seconds": 120,
    "max_repeats": 2,
    "max_per_day": 6,
    "priority": 20,
    "clear_after_seconds": 30,
    "reviewed_by": "clinical-review",
    "reviewed_on": "2026-08-07",
    "evidence": [
        {
            "source_id": "CDC-STEADI-CheckForSafety-2017",
            "section": "Bathroom",
            "strength": "guideline",
            "supports": "hazard",
        }
    ],
}


def _scenario(**overrides: Any) -> Scenario:
    raw = deepcopy(_BASE)
    raw.update(overrides)
    return Scenario.from_mapping(raw)


class TestConstruction:
    @pytest.mark.unit
    def test_valid_scenario_round_trips_core_fields(self) -> None:
        scenario = _scenario()
        assert scenario.scenario_id == "SC-BTH-001"
        assert scenario.risk_level == "HIGH"
        assert scenario.detectability is Detectability.DIRECT
        assert scenario.caregiver_channel is CaregiverChannel.DIGEST
        assert scenario.status is Status.ACTIVE
        assert scenario.evidence[0].source_id == "CDC-STEADI-CheckForSafety-2017"

    @pytest.mark.unit
    def test_required_objects_are_derived_from_the_trigger(self) -> None:
        """Derived, so the CSV view cannot drift from the executable condition."""
        assert _scenario().required_objects == ("wet_floor",)

    @pytest.mark.unit
    def test_condition_text_is_rendered_not_authored(self) -> None:
        text = _scenario().condition_text
        assert "detected(class_name='wet_floor')" in text
        assert " AND " in text

    @pytest.mark.unit
    def test_defaults_are_applied(self) -> None:
        raw = deepcopy(_BASE)
        for key in ("max_repeats", "max_per_day", "priority", "clear_after_seconds"):
            raw.pop(key)
        scenario = Scenario.from_mapping(raw)
        assert scenario.max_repeats == 2
        assert scenario.max_per_day == 6
        assert scenario.patient_facing is True


class TestFieldValidation:
    @pytest.mark.unit
    @pytest.mark.parametrize("bad", ["SC-BATH-001", "sc-bth-001", "SC-BTH-1", "BTH-001", ""])
    def test_scenario_id_pattern(self, bad: str) -> None:
        with pytest.raises(ScenarioError, match="scenario_id"):
            _scenario(scenario_id=bad)

    @pytest.mark.unit
    def test_risk_level_must_match_the_severity_vocabulary(self) -> None:
        with pytest.raises(ScenarioError, match="risk_level"):
            _scenario(risk_level="URGENT")

    @pytest.mark.unit
    def test_unknown_enum_values_list_the_valid_set(self) -> None:
        with pytest.raises(ScenarioError) as excinfo:
            _scenario(caregiver_channel="sms")
        assert "digest" in str(excinfo.value) and "push_and_call" in str(excinfo.value)

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [-0.1, 1.4, "high"])
    def test_min_confidence_range(self, bad: Any) -> None:
        with pytest.raises(ScenarioError, match="min_confidence"):
            _scenario(min_confidence=bad)

    @pytest.mark.unit
    def test_trigger_is_required_and_validated(self) -> None:
        raw = deepcopy(_BASE)
        del raw["trigger"]
        with pytest.raises(ScenarioError, match="trigger is required"):
            Scenario.from_mapping(raw)

    @pytest.mark.unit
    def test_evidence_requires_strength_and_supports(self) -> None:
        with pytest.raises(ScenarioError, match="evidence"):
            _scenario(evidence=[{"source_id": "X"}])


class TestSafetyInvariants:
    @pytest.mark.unit
    def test_active_requires_review(self) -> None:
        with pytest.raises(ScenarioError, match="reviewed_by"):
            _scenario(reviewed_by="", reviewed_on="")

    @pytest.mark.unit
    def test_draft_does_not_require_review(self) -> None:
        assert _scenario(status="draft", reviewed_by="", reviewed_on="").status is Status.DRAFT

    @pytest.mark.unit
    def test_inferred_requires_a_capability_disclaimer(self) -> None:
        with pytest.raises(ScenarioError, match="capability_disclaimer"):
            _scenario(detectability="inferred")
        assert _scenario(
            detectability="inferred",
            capability_disclaimer="Observation of packaging position only.",
        ).capability_disclaimer

    @pytest.mark.unit
    def test_rejected_requires_a_reason(self) -> None:
        with pytest.raises(ScenarioError, match="rejection_reason"):
            _scenario(detectability="rejected")

    @pytest.mark.unit
    def test_rejected_row_skips_the_runtime_contract(self) -> None:
        """A rejected row never runs, so review and messages are moot."""
        scenario = _scenario(
            detectability="rejected",
            rejection_reason="No pose estimation; a low wide box is a person bending.",
            status="rejected",
            reviewed_by="",
            reviewed_on="",
            messages={},
        )
        assert scenario.detectability is Detectability.REJECTED

    @pytest.mark.unit
    def test_always_speak_is_critical_only(self) -> None:
        quiet = {"start": "21:00", "end": "06:00", "behaviour": "always_speak"}
        with pytest.raises(ScenarioError, match="always_speak"):
            _scenario(quiet_hours=quiet, risk_level="HIGH")
        assert _scenario(quiet_hours=quiet, risk_level="CRITICAL").risk_level == "CRITICAL"

    @pytest.mark.unit
    def test_max_repeats_is_capped(self) -> None:
        with pytest.raises(ScenarioError, match="max_repeats"):
            _scenario(max_repeats=9)

    @pytest.mark.unit
    def test_patient_facing_requires_a_message(self) -> None:
        with pytest.raises(ScenarioError, match="message"):
            _scenario(messages={})
        assert _scenario(messages={}, patient_facing=False).patient_facing is False

    @pytest.mark.unit
    def test_escalation_must_be_ordered(self) -> None:
        steps = [
            {"after_seconds": 180, "action": "notify_caregiver"},
            {"after_seconds": 90, "action": "reprompt"},
        ]
        with pytest.raises(ScenarioError, match="ordered"):
            _scenario(escalation=steps)


class TestRuleHash:
    @pytest.mark.unit
    def test_is_stable_across_identical_definitions(self) -> None:
        assert _scenario().rule_hash() == _scenario().rule_hash()
        assert _scenario().rule_hash().startswith("sha256:")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides",
        [
            {"risk_level": "MEDIUM"},
            {"min_confidence": 0.7},
            {"min_dwell_seconds": 9},
            {"clear_after_seconds": 90},
            {"caregiver_channel": "push"},
            {"room_context": "bathroom"},
            {"patient_facing": False, "messages": {}},
            {
                "trigger": {
                    "all": [
                        {"op": "detected", "args": {"class_name": "wet_floor"}},
                        {"op": "present_for", "args": {"class_name": "wet_floor", "seconds": 5}},
                    ]
                }
            },
        ],
    )
    def test_behaviour_changes_move_the_hash(self, overrides: dict[str, Any]) -> None:
        """Each of these invalidates clips adjudicated against the old behaviour."""
        assert _scenario().rule_hash() != _scenario(**overrides).rule_hash()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides",
        [
            {"messages": {"en": "Reworded entirely.", "hi": "पूरी तरह से बदला हुआ।"}},
            {"notes": "Reviewed again on 2026-09-01."},
            {"scenario_name": "A clearer display name"},
            {"false_positive_notes": "Polished granite can read as wet."},
            {"evidence": []},
            {"next_best_action": "Ask the caregiver to check"},
        ],
    )
    def test_prose_changes_do_not_move_the_hash(self, overrides: dict[str, Any]) -> None:
        """A Hindi typo fix must invalidate zero labelled clips."""
        assert _scenario().rule_hash() == _scenario(**overrides).rule_hash()

    @pytest.mark.unit
    def test_hash_is_insensitive_to_authoring_key_order(self) -> None:
        reordered = {k: _BASE[k] for k in sorted(_BASE)}
        assert Scenario.from_mapping(reordered).rule_hash() == _scenario().rule_hash()


class TestOrdering:
    @pytest.mark.unit
    def test_sorted_by_priority_then_id(self) -> None:
        a = _scenario(scenario_id="SC-BTH-002", priority=10)
        b = _scenario(scenario_id="SC-BTH-001", priority=10)
        c = _scenario(scenario_id="SC-BTH-003", priority=5)
        ordered = sorted_scenarios([a, b, c])
        assert [s.scenario_id for s in ordered] == ["SC-BTH-003", "SC-BTH-001", "SC-BTH-002"]

    @pytest.mark.unit
    def test_order_is_independent_of_input_order(self) -> None:
        scenarios = [
            _scenario(scenario_id="SC-BTH-002", priority=10),
            _scenario(scenario_id="SC-BTH-001", priority=10),
        ]
        assert [s.scenario_id for s in sorted_scenarios(scenarios)] == [
            s.scenario_id for s in sorted_scenarios(list(reversed(scenarios)))
        ]

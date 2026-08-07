"""
Unit tests for src.scenario_engine.validators.

Each message-lint test names the real string it would have caught from
configs/risk_rules.yaml, so the checks stay tied to observed failures rather
than hypothetical ones.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.scenario_engine.schema import Scenario
from src.scenario_engine.validators import (
    ERROR,
    WARN,
    Finding,
    check_confidence_reachable,
    check_conflicting_arbitration,
    check_determinism,
    check_locale_completeness,
    check_message_length,
    check_regulated_claims,
    check_safety_class_coverage,
    check_startle_language,
    check_temporal_within_memory,
    check_unanswerable_questions,
    check_unobservable_assertions,
    errors,
    run_all,
    warnings,
)

_BASE: dict[str, Any] = {
    "scenario_id": "SC-KIT-001",
    "schema_version": 1,
    "scenario_name": "Cooking left unattended",
    "category": "kitchen",
    "trigger": {
        "all": [
            {"op": "detected", "args": {"class_name": "stove"}},
            {"op": "absent_for", "args": {"class_name": "person", "seconds": 120}},
        ]
    },
    "risk_level": "HIGH",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "draft",
    "caregiver_channel": "digest",
    "next_best_action": "Continue monitoring",
    "messages": {"en": "The stove is visible.", "hi": "चूल्हा दिख रहा है।"},
    "min_confidence": 0.5,
    "min_dwell_seconds": 3,
}


def _scenario(**overrides: Any) -> Scenario:
    raw = deepcopy(_BASE)
    raw.update(overrides)
    return Scenario.from_mapping(raw)


def _codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


class TestUnobservableAssertions:
    @pytest.mark.unit
    def test_catches_the_stove_is_on_claim(self) -> None:
        """risk_rules.yaml:54 — 'The stove appears to be on'."""
        scenario = _scenario(
            messages={"en": "The stove appears to be on without anyone nearby.", "hi": "चूल्हा।"}
        )
        findings = check_unobservable_assertions([scenario])
        assert _codes(findings) == {"unobservable-assertion"}
        assert findings[0].severity == ERROR

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "The tap is running.",
            "She has fallen in the bathroom.",
            "Confirm you have taken the tablets.",
            "The light was left on.",
        ],
    )
    def test_catches_other_state_claims(self, text: str) -> None:
        assert check_unobservable_assertions([_scenario(messages={"en": text})])

    @pytest.mark.unit
    def test_appearance_wording_passes(self) -> None:
        ok = _scenario(messages={"en": "The floor near the sink looks wet.", "hi": "फर्श।"})
        assert check_unobservable_assertions([ok]) == []

    @pytest.mark.unit
    def test_checks_escalation_messages_too(self) -> None:
        scenario = _scenario(
            escalation=[
                {"after_seconds": 90, "action": "reprompt", "messages": {"en": "The stove is on."}}
            ]
        )
        assert check_unobservable_assertions([scenario])


class TestUnanswerableQuestions:
    @pytest.mark.unit
    def test_catches_the_medicine_question(self) -> None:
        """risk_rules.yaml:63 — 'Have you taken your medication today?'"""
        scenario = _scenario(messages={"en": "Have you taken your medicine today?"})
        findings = check_unanswerable_questions([scenario])
        assert _codes(findings) == {"unanswerable-question"}

    @pytest.mark.unit
    def test_statements_pass(self) -> None:
        assert check_unanswerable_questions([_scenario()]) == []

    @pytest.mark.unit
    def test_caregiver_only_scenarios_are_exempt(self) -> None:
        scenario = _scenario(patient_facing=False, messages={"en": "Was the regulator closed?"})
        assert check_unanswerable_questions([scenario]) == []


class TestStartleLanguage:
    @pytest.mark.unit
    def test_catches_please_be_careful(self) -> None:
        """risk_rules.yaml:31 — 'Please be careful. There is a knife nearby.'"""
        scenario = _scenario(messages={"en": "Please be careful. There is a knife nearby."})
        findings = check_startle_language([scenario])
        assert _codes(findings) == {"startle-language"}
        assert findings[0].severity == ERROR

    @pytest.mark.unit
    def test_calm_observation_passes(self) -> None:
        assert check_startle_language([_scenario()]) == []


class TestRegulatedClaims:
    @pytest.mark.unit
    def test_warns_on_medical_device_verbs(self) -> None:
        scenario = _scenario(messages={"en": "This will prevent a fall."})
        findings = check_regulated_claims([scenario])
        assert _codes(findings) == {"regulated-claim"}
        assert findings[0].severity == WARN


class TestMessageLength:
    @pytest.mark.unit
    def test_warns_on_a_long_prompt(self) -> None:
        long_text = " ".join(["word"] * 25)
        assert _codes(check_message_length([_scenario(messages={"en": long_text})])) == {
            "message-too-long"
        }

    @pytest.mark.unit
    def test_short_prompt_passes(self) -> None:
        assert check_message_length([_scenario()]) == []


class TestLocaleCompleteness:
    @pytest.mark.unit
    def test_missing_hindi_warns(self) -> None:
        assert "missing-locale" in _codes(
            check_locale_completeness([_scenario(messages={"en": "The stove is visible."})])
        )

    @pytest.mark.unit
    def test_english_pasted_into_the_hindi_field_is_an_error(self) -> None:
        scenario = _scenario(
            messages={"en": "The stove is visible.", "hi": "The stove is visible."}
        )
        findings = check_locale_completeness([scenario])
        assert "locale-not-translated" in _codes(findings)
        assert [f.severity for f in findings if f.code == "locale-not-translated"] == [ERROR]

    @pytest.mark.unit
    def test_real_devanagari_passes(self) -> None:
        assert check_locale_completeness([_scenario()]) == []


class TestReachability:
    @pytest.mark.unit
    def test_confidence_below_the_detector_floor_warns(self) -> None:
        findings = check_confidence_reachable(
            [_scenario(min_confidence=0.1)], {"stove": 0.22}, global_floor=0.25
        )
        assert _codes(findings) == {"confidence-unreachable"}

    @pytest.mark.unit
    def test_confidence_above_the_floor_passes(self) -> None:
        assert check_confidence_reachable([_scenario()], {"stove": 0.22}, 0.25) == []

    @pytest.mark.unit
    def test_dwell_beyond_the_memory_window_is_an_error(self) -> None:
        """150 frames at 15 FPS is a hard 10.0s ceiling."""
        findings = check_temporal_within_memory([_scenario(min_dwell_seconds=30)], 150, 15.0)
        assert _codes(findings) == {"dwell-exceeds-memory"}
        assert findings[0].severity == ERROR

    @pytest.mark.unit
    def test_dwell_within_the_window_passes(self) -> None:
        assert check_temporal_within_memory([_scenario(min_dwell_seconds=3)], 150, 15.0) == []


class TestCoverage:
    @pytest.mark.unit
    def test_uncovered_safety_class_warns(self) -> None:
        active = _scenario(status="active", reviewed_by="r", reviewed_on="2026-08-07")
        findings = check_safety_class_coverage([active], ["stove", "wet_floor", "knife"])
        uncovered = {f.message.split("'")[1] for f in findings}
        assert uncovered == {"wet_floor", "knife"}

    @pytest.mark.unit
    def test_draft_scenarios_do_not_count_as_coverage(self) -> None:
        assert check_safety_class_coverage([_scenario(status="draft")], ["stove"])


class TestDeterminism:
    @pytest.mark.unit
    def test_a_well_formed_set_is_deterministic(self) -> None:
        scenarios = [
            _scenario(scenario_id="SC-KIT-001", priority=10),
            _scenario(
                scenario_id="SC-KIT-002",
                priority=20,
                trigger={"op": "absent_for", "args": {"class_name": "knife", "seconds": 5}},
            ),
        ]
        assert check_determinism(scenarios) == []

    @pytest.mark.unit
    def test_ties_are_broken_by_id_not_by_file_order(self) -> None:
        """Same priority, same trigger — order must still be stable."""
        scenarios = [
            _scenario(scenario_id="SC-KIT-002", priority=10),
            _scenario(scenario_id="SC-KIT-001", priority=10),
        ]
        assert check_determinism(scenarios) == []


class TestArbitration:
    @pytest.mark.unit
    def test_same_priority_overlapping_classes_different_risk_warns(self) -> None:
        a = _scenario(scenario_id="SC-KIT-001", priority=10, risk_level="HIGH")
        b = _scenario(scenario_id="SC-KIT-002", priority=10, risk_level="INFO")
        findings = check_conflicting_arbitration([a, b])
        assert _codes(findings) == {"ambiguous-arbitration"}

    @pytest.mark.unit
    def test_distinct_priorities_pass(self) -> None:
        a = _scenario(scenario_id="SC-KIT-001", priority=10, risk_level="HIGH")
        b = _scenario(scenario_id="SC-KIT-002", priority=20, risk_level="INFO")
        assert check_conflicting_arbitration([a, b]) == []


class TestRunAll:
    @pytest.mark.unit
    def test_clean_scenario_produces_no_errors(self) -> None:
        findings = run_all([_scenario()], class_thresholds={"stove": 0.22})
        assert errors(findings) == []

    @pytest.mark.unit
    def test_aggregates_across_validators(self) -> None:
        bad = _scenario(
            messages={"en": "Please be careful, the stove is on. Have you checked?"},
            min_dwell_seconds=30,
        )
        findings = run_all([bad], class_thresholds={"stove": 0.22})
        codes = _codes(findings)
        assert {"startle-language", "unobservable-assertion", "unanswerable-question"} <= codes
        assert len(errors(findings)) >= 3

    @pytest.mark.unit
    def test_severity_split_is_reported(self) -> None:
        findings = run_all([_scenario(min_confidence=0.05)], class_thresholds={"stove": 0.22})
        assert warnings(findings)
        assert all(f.severity == WARN for f in warnings(findings))

    @pytest.mark.unit
    def test_repo_scenarios_have_no_errors(self) -> None:
        """The committed scenario set must stay clean."""
        from src.scenario_engine.compile import load_scenario_files

        scenarios = load_scenario_files("configs/scenarios")
        findings = run_all(scenarios, safety_classes=["wet_floor"])
        assert errors(findings) == [], [f.to_dict() for f in errors(findings)]

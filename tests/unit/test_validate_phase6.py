"""
Unit tests for scripts/qa/validate_phase6.py — the M6 correctness gate.

The point of these is falsifiability. This repository has a recorded incident
where release gate RG6 read green for an entire cycle while being structurally
incapable of failing, certifying a release over 21,964 un-pushed objects
(`src/dataset/release/gates.py:394-421`). A gate that cannot fail is worse than
no gate, because it converts an unchecked property into a documented assurance.

So each gate is exercised against a *deliberately broken* input as well as the
real repository.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_GATE_PATH = Path("scripts/qa/validate_phase6.py")


def _load() -> Any:
    """Import the numbered-script module, whose name is not a valid identifier.

    The module must be registered in ``sys.modules`` *before* execution: it
    defines a ``@dataclass``, and dataclasses resolve annotations by looking
    their own module up there.
    """
    spec = importlib.util.spec_from_file_location("validate_phase6", _GATE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_phase6"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> Any:
    return _load()


class TestGatesAreFalsifiable:
    @pytest.mark.unit
    def test_validator_gate_fails_on_a_bad_scenario(self, gate: Any) -> None:
        """G2 must fail when a scenario asserts unobservable state."""
        from src.scenario_engine.schema import Scenario

        bad = Scenario.from_mapping(
            {
                "scenario_id": "SC-KIT-999",
                "schema_version": 1,
                "scenario_name": "Broken",
                "category": "kitchen",
                "trigger": {"op": "absent_for", "args": {"class_name": "person", "seconds": 30}},
                "risk_level": "HIGH",
                "detectability": "direct",
                "claim_class": "observation",
                "status": "draft",
                "caregiver_channel": "digest",
                "next_best_action": "x",
                "messages": {"en": "The stove is on.", "hi": "चूल्हा।"},
            }
        )
        result = gate.gate_g2_no_validator_errors([bad], {"memory_window_frames": 2700})
        assert result.status == "fail"
        assert result.details["errors"]

    @pytest.mark.unit
    def test_volume_gate_fails_on_an_over_budget_set(self, gate: Any) -> None:
        """G3 must fail when the projected volume exceeds the budget."""
        from src.scenario_engine.schema import Scenario

        noisy = [
            Scenario.from_mapping(
                {
                    "scenario_id": f"SC-KIT-{i:03d}",
                    "schema_version": 1,
                    "scenario_name": "Noisy",
                    "category": "kitchen",
                    "trigger": {
                        "op": "present_for",
                        "args": {"class_name": "gas_cylinder", "seconds": 30},
                    },
                    "risk_level": "INFO",
                    "detectability": "direct",
                    "claim_class": "observation",
                    "status": "draft",
                    "caregiver_channel": "none",
                    "next_best_action": "x",
                    "messages": {},
                    "patient_facing": False,
                    "min_dwell_seconds": 30,
                    "cooldown_seconds": 600,
                    "max_repeats": 3,
                    "max_per_day": 4,
                    "clear_after_seconds": 0,
                }
            )
            for i in range(1, 16)
        ]
        result = gate.gate_g3_volume_within_budget(noisy)
        assert result.status == "fail"
        assert result.details["total_alerts"] > result.details["budget"]

    @pytest.mark.unit
    def test_determinism_gate_passes_on_the_real_set(self, gate: Any) -> None:
        from src.scenario_engine.compile import load_scenario_files

        result = gate.gate_g4_determinism(load_scenario_files("configs/scenarios"))
        assert result.status == "pass"
        assert result.details["scenarios_checked"] >= 1


class TestCommittedEvidence:
    @pytest.mark.unit
    def test_report_exists_and_passed(self) -> None:
        """The committed M6 evidence must record a PASS."""
        report = json.loads(
            Path("data/qa_reports/phase6_validation_report.json").read_text(encoding="utf-8")
        )
        assert report["verdict"] == "PASS"
        assert report["milestone"] == "M6"
        assert [g["gate_id"] for g in report["gates"]] == [f"G{i}" for i in range(1, 9)]
        assert all(g["status"] == "pass" for g in report["gates"])

    @pytest.mark.unit
    def test_report_states_its_limitations(self) -> None:
        """A PASS that hides what it did not check is a false assurance."""
        report = json.loads(
            Path("data/qa_reports/phase6_validation_report.json").read_text(encoding="utf-8")
        )
        limitations = " ".join(report["known_limitations"]).lower()
        assert "draft" in limitations and "clinical review" in limitations
        assert "clips" in limitations
        assert "risk_rules.yaml" in limitations

    @pytest.mark.unit
    def test_report_records_the_unusable_class(self) -> None:
        report = json.loads(
            Path("data/qa_reports/phase6_validation_report.json").read_text(encoding="utf-8")
        )
        assert "passport" in report["unusable_classes"]

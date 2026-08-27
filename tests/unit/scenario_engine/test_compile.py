"""
Unit tests for src.scenario_engine.taxonomy and .compile.

The checks that carry weight here are the ones that catch a scenario which
would compile green and never fire:

* a class that is demoted to scene-level or disabled by a feature flag — the
  taxonomy fingerprint provably cannot see either (ADR-P6-05);
* a trigger satisfiable by object presence alone;
* an artifact edited by hand or truncated, which the legacy loader would have
  accepted while logging "Loaded 0 rules" at INFO.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.scenario_engine.compile import (
    CompileError,
    build_inverted_index,
    compile_scenarios,
    load_scenario_files,
    verify_artifact,
    write_artifact,
)
from src.scenario_engine.taxonomy import (
    CapabilityMap,
    ClassCapability,
    TaxonomyError,
    build_capability_map,
    taxonomy_fingerprint,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

_NAMES = {0: "person", 5: "knife", 6: "stove", 8: "passport", 20: "wet_floor"}


def _capability(**overrides: ClassCapability) -> CapabilityMap:
    classes = {
        name: ClassCapability(name=name, class_id=cid, detection_level="bbox", enabled=True)
        for cid, name in _NAMES.items()
    }
    classes.update(overrides)
    return CapabilityMap(fingerprint=taxonomy_fingerprint(23, _NAMES), classes=classes)


_SCENARIO: dict[str, Any] = {
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
    "risk_level": "CRITICAL",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "active",
    "caregiver_channel": "push",
    "next_best_action": "Ask someone to check the kitchen",
    "messages": {"en": "The stove is visible and nobody has been nearby for a while."},
    "reviewed_by": "clinical-review",
    "reviewed_on": "2026-08-07",
    "evidence": [
        {"source_id": "NFPA-CookingSafety", "strength": "guideline", "supports": "hazard"}
    ],
}


def _write(tmp_path: Path, *scenarios: dict[str, Any]) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir(exist_ok=True)
    for scenario in scenarios:
        path = directory / f"{scenario['scenario_id']}.yaml"
        path.write_text(yaml.safe_dump(scenario, allow_unicode=True), encoding="utf-8")
    return directory


def _variant(**overrides: Any) -> dict[str, Any]:
    raw = deepcopy(_SCENARIO)
    raw.update(overrides)
    return raw


# ─── Taxonomy ─────────────────────────────────────────────────────────────────


class TestTaxonomyFingerprint:
    @pytest.mark.unit
    def test_matches_the_dataset_implementation_byte_for_byte(self) -> None:
        """Deliberately duplicated across a layering boundary — pin the drift."""
        from src.dataset.completeness import taxonomy_fingerprint as dataset_impl

        assert taxonomy_fingerprint(23, _NAMES) == dataset_impl(23, _NAMES)

    @pytest.mark.unit
    def test_changes_when_a_name_changes(self) -> None:
        renamed = {**_NAMES, 5: "blade"}
        assert taxonomy_fingerprint(23, _NAMES) != taxonomy_fingerprint(23, renamed)


class TestCapabilityMap:
    @pytest.mark.unit
    def test_built_from_the_live_config(self) -> None:
        capability = build_capability_map()
        assert "wet_floor" in capability.classes
        assert capability.classes["person"].usable is True

    @pytest.mark.unit
    def test_feature_flag_disables_a_class(self) -> None:
        """`passport: false  # privacy` must make the class unusable."""
        capability = build_capability_map()
        passport = capability.classes["passport"]
        assert passport.enabled is False
        assert passport.usable is False

    @pytest.mark.unit
    def test_demotion_is_read_from_the_decision_artifact(self, tmp_path: Path) -> None:
        decision = tmp_path / "wet_floor_decision.json"
        decision.write_text(
            json.dumps({"decision": "demote", "reason": "IAA 0.54 < 0.60"}), encoding="utf-8"
        )
        capability = build_capability_map(wet_floor_decision_path=decision)
        wet_floor = capability.classes["wet_floor"]
        assert wet_floor.detection_level == "scene"
        assert wet_floor.usable is False

    @pytest.mark.unit
    def test_fingerprint_is_blind_to_the_demotion(self, tmp_path: Path) -> None:
        """The whole reason the capability map exists (ADR-P6-05)."""
        decision = tmp_path / "wet_floor_decision.json"
        decision.write_text(json.dumps({"decision": "demote"}), encoding="utf-8")
        before = build_capability_map()
        after = build_capability_map(wet_floor_decision_path=decision)
        assert before.fingerprint == after.fingerprint
        assert before.classes["wet_floor"].usable != after.classes["wet_floor"].usable

    @pytest.mark.unit
    def test_malformed_decision_artifact_raises(self, tmp_path: Path) -> None:
        decision = tmp_path / "wet_floor_decision.json"
        decision.write_text(json.dumps({"decision": "maybe"}), encoding="utf-8")
        with pytest.raises(TaxonomyError, match="keep"):
            build_capability_map(wet_floor_decision_path=decision)


# ─── Loading ──────────────────────────────────────────────────────────────────


class TestLoading:
    @pytest.mark.unit
    def test_missing_directory_explains_the_placement(self, tmp_path: Path) -> None:
        with pytest.raises(CompileError, match="configs/"):
            load_scenario_files(tmp_path / "nope")

    @pytest.mark.unit
    def test_row_errors_name_the_file(self, tmp_path: Path) -> None:
        directory = _write(tmp_path, _variant(risk_level="URGENT"))
        with pytest.raises(CompileError, match="SC-KIT-001.yaml"):
            load_scenario_files(directory)

    @pytest.mark.unit
    def test_invalid_yaml_names_the_file(self, tmp_path: Path) -> None:
        directory = tmp_path / "scenarios"
        directory.mkdir()
        (directory / "SC-KIT-009.yaml").write_text("a: [unclosed\n", encoding="utf-8")
        with pytest.raises(CompileError, match="SC-KIT-009"):
            load_scenario_files(directory)


# ─── Set-level validation ─────────────────────────────────────────────────────


class TestSetValidation:
    @pytest.mark.unit
    def test_happy_path(self, tmp_path: Path) -> None:
        directory = _write(tmp_path, _SCENARIO)
        result = compile_scenarios(directory, capability=_capability())
        assert result.artifact["rule_count"] == 1
        assert result.artifact["scenarios"][0]["scenario_id"] == "SC-KIT-001"

    @pytest.mark.unit
    def test_unknown_class_is_rejected(self, tmp_path: Path) -> None:
        bad = _variant(trigger={"op": "detected", "args": {"class_name": "toaster"}})
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="toaster"):
            compile_scenarios(directory, capability=_capability())

    @pytest.mark.unit
    def test_disabled_class_is_rejected(self, tmp_path: Path) -> None:
        capability = _capability(
            passport=ClassCapability("passport", 8, "bbox", False, "disabled for privacy")
        )
        bad = _variant(
            trigger={
                "all": [
                    {"op": "detected", "args": {"class_name": "passport"}},
                    {"op": "absent_for", "args": {"class_name": "person", "seconds": 60}},
                ]
            }
        )
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="passport"):
            compile_scenarios(directory, capability=capability)

    @pytest.mark.unit
    def test_demoted_class_is_rejected(self, tmp_path: Path) -> None:
        """A scene-level class cannot satisfy a bbox trigger."""
        capability = _capability(
            wet_floor=ClassCapability("wet_floor", 20, "scene", True, "R24 demotion")
        )
        bad = _variant(
            trigger={
                "all": [
                    {"op": "detected", "args": {"class_name": "wet_floor"}},
                    {"op": "present_for", "args": {"class_name": "wet_floor", "seconds": 3}},
                ]
            }
        )
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="wet_floor"):
            compile_scenarios(directory, capability=capability)

    @pytest.mark.unit
    def test_static_only_trigger_is_rejected(self, tmp_path: Path) -> None:
        """knife_near_person — the cooking metronome."""
        bad = _variant(
            trigger={
                "all": [
                    {"op": "detected", "args": {"class_name": "knife"}},
                    {"op": "detected", "args": {"class_name": "person"}},
                ]
            }
        )
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="presence alone"):
            compile_scenarios(directory, capability=_capability())

    @pytest.mark.unit
    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        directory = tmp_path / "scenarios"
        directory.mkdir()
        for filename in ("SC-KIT-001.yaml", "SC-KIT-001-copy.yaml"):
            (directory / filename).write_text(yaml.safe_dump(_SCENARIO), encoding="utf-8")
        with pytest.raises(CompileError, match="Duplicate"):
            compile_scenarios(directory, capability=_capability())

    @pytest.mark.unit
    def test_uncited_scenario_may_not_alarm(self, tmp_path: Path) -> None:
        """If it cannot be cited, it cannot alarm."""
        directory = _write(tmp_path, _variant(evidence=[]))
        with pytest.raises(CompileError, match="evidence"):
            compile_scenarios(directory, capability=_capability())

    @pytest.mark.unit
    def test_strength_none_does_not_count_as_evidence(self, tmp_path: Path) -> None:
        bad = _variant(evidence=[{"source_id": "hunch", "strength": "none", "supports": "hazard"}])
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="evidence"):
            compile_scenarios(directory, capability=_capability())

    @pytest.mark.unit
    def test_uncited_low_risk_digest_scenario_is_allowed(self, tmp_path: Path) -> None:
        ok = _variant(
            risk_level="INFO", caregiver_channel="digest", evidence=[], scenario_id="SC-KIT-002"
        )
        directory = _write(tmp_path, ok)
        assert compile_scenarios(directory, capability=_capability()).artifact["rule_count"] == 1

    @pytest.mark.unit
    def test_rejected_row_is_kept_but_not_runnable(self, tmp_path: Path) -> None:
        rejected = _variant(
            scenario_id="SC-SAF-001",
            detectability="rejected",
            status="rejected",
            rejection_reason="No pose estimation; a low wide box is a person bending.",
            trigger={"op": "detected", "args": {"class_name": "person"}},
            evidence=[],
        )
        directory = _write(tmp_path, _SCENARIO, rejected)
        artifact = compile_scenarios(directory, capability=_capability()).artifact
        assert artifact["rule_count"] == 1
        assert [r["scenario_id"] for r in artifact["rejected"]] == ["SC-SAF-001"]

    @pytest.mark.unit
    def test_rejected_row_may_not_be_active(self, tmp_path: Path) -> None:
        bad = _variant(
            detectability="rejected",
            status="active",
            rejection_reason="No pose estimation.",
        )
        directory = _write(tmp_path, bad)
        with pytest.raises(CompileError, match="rejected"):
            compile_scenarios(directory, capability=_capability())


# ─── Artifact ─────────────────────────────────────────────────────────────────


class TestArtifact:
    @pytest.mark.unit
    def test_carries_capability_map_and_fingerprint(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        assert artifact["taxonomy_fingerprint"].startswith("sha256:")
        assert artifact["classes"]["person"]["detection_level"] == "bbox"

    @pytest.mark.unit
    def test_inverted_index_maps_classes_to_scenarios(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        assert artifact["inverted_index"] == {
            "person": ["SC-KIT-001"],
            "stove": ["SC-KIT-001"],
        }

    @pytest.mark.unit
    def test_inverted_index_is_deterministic_and_sorted(self) -> None:
        from src.scenario_engine.schema import Scenario

        a = Scenario.from_mapping(_variant(scenario_id="SC-KIT-002", priority=5))
        b = Scenario.from_mapping(_SCENARIO)
        assert build_inverted_index([a, b]) == build_inverted_index([b, a])
        assert build_inverted_index([a, b])["stove"] == ["SC-KIT-001", "SC-KIT-002"]

    @pytest.mark.unit
    def test_rows_carry_rule_hashes(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        assert artifact["scenarios"][0]["rule_hash"].startswith("sha256:")

    @pytest.mark.unit
    def test_compilation_is_deterministic(self, tmp_path: Path) -> None:
        directory = _write(tmp_path, _SCENARIO)
        first = compile_scenarios(directory, capability=_capability()).artifact
        second = compile_scenarios(directory, capability=_capability()).artifact
        assert first == second
        assert first["content_hash"] == second["content_hash"]

    @pytest.mark.unit
    def test_written_bytes_use_lf(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        path = write_artifact(artifact, tmp_path / "out" / "scenarios.compiled.json")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")


class TestVerifyArtifact:
    @pytest.mark.unit
    def test_accepts_a_freshly_compiled_artifact(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        verify_artifact(artifact)

    @pytest.mark.unit
    def test_rejects_a_hand_edited_artifact(self, tmp_path: Path) -> None:
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        artifact["scenarios"][0]["risk_level"] = "INFO"
        with pytest.raises(CompileError, match="content_hash mismatch"):
            verify_artifact(artifact)

    @pytest.mark.unit
    def test_rejects_a_truncated_artifact(self, tmp_path: Path) -> None:
        """The legacy loader accepted this and logged 'Loaded 0 rules' at INFO."""
        artifact = compile_scenarios(_write(tmp_path, _SCENARIO), capability=_capability()).artifact
        artifact["scenarios"] = []
        with pytest.raises(CompileError):
            verify_artifact(artifact)

    @pytest.mark.unit
    def test_rejects_a_missing_content_hash(self) -> None:
        with pytest.raises(CompileError, match="no content_hash"):
            verify_artifact({"artifact_schema_version": 1, "scenarios": [], "rule_count": 0})

    @pytest.mark.unit
    def test_rejects_an_unsupported_schema_version(self) -> None:
        with pytest.raises(CompileError, match="schema version"):
            verify_artifact({"artifact_schema_version": 99})

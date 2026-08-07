"""
Unit tests for src.scenario_engine.csv_view.

Three tests here are the ADR-P6-02 contract, not incidental coverage:

* the view is byte-identical LF output, which is what catches a CRLF regression
  on the Windows CI leg;
* the editable subset excludes prompt fields, which is what keeps Devanagari out
  of Excel's cp1252 reach;
* the editable subset cannot create or delete scenarios.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.scenario_engine.csv_view import (
    EDITABLE_COLUMNS,
    VIEW_COLUMNS,
    CsvViewError,
    artifact_rows,
    diff_editable,
    read_editable_csv,
    write_editable_csv,
    write_view_csv,
)

_ARTIFACT: dict[str, Any] = {
    "scenarios": [
        {
            "scenario_id": "SC-KIT-001",
            "scenario_name": "Cooking left unattended",
            "category": "kitchen",
            "status": "active",
            "detectability": "direct",
            "claim_class": "observation",
            "risk_level": "CRITICAL",
            "priority": 10,
            "condition": "(detected(class_name='stove') AND absent_for(...))",
            "required_objects": ["person", "stove"],
            "optional_objects": [],
            "room_context": "kitchen",
            "min_confidence": 0.5,
            "min_dwell_seconds": 120.0,
            "clear_after_seconds": 30.0,
            "cooldown_seconds": 180,
            "max_repeats": 2,
            "max_per_day": 4,
            "patient_facing": True,
            "messages": {"en": "Nobody has been near the stove for a while.", "hi": "…"},
            "next_best_action": "Ask someone to check the kitchen",
            "caregiver_channel": "push",
            "evidence": [{"source_id": "NFPA-CookingSafety", "strength": "guideline"}],
            "capability_disclaimer": "",
            "rejection_reason": "",
            "reviewed_by": "clinical-review",
            "reviewed_on": "2026-08-07",
            "rule_hash": "sha256:abc",
        }
    ],
    "deprecated": [],
    "rejected": [
        {
            "scenario_id": "SC-SAF-001",
            "scenario_name": "Fall detection",
            "category": "safety",
            "status": "rejected",
            "detectability": "rejected",
            "claim_class": "inference",
            "risk_level": "CRITICAL",
            "priority": 0,
            "condition": "detected(class_name='person')",
            "required_objects": ["person"],
            "optional_objects": [],
            "room_context": None,
            "min_confidence": 0.5,
            "min_dwell_seconds": 0.0,
            "clear_after_seconds": 30.0,
            "cooldown_seconds": 60,
            "max_repeats": 2,
            "max_per_day": 6,
            "patient_facing": False,
            "messages": {},
            "next_best_action": "",
            "caregiver_channel": "none",
            "evidence": [],
            "capability_disclaimer": "",
            "rejection_reason": "No pose estimation; a low wide box is a person bending.",
            "reviewed_by": "",
            "reviewed_on": "",
            "rule_hash": "sha256:def",
        }
    ],
}


class TestViewCsv:
    @pytest.mark.unit
    def test_rows_are_sorted_and_include_every_lifecycle_state(self) -> None:
        rows = artifact_rows(_ARTIFACT)
        assert [row["scenario_id"] for row in rows] == ["SC-KIT-001", "SC-SAF-001"]

    @pytest.mark.unit
    def test_lists_are_flattened_with_the_declared_separator(self) -> None:
        row = artifact_rows(_ARTIFACT)[0]
        assert row["required_objects"] == "person; stove"
        assert row["optional_objects"] == ""

    @pytest.mark.unit
    def test_evidence_is_projected_into_flat_columns(self) -> None:
        row = artifact_rows(_ARTIFACT)[0]
        assert row["evidence_sources"] == "NFPA-CookingSafety"
        assert row["evidence_strength"] == "guideline"

    @pytest.mark.unit
    def test_written_bytes_use_lf_and_quote_everything(self, tmp_path: Path) -> None:
        """Catches a CRLF regression on the Windows CI leg."""
        path = write_view_csv(_ARTIFACT, tmp_path / "scenarios.view.csv")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.startswith(b'"scenario_id","scenario_name"')

    @pytest.mark.unit
    def test_output_is_byte_stable_across_runs(self, tmp_path: Path) -> None:
        first = write_view_csv(_ARTIFACT, tmp_path / "a.view.csv").read_bytes()
        second = write_view_csv(_ARTIFACT, tmp_path / "b.view.csv").read_bytes()
        assert first == second

    @pytest.mark.unit
    def test_filename_must_announce_it_is_a_view(self, tmp_path: Path) -> None:
        with pytest.raises(CsvViewError, match="view.csv"):
            write_view_csv(_ARTIFACT, tmp_path / "scenarios.csv")

    @pytest.mark.unit
    def test_embedded_newlines_are_rejected_not_escaped(self, tmp_path: Path) -> None:
        artifact = deepcopy(_ARTIFACT)
        artifact["scenarios"][0]["next_best_action"] = "Check the kitchen.\nThen call."
        with pytest.raises(CsvViewError, match="newline"):
            write_view_csv(artifact, tmp_path / "scenarios.view.csv")

    @pytest.mark.unit
    def test_prompts_are_not_a_view_column_either(self) -> None:
        assert "messages" not in VIEW_COLUMNS


class TestEditableCsv:
    @pytest.mark.unit
    def test_excludes_prompt_fields(self) -> None:
        """What keeps Devanagari out of Excel's cp1252 reach."""
        assert "messages" not in EDITABLE_COLUMNS
        assert not any("message" in column or "prompt" in column for column in EDITABLE_COLUMNS)

    @pytest.mark.unit
    def test_round_trips_the_restricted_subset(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "scenarios.editable.csv")
        loaded = read_editable_csv(path)
        assert set(loaded) == {"SC-KIT-001", "SC-SAF-001"}
        assert loaded["SC-KIT-001"]["risk_level"] == "CRITICAL"
        assert set(loaded["SC-KIT-001"]) == set(EDITABLE_COLUMNS)

    @pytest.mark.unit
    def test_written_with_a_bom_for_excel(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "scenarios.editable.csv")
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    @pytest.mark.unit
    def test_extra_columns_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text(
            "scenario_id,risk_level,messages\nSC-KIT-001,HIGH,hello\n", encoding="utf-8-sig"
        )
        with pytest.raises(CsvViewError, match="not editable"):
            read_editable_csv(path)

    @pytest.mark.unit
    def test_a_row_without_an_id_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("scenario_id,risk_level\n,HIGH\n", encoding="utf-8-sig")
        with pytest.raises(CsvViewError, match="no scenario_id"):
            read_editable_csv(path)


class TestDiffEditable:
    @pytest.mark.unit
    def test_no_changes_is_empty(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "e.csv")
        assert diff_editable(_ARTIFACT, read_editable_csv(path)) == {}

    @pytest.mark.unit
    def test_detects_an_allowed_change(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "e.csv")
        edited = read_editable_csv(path)
        edited["SC-KIT-001"]["risk_level"] = "HIGH"
        changes = diff_editable(_ARTIFACT, edited)
        assert changes == {"SC-KIT-001": {"risk_level": ("CRITICAL", "HIGH")}}

    @pytest.mark.unit
    def test_cannot_create_a_scenario(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "e.csv")
        edited = dict(read_editable_csv(path))
        edited["SC-NEW-001"] = {column: "x" for column in EDITABLE_COLUMNS}
        with pytest.raises(CsvViewError, match="cannot create"):
            diff_editable(_ARTIFACT, edited)

    @pytest.mark.unit
    def test_cannot_delete_a_scenario(self, tmp_path: Path) -> None:
        path = write_editable_csv(_ARTIFACT, tmp_path / "e.csv")
        edited = dict(read_editable_csv(path))
        del edited["SC-SAF-001"]
        with pytest.raises(CsvViewError, match="cannot delete"):
            diff_editable(_ARTIFACT, edited)

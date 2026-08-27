"""
Unit tests for src.scenario_engine.clip_dataset.

The report's whole job is to refuse to call a directory of files a dataset, so
the tests are mostly about what makes it say FAIL.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.scenario_engine.clip_dataset import (
    build_report,
    impossible_positives,
    min_clip_seconds,
)
from src.scenario_engine.clips import ClipManifest

_BASE: dict[str, Any] = {
    "session_id": "h01_kitchen_s001",
    "house_id": "h01",
    "room": "kitchen",
    "lighting": "tubelight",
    "scenario_id": "SC-BTH-001",
    "consent_reference": "CONSENT-h01-2026-002",
    "sha256": "abc123",
    "duration_s": 15.0,
    "fps": 30.0,
    "rule_hash_at_label_time": "sha256:deadbeef",
    "metadata_stripped": True,
}


def _clip(index: int, **overrides: Any) -> ClipManifest:
    negative = overrides.pop("negative", False)
    raw: dict[str, Any] = {
        **_BASE,
        "clip_id": f"h01_kitchen_s001_c{index:03d}",
        "polarity": "negative" if negative else "positive",
        "expected": (
            {"fires": False}
            if negative
            else {
                "fires": True,
                "scenario_id": overrides.get("scenario_id", _BASE["scenario_id"]),
                "first_alert_within_s": 10.0,
            }
        ),
        **overrides,
    }
    if negative:
        raw.setdefault("negative_kind", "absence")
    return ClipManifest.from_mapping(raw)


def _accepted(index: int, **overrides: Any) -> ClipManifest:
    return _clip(index, review_status="accepted", reviewed_by="qa", **overrides)


class TestAcceptanceIsWhatCounts:
    @pytest.mark.unit
    def test_pending_clips_do_not_count_toward_the_target(self) -> None:
        """100 files on disk, none reviewed, is zero clips."""
        clips = [_clip(i) for i in range(100)]
        report = build_report(clips, scenario_ids=["SC-BTH-001"], target_accepted=100)
        assert report.total_manifests == 100
        assert report.accepted == 0
        assert report.target_met is False
        assert report.verdict == "FAIL"
        assert any("not accepted" in p for p in report.problems)

    @pytest.mark.unit
    def test_rejected_clips_do_not_count(self) -> None:
        clips = [_clip(i, review_status="rejected", rejection_reason="too dark") for i in range(10)]
        report = build_report(clips, scenario_ids=[], target_accepted=10)
        assert report.accepted == 0
        assert report.rejected == 10
        assert len(report.rejections) == 10
        assert report.rejections[0]["reason"] == "too dark"


class TestNegativeFloor:
    @pytest.mark.unit
    def test_an_all_positive_batch_fails(self) -> None:
        clips = [_accepted(i) for i in range(10)]
        report = build_report(clips, scenario_ids=[], target_accepted=10, min_per_scenario=1)
        assert report.negative_fraction == 0.0
        assert report.negative_fraction_met is False
        assert any("blind to the false-positive rate" in p for p in report.problems)

    @pytest.mark.unit
    def test_thirty_percent_negatives_passes(self) -> None:
        clips = [_accepted(i, negative=i < 3) for i in range(10)]
        report = build_report(clips, scenario_ids=[], target_accepted=10, min_per_scenario=1)
        assert report.negative_fraction == pytest.approx(0.30)
        assert report.negative_fraction_met is True
        assert report.verdict == "PASS"

    @pytest.mark.unit
    def test_twenty_nine_percent_fails(self) -> None:
        """The floor is a floor, not a target to approach."""
        clips = [_accepted(i, negative=i < 29) for i in range(100)]
        report = build_report(clips, scenario_ids=[], target_accepted=100, min_per_scenario=1)
        assert report.negative_fraction_met is False


class TestCoverage:
    @pytest.mark.unit
    def test_a_scenario_with_no_clips_is_visible_not_absent(self) -> None:
        """Otherwise an uncollected scenario looks like a scenario with no problems."""
        clips = [_accepted(i, negative=i < 4) for i in range(10)]
        report = build_report(clips, scenario_ids=["SC-BTH-001", "SC-KIT-001"], target_accepted=10)
        ids = {c.scenario_id for c in report.coverage}
        assert ids == {"SC-BTH-001", "SC-KIT-001"}
        kitchen = next(c for c in report.coverage if c.scenario_id == "SC-KIT-001")
        assert kitchen.accepted == 0
        assert kitchen.shortfall == 4
        assert "SC-KIT-001" in report.uncovered_scenarios

    @pytest.mark.unit
    def test_shortfall_counts_down(self) -> None:
        clips = [_accepted(0), _accepted(1)]
        report = build_report(clips, scenario_ids=["SC-BTH-001"], min_per_scenario=4)
        coverage = report.coverage[0]
        assert (coverage.accepted, coverage.shortfall, coverage.sufficient) == (2, 2, False)


class TestPrivacyAndProvenanceAccounting:
    @pytest.mark.unit
    def test_unsanitised_clips_are_a_problem(self) -> None:
        clips = [_accepted(0, metadata_stripped=False)]
        report = build_report(clips, scenario_ids=[], target_accepted=1, min_per_scenario=1)
        assert report.privacy_unstripped == 1
        assert any("metadata_stripped" in p for p in report.problems)

    @pytest.mark.unit
    def test_external_and_own_capture_are_counted_separately(self) -> None:
        external = {
            "source_type": "external",
            "source_url": "https://commons.wikimedia.org/x",
            "source_platform": "wikimedia",
            "creator": "someone",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        }
        raw = {**_BASE, "provenance": external, "indian_home": False}
        del raw["consent_reference"]
        clips = [
            _accepted(0),
            ClipManifest.from_mapping(
                {
                    **raw,
                    "clip_id": "h01_kitchen_s001_c001",
                    "polarity": "negative",
                    "negative_kind": "confuser",
                    "expected": {"fires": False},
                    "review_status": "accepted",
                    "reviewed_by": "qa",
                }
            ),
        ]
        report = build_report(clips, scenario_ids=[], target_accepted=2, min_per_scenario=1)
        assert (report.own_capture, report.external) == (1, 1)
        assert (report.consent_complete, report.licence_complete) == (1, 1)
        assert report.indian_home == 1


_SC_KIT_001 = (
    "(in_room(room='kitchen') AND present_for(class_name='stove', seconds=60) "
    "AND ever_seen(class_name='person') AND absent_for(class_name='person', seconds=900))"
)
_SC_BTH_001 = (
    "(detected(class_name='wet_floor') AND present_for(class_name='wet_floor', seconds=3) "
    "AND detected(class_name='person'))"
)


class TestMinClipSeconds:
    @pytest.mark.unit
    def test_present_and_absent_add(self) -> None:
        """The absence has to elapse after the presence established the state."""
        assert min_clip_seconds(_SC_KIT_001) == 960.0

    @pytest.mark.unit
    def test_a_short_dwell_scenario_is_filmable(self) -> None:
        assert min_clip_seconds(_SC_BTH_001) == 3.0

    @pytest.mark.unit
    def test_no_temporal_predicate_is_zero(self) -> None:
        assert min_clip_seconds("(detected(class_name='knife'))") == 0.0


class TestImpossiblePositives:
    @pytest.mark.unit
    def test_a_twenty_second_kitchen_positive_is_impossible(self) -> None:
        """The likeliest mislabelling in a short-clip batch, and the hardest to
        spot by eye: the footage looks exactly like the hazard."""
        clips = [_accepted(0, scenario_id="SC-KIT-001", duration_s=20.0)]
        problems = impossible_positives(clips, {"SC-KIT-001": _SC_KIT_001})
        assert len(problems) == 1
        assert "pre_dwell" in problems[0]

    @pytest.mark.unit
    def test_a_long_enough_positive_is_accepted(self) -> None:
        clips = [_accepted(0, scenario_id="SC-KIT-001", duration_s=1000.0)]
        assert impossible_positives(clips, {"SC-KIT-001": _SC_KIT_001}) == []

    @pytest.mark.unit
    def test_negatives_are_never_impossible(self) -> None:
        """A short clip that expects nothing to fire is always a valid claim."""
        clips = [_accepted(0, negative=True, scenario_id="SC-KIT-001", duration_s=15.0)]
        assert impossible_positives(clips, {"SC-KIT-001": _SC_KIT_001}) == []

    @pytest.mark.unit
    def test_a_short_scenario_positive_is_fine_at_fifteen_seconds(self) -> None:
        clips = [_accepted(0, scenario_id="SC-BTH-001", duration_s=15.0)]
        assert impossible_positives(clips, {"SC-BTH-001": _SC_BTH_001}) == []

    @pytest.mark.unit
    def test_it_surfaces_through_the_report(self) -> None:
        clips = [_accepted(i, negative=i < 4, scenario_id="SC-KIT-001") for i in range(10)]
        report = build_report(
            clips,
            scenario_ids=["SC-KIT-001"],
            target_accepted=10,
            min_per_scenario=1,
            conditions={"SC-KIT-001": _SC_KIT_001},
        )
        assert report.verdict == "FAIL"
        assert any("pre_dwell" in p for p in report.problems)


class TestEmptyBatch:
    @pytest.mark.unit
    def test_zero_clips_reports_the_gap_without_crashing(self) -> None:
        report = build_report([], scenario_ids=["SC-BTH-001"], target_accepted=100)
        assert report.accepted == 0
        assert report.negative_fraction == 0.0
        assert report.verdict == "FAIL"
        assert report.to_dict()["totals"]["target_accepted"] == 100

"""
Unit tests for src.scenario_engine.clips.

The sanitisation *policy* is tested without ffmpeg installed, because the policy
is the part worth testing — which keys must not survive, and that a missing
toolchain is an error rather than a skip. The subprocess call itself is thin.

ffmpeg is absent on most dev boxes and in CI, so the tests that genuinely need
it are guarded, following the house pattern for optional heavy dependencies
(ADR-P5-11).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.scenario_engine.clips import (
    FORBIDDEN_METADATA_KEYS,
    ClipError,
    ClipManifest,
    ExpectedOutcome,
    FfmpegUnavailableError,
    ffmpeg_available,
    read_metadata,
    residual_metadata,
    strip_metadata,
)

_CLIP: dict[str, Any] = {
    "clip_id": "h01_kitchen_s001_c001",
    "session_id": "h01_kitchen_s001",
    "house_id": "h01",
    "room": "kitchen",
    "lighting": "evening",
    "polarity": "positive",
    "scenario_id": "SC-KIT-001",
    "consent_reference": "CONSENT-h01-2026-001",
    "sha256": "abc123",
    "duration_s": 45.0,
    "fps": 30.0,
    "rule_hash_at_label_time": "sha256:deadbeef",
    "expected": {
        "fires": True,
        "scenario_id": "SC-KIT-001",
        "severity": "CRITICAL",
        "first_alert_within_s": 30.0,
    },
}


def _clip(**overrides: Any) -> ClipManifest:
    raw = {**_CLIP, **overrides}
    return ClipManifest.from_mapping(raw)


class TestSanitisationPolicy:
    @pytest.mark.unit
    def test_gps_carriers_are_forbidden(self) -> None:
        """The moov/udta atoms an EXIF stripper never touches."""
        assert "location" in FORBIDDEN_METADATA_KEYS
        assert "com.apple.quicktime.location.iso6709" in FORBIDDEN_METADATA_KEYS

    @pytest.mark.unit
    def test_residual_metadata_matches_case_insensitively(self) -> None:
        surviving = residual_metadata({"Location": "+12.9716+77.5946/", "title": "kitchen clip"})
        assert set(surviving) == {"Location"}

    @pytest.mark.unit
    def test_clean_tags_leave_no_residue(self) -> None:
        assert residual_metadata({"title": "clip", "comment": "for review"}) == {}

    @pytest.mark.unit
    def test_device_identifiers_are_caught(self) -> None:
        tags = {"com.apple.quicktime.model": "iPhone 15", "creation_time": "2026-08-07T10:00:00Z"}
        assert set(residual_metadata(tags)) == set(tags)


@pytest.mark.skipif(ffmpeg_available(), reason="ffmpeg is installed on this machine")
class TestMissingToolchainIsAnError:
    """A privacy control that silently degrades is not a control."""

    @pytest.mark.unit
    def test_strip_refuses_rather_than_skipping(self, tmp_path: Any) -> None:
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"not-a-real-mp4")
        with pytest.raises(FfmpegUnavailableError, match="refused"):
            strip_metadata(source, tmp_path / "out.mp4")

    @pytest.mark.unit
    def test_read_metadata_also_refuses(self, tmp_path: Any) -> None:
        with pytest.raises(FfmpegUnavailableError):
            read_metadata(tmp_path / "clip.mp4")

    @pytest.mark.unit
    def test_the_error_explains_why_exif_stripping_is_insufficient(self, tmp_path: Any) -> None:
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"x")
        with pytest.raises(FfmpegUnavailableError) as excinfo:
            strip_metadata(source, tmp_path / "out.mp4")
        assert "EXIF" in str(excinfo.value)


class TestExpectedOutcome:
    @pytest.mark.unit
    def test_positive_clip_needs_a_deadline(self) -> None:
        """Without a deadline a clip records behaviour instead of asserting it."""
        with pytest.raises(ClipError, match="first_alert_within_s"):
            ExpectedOutcome.from_mapping({"fires": True, "scenario_id": "SC-KIT-001"}, "expected")

    @pytest.mark.unit
    def test_positive_clip_needs_a_scenario_id(self) -> None:
        with pytest.raises(ClipError, match="scenario_id"):
            ExpectedOutcome.from_mapping({"fires": True, "first_alert_within_s": 5}, "expected")

    @pytest.mark.unit
    def test_negative_clip_needs_neither(self) -> None:
        outcome = ExpectedOutcome.from_mapping({"fires": False}, "expected")
        assert outcome.fires is False


class TestClipManifest:
    @pytest.mark.unit
    def test_valid_positive_clip(self) -> None:
        clip = _clip()
        assert clip.clip_id == "h01_kitchen_s001_c001"
        assert clip.expected.first_alert_within_s == 30.0

    @pytest.mark.unit
    def test_missing_consent_reference_is_refused(self) -> None:
        raw = {**_CLIP}
        del raw["consent_reference"]
        with pytest.raises(ClipError, match="consent_reference"):
            ClipManifest.from_mapping(raw)

    @pytest.mark.unit
    def test_negative_clip_must_state_why_it_exists(self) -> None:
        with pytest.raises(ClipError, match="negative_kind"):
            _clip(polarity="negative", expected={"fires": False})

    @pytest.mark.unit
    def test_negative_kind_is_a_controlled_vocabulary(self) -> None:
        with pytest.raises(ClipError, match="negative_kind"):
            _clip(polarity="negative", negative_kind="misc", expected={"fires": False})

    @pytest.mark.unit
    def test_confuser_negative_is_accepted(self) -> None:
        """Shiny floor read as wet — the documented confuser for wet_floor."""
        clip = _clip(polarity="negative", negative_kind="confuser", expected={"fires": False})
        assert clip.polarity == "negative"

    @pytest.mark.unit
    def test_polarity_and_expectation_must_agree(self) -> None:
        with pytest.raises(ClipError, match="negative clip cannot expect"):
            _clip(polarity="negative", negative_kind="absence")
        with pytest.raises(ClipError, match="positive clip must expect"):
            _clip(expected={"fires": False})

    @pytest.mark.unit
    def test_frame_prefix_matches_the_existing_group_pattern(self) -> None:
        """So dataset_utils' leakage grouping recognises the clip for free."""
        assert _clip().frame_prefix() == "h01_kitchen_s001_c001_frame_"

    @pytest.mark.unit
    def test_staleness_is_detected_against_the_current_rule_hash(self) -> None:
        clip = _clip()
        assert clip.is_stale("sha256:deadbeef") is False
        assert clip.is_stale("sha256:something-else") is True

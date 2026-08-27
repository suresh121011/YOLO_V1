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

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.scenario_engine.clips import (
    DEFAULT_CAPTURE_CONFIG,
    FORBIDDEN_METADATA_KEYS,
    ClipError,
    ClipManifest,
    ClipRequirements,
    ExpectedOutcome,
    FfmpegUnavailableError,
    SourceProvenance,
    _parse_frame_rate,
    device_identifying_handlers,
    ffmpeg_available,
    load_clip_manifests,
    load_clip_requirements,
    parse_clip_id,
    read_metadata,
    residual_metadata,
    strip_metadata,
    validate_clip_set,
    verify_clip_consent,
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


class TestHandlerNameGate:
    """The MP4 `hdlr` box is mandatory, so absence is not an achievable gate.

    Found by the M7 acceptance run against ffmpeg 9.0: `handler_name` survives
    `-map_metadata -1` and `-metadata:s handler_name=` alike, because the muxer
    rewrites the box. Requiring its absence rejected every clip. The check is
    therefore on the value.
    """

    @pytest.mark.unit
    def test_handler_name_is_not_required_to_be_absent(self) -> None:
        assert "handler_name" not in FORBIDDEN_METADATA_KEYS

    @pytest.mark.unit
    def test_ffmpegs_own_handlers_are_neutral(self) -> None:
        assert device_identifying_handlers({"handler_name": "VideoHandler"}) == {}
        assert device_identifying_handlers({"handler_name": "SoundHandler"}) == {}

    @pytest.mark.unit
    def test_a_device_chosen_handler_leaks_the_phone_model(self) -> None:
        surviving = device_identifying_handlers({"handler_name": "Samsung Video Handler"})
        assert surviving == {"handler_name": "Samsung Video Handler"}

    @pytest.mark.unit
    def test_matching_is_case_insensitive(self) -> None:
        assert device_identifying_handlers({"Handler_Name": "videohandler"}) == {}

    @pytest.mark.unit
    def test_other_tags_are_not_examined_by_this_gate(self) -> None:
        assert device_identifying_handlers({"encoder": "Lavf63.1.100"}) == {}

    @pytest.mark.unit
    def test_encoder_is_still_required_to_be_absent(self) -> None:
        """It IS removable, via -fflags +bitexact, so the gate stays strict."""
        assert "encoder" in FORBIDDEN_METADATA_KEYS
        assert residual_metadata({"encoder": "Lavf63.1.100"}) != {}


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

    @pytest.mark.unit
    def test_round_trips_through_to_dict(self) -> None:
        """The manifest on disk must reload into the same assertion."""
        restored = ClipManifest.from_mapping(_clip().to_dict())
        assert restored == _clip()


_EXTERNAL: dict[str, Any] = {
    "source_type": "external",
    "source_url": "https://commons.wikimedia.org/wiki/File:Kitchen.webm",
    "source_platform": "wikimedia",
    "creator": "A. Contributor",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "download_date": "2026-08-11",
    "original_video_id": "File:Kitchen.webm",
    "start_time_s": 12.0,
    "end_time_s": 27.0,
}


class TestSourceProvenance:
    """Permission comes from consent OR a licence, never from neither."""

    @pytest.mark.unit
    def test_own_capture_is_the_default(self) -> None:
        assert SourceProvenance.from_mapping(None).source_type == "own_capture"

    @pytest.mark.unit
    def test_external_clip_needs_the_full_provenance_record(self) -> None:
        """A publicly viewable video is not automatically reusable."""
        for field_name in ("source_url", "creator", "license", "license_url", "source_platform"):
            raw = {**_EXTERNAL, field_name: ""}
            with pytest.raises(ClipError, match=field_name):
                SourceProvenance.from_mapping(raw)

    @pytest.mark.unit
    def test_unrecognised_licence_is_refused(self) -> None:
        """An allowlist, so 'free to use' cannot be recorded as a licence."""
        with pytest.raises(ClipError, match="allowlist"):
            SourceProvenance.from_mapping({**_EXTERNAL, "license": "free-to-use"})

    @pytest.mark.unit
    def test_a_complete_external_record_is_accepted(self) -> None:
        provenance = SourceProvenance.from_mapping(_EXTERNAL)
        assert provenance.is_external
        assert provenance.start_time_s == 12.0

    @pytest.mark.unit
    def test_licence_fields_on_own_footage_are_refused(self) -> None:
        """Own footage is permitted by consent; a licence there is a mix-up."""
        with pytest.raises(ClipError, match="own_capture"):
            SourceProvenance.from_mapping(
                {"source_type": "own_capture", "license": "CC-BY-4.0", "license_url": "x"}
            )

    @pytest.mark.unit
    def test_unknown_source_type_is_refused(self) -> None:
        with pytest.raises(ClipError, match="source_type"):
            SourceProvenance.from_mapping({"source_type": "scraped"})


class TestPermissionIsExclusive:
    @pytest.mark.unit
    def test_external_clip_must_not_carry_a_consent_reference(self) -> None:
        """An id there would record a consent no human ever gave."""
        with pytest.raises(ClipError, match="no human gave"):
            _clip(provenance=_EXTERNAL)

    @pytest.mark.unit
    def test_external_clip_without_consent_is_accepted(self) -> None:
        raw = {**_CLIP, "provenance": _EXTERNAL}
        del raw["consent_reference"]
        clip = ClipManifest.from_mapping(raw)
        assert clip.consent_reference == ""
        assert clip.provenance.license == "CC-BY-4.0"

    @pytest.mark.unit
    def test_own_capture_without_consent_is_refused(self) -> None:
        raw = {**_CLIP}
        del raw["consent_reference"]
        with pytest.raises(ClipError, match="consent_reference"):
            ClipManifest.from_mapping(raw)


class TestReviewLifecycle:
    @pytest.mark.unit
    def test_clips_start_pending_not_accepted(self) -> None:
        """A file on disk is not a dataset member."""
        assert _clip().review_status == "pending"
        assert _clip().is_accepted is False

    @pytest.mark.unit
    def test_acceptance_must_name_a_human(self) -> None:
        """Sanitisation is machine-checked and does not stand in for review."""
        with pytest.raises(ClipError, match="reviewed_by"):
            _clip(review_status="accepted")

    @pytest.mark.unit
    def test_accepted_clip_with_a_reviewer(self) -> None:
        clip = _clip(review_status="accepted", reviewed_by="qa-lead")
        assert clip.is_accepted is True

    @pytest.mark.unit
    def test_rejection_must_state_a_reason(self) -> None:
        with pytest.raises(ClipError, match="rejection_reason"):
            _clip(review_status="rejected")

    @pytest.mark.unit
    def test_unknown_review_status_is_refused(self) -> None:
        with pytest.raises(ClipError, match="review_status"):
            _clip(review_status="probably-fine")


class TestPreDwellNegative:
    @pytest.mark.unit
    def test_pre_dwell_is_a_valid_negative_kind(self) -> None:
        """A 20 s clip of a person at a stove: the engine correctly stays silent
        because SC-KIT-001 needs 15 minutes of absence. That is a true negative,
        and it is the shape most likely to be mislabelled as a positive."""
        clip = _clip(polarity="negative", negative_kind="pre_dwell", expected={"fires": False})
        assert clip.negative_kind == "pre_dwell"


def _write_config(tmp_path: Path, **clip_overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "capture": {"inbox_dir": "data/capture_inbox"},
        "clips": {
            "inbox_dir": "data/clip_inbox",
            "clips_root": "data/scenario_engine/clips",
            "clip_id_pattern": r"^h\d{2}_[a-z_]+_s\d{3}_c\d{3}$",
            "consent_scope": "scenario-video",
            "min_negative_fraction": 0.30,
            "video": {
                "allowed_extensions": [".mp4", ".mov"],
                "min_duration_s": 10,
                "max_duration_s": 120,
                "min_fps": 15,
                "max_file_mb": 500,
                "strip_metadata": True,
            },
            **clip_overrides,
        },
    }
    path = tmp_path / "capture_config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestClipRequirements:
    @pytest.mark.unit
    def test_the_repo_config_carries_a_clips_block(self) -> None:
        """A `clips:` block that silently falls back to defaults is untested config."""
        raw = yaml.safe_load(DEFAULT_CAPTURE_CONFIG.read_text(encoding="utf-8"))
        assert "clips" in raw, "configs/capture_config.yaml lost its clips: block"
        requirements = load_clip_requirements()
        assert requirements.consent_scope == "scenario-video"

    @pytest.mark.unit
    def test_image_extensions_were_not_widened_to_video(self) -> None:
        """Adding .mp4 there would send clips through the PIL-backed min_dim gate."""
        raw = yaml.safe_load(DEFAULT_CAPTURE_CONFIG.read_text(encoding="utf-8"))
        image_extensions = raw["capture"]["image"]["allowed_extensions"]
        assert not {".mp4", ".mov"} & set(image_extensions)

    @pytest.mark.unit
    def test_missing_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        requirements = load_clip_requirements(tmp_path / "absent.yaml")
        assert requirements == ClipRequirements()

    @pytest.mark.unit
    def test_strip_metadata_cannot_be_switched_off(self, tmp_path: Path) -> None:
        """The privacy control is not a preference."""
        path = _write_config(tmp_path, video={"strip_metadata": False})
        with pytest.raises(ClipError, match="strip_metadata"):
            load_clip_requirements(path)

    @pytest.mark.unit
    def test_inverted_duration_bounds_are_refused(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, video={"min_duration_s": 90, "max_duration_s": 30})
        with pytest.raises(ClipError, match="min_duration_s"):
            load_clip_requirements(path)

    @pytest.mark.unit
    def test_bad_clip_id_pattern_fails_at_load(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, clip_id_pattern="^h(\\d{2}$")
        with pytest.raises(ClipError, match="clip_id_pattern"):
            load_clip_requirements(path)

    @pytest.mark.unit
    def test_clip_id_grammar(self) -> None:
        requirements = ClipRequirements()
        assert requirements.validate_clip_id("h01_pooja_room_s002_c010") == []
        assert requirements.validate_clip_id("h01_kitchen_s001") != []

    @pytest.mark.unit
    def test_short_clip_cannot_exercise_a_dwell_threshold(self) -> None:
        problems = ClipRequirements().check_media(duration_s=4.0, fps=30.0)
        assert any("min_duration_s" in p for p in problems)

    @pytest.mark.unit
    def test_low_frame_rate_is_rejected_not_warned(self) -> None:
        """A 10 fps clip silently doubles every frame-counted dwell it tests."""
        problems = ClipRequirements().check_media(duration_s=45.0, fps=10.0)
        assert any("min_fps" in p for p in problems)

    @pytest.mark.unit
    def test_a_conforming_clip_passes_both_gates(self) -> None:
        assert ClipRequirements().check_media(duration_s=45.0, fps=30.0) == []

    @pytest.mark.unit
    def test_unprobeable_media_is_a_problem_not_a_pass(self) -> None:
        """probe_media returns 0.0 for values ffprobe cannot determine."""
        problems = ClipRequirements().check_media(duration_s=0.0, fps=0.0)
        assert len(problems) == 2

    @pytest.mark.unit
    def test_wrong_extension_is_refused(self, tmp_path: Path) -> None:
        source = tmp_path / "clip.avi"
        source.write_bytes(b"x")
        assert any("extension" in p for p in ClipRequirements().check_file(source))


class TestFrameRateParsing:
    @pytest.mark.unit
    def test_ntsc_rational_notation(self) -> None:
        assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.01)

    @pytest.mark.unit
    def test_degenerate_values_do_not_raise(self) -> None:
        for raw in ("0/0", "N/A", "", None, "garbage"):
            assert _parse_frame_rate(raw) == 0.0


class TestClipConsent:
    """Scope enforcement — the control that image ingest does not have."""

    @staticmethod
    def _registry(tmp_path: Path, **record: Any) -> Path:
        base = {"house_id": "h01", "granted_on": "2026-07-20", "scope": "scenario-video"}
        path = tmp_path / "consent_registry.yaml"
        path.write_text(yaml.safe_dump({"CONSENT-h01-2026-001": {**base, **record}}))
        return path

    @pytest.mark.unit
    def test_video_scope_is_accepted(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        assert (
            verify_clip_consent("CONSENT-h01-2026-001", "h01", ClipRequirements(), registry) == []
        )

    @pytest.mark.unit
    def test_image_consent_does_not_imply_video_consent(self, tmp_path: Path) -> None:
        """A still can be curated to avoid faces; 30 seconds of video cannot."""
        registry = self._registry(tmp_path, scope="dataset-training")
        problems = verify_clip_consent("CONSENT-h01-2026-001", "h01", ClipRequirements(), registry)
        assert any("scenario-video" in p for p in problems)

    @pytest.mark.unit
    def test_withdrawn_consent_blocks_ingest(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path, withdrawn=True)
        problems = verify_clip_consent("CONSENT-h01-2026-001", "h01", ClipRequirements(), registry)
        assert any("WITHDRAWN" in p for p in problems)

    @pytest.mark.unit
    def test_consent_for_another_house_is_refused(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        problems = verify_clip_consent("CONSENT-h01-2026-001", "h02", ClipRequirements(), registry)
        assert any("covers house" in p for p in problems)

    @pytest.mark.unit
    def test_missing_registry_is_fatal_unlike_image_ingest(self, tmp_path: Path) -> None:
        """Scope cannot be checked without the registry, so it is not downgraded."""
        problems = verify_clip_consent(
            "CONSENT-h01-2026-001", "h01", ClipRequirements(), tmp_path / "absent.yaml"
        )
        assert any("consent registry" in p for p in problems)

    @pytest.mark.unit
    def test_empty_reference_is_refused(self, tmp_path: Path) -> None:
        assert verify_clip_consent("", "h01", ClipRequirements(), tmp_path / "absent.yaml") != []


class TestParseClipId:
    @pytest.mark.unit
    def test_multi_word_room(self) -> None:
        assert parse_clip_id("h01_pooja_room_s002_c010") == (
            "h01",
            "pooja_room",
            "h01_pooja_room_s002",
        )

    @pytest.mark.unit
    def test_session_prefix_matches_the_capture_grammar(self) -> None:
        """So a clip's session is the same session the image workflow knows."""
        assert parse_clip_id("h01_kitchen_s001_c001")[2] == "h01_kitchen_s001"

    @pytest.mark.unit
    def test_too_few_tokens_raises(self) -> None:
        with pytest.raises(ClipError, match="clip id"):
            parse_clip_id("h01_kitchen")


class TestClipSet:
    @staticmethod
    def _set(count: int, negatives: int) -> list[ClipManifest]:
        clips: list[ClipManifest] = []
        for index in range(count):
            is_negative = index < negatives
            clips.append(
                _clip(
                    clip_id=f"h01_kitchen_s001_c{index:03d}",
                    metadata_stripped=True,
                    polarity="negative" if is_negative else "positive",
                    negative_kind="confuser" if is_negative else "",
                    expected=(
                        {"fires": False}
                        if is_negative
                        else {
                            "fires": True,
                            "scenario_id": "SC-KIT-001",
                            "first_alert_within_s": 30.0,
                        }
                    ),
                )
            )
        return clips

    @pytest.mark.unit
    def test_empty_set_is_not_a_failure(self) -> None:
        """The expected state before the first capture session."""
        assert validate_clip_set([], ClipRequirements()) == []

    @pytest.mark.unit
    def test_all_positive_suite_is_refused(self) -> None:
        """It measures sensitivity and is blind to the false-positive rate."""
        problems = validate_clip_set(self._set(10, negatives=0), ClipRequirements())
        assert any("negatives" in p for p in problems)

    @pytest.mark.unit
    def test_the_validation_strategy_ratio_passes(self) -> None:
        """3 negatives in 10 — the matrix at validation_strategy.md:63-76."""
        assert validate_clip_set(self._set(10, negatives=3), ClipRequirements()) == []

    @pytest.mark.unit
    def test_unsanitised_clip_is_caught_at_set_level(self) -> None:
        clips = self._set(10, negatives=3)
        clips[0] = _clip(clip_id=clips[0].clip_id, metadata_stripped=False)
        problems = validate_clip_set(clips, ClipRequirements())
        assert any("metadata_stripped" in p for p in problems)

    @pytest.mark.unit
    def test_stale_clip_is_reported_against_the_artifact(self) -> None:
        problems = validate_clip_set(
            self._set(10, negatives=3),
            ClipRequirements(),
            rule_hashes={"SC-KIT-001": "sha256:moved-on"},
        )
        assert any("re-review" in p for p in problems)

    @pytest.mark.unit
    def test_current_clip_is_not_reported_stale(self) -> None:
        assert (
            validate_clip_set(
                self._set(10, negatives=3),
                ClipRequirements(),
                rule_hashes={"SC-KIT-001": "sha256:deadbeef"},
            )
            == []
        )

    @pytest.mark.unit
    def test_clip_for_an_unknown_scenario_is_reported(self) -> None:
        problems = validate_clip_set(
            self._set(10, negatives=3), ClipRequirements(), rule_hashes={"SC-BTH-001": "sha256:x"}
        )
        assert any("not in the compiled artifact" in p for p in problems)


class TestLoadClipManifests:
    @pytest.mark.unit
    def test_absent_tree_loads_as_empty(self, tmp_path: Path) -> None:
        assert load_clip_manifests(tmp_path) == []

    @pytest.mark.unit
    def test_manifests_load_sorted_by_clip_id(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        for clip_id in ("h01_kitchen_s001_c002", "h01_kitchen_s001_c001"):
            payload = _clip(clip_id=clip_id).to_dict()
            (manifest_dir / f"{clip_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        assert [c.clip_id for c in load_clip_manifests(tmp_path)] == [
            "h01_kitchen_s001_c001",
            "h01_kitchen_s001_c002",
        ]

    @pytest.mark.unit
    def test_one_broken_manifest_fails_the_load(self, tmp_path: Path) -> None:
        """A suite with an unparseable member cannot be trusted green."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        (manifest_dir / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ClipError, match="readable JSON"):
            load_clip_manifests(tmp_path)

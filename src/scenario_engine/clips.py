"""
src.scenario_engine.clips — Scenario Clip Metadata and MP4 Sanitisation
=======================================================================

Scenario clips are short videos that turn a scenario into a *test*: this clip
should fire ``SC-KIT-001`` at CRITICAL within 30 s, or should fire nothing at
all.

Why MP4 needs its own stripper
------------------------------
``src/dataset/capture/exif.py`` strips EXIF from images, and
``docs/04_dataset_engineering/capture_annotation_runbook.md`` §4 requires
metadata removal at ingest. **Neither covers video.** MP4 is a different
container: location rides in the ``moov/udta`` ``©xyz`` atom, alongside creation
time and device model, and an EXIF stripper never touches it. A clip pushed to
the S3 remote with its GPS atom intact is a participant's home address in a
versioned bucket, and that bucket has versioning enabled — deleting the object
later does not remove prior versions.

So stripping is a **hard ingest gate with a read-back assertion**, not a
best-effort pass. If ffmpeg is unavailable the ingest fails rather than
proceeding unsanitised: a privacy control that silently degrades is not a
control.

Layering
--------
This module lives in ``src/scenario_engine`` rather than ``src/dataset/capture``
because scenario clips belong to the knowledge layer, and because
``src/scenario_engine`` may not import ``src.dataset`` (ADR-P6-04). It shares no
code with the image path; the containers genuinely have nothing in common.

ffmpeg is an optional external tool, handled the same way the annotation stack
handles heavy optional dependencies (ADR-P5-11): absent by default, and the code
that needs it says so loudly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Container metadata keys that must not survive ingest. ``location`` and
#: ``com.apple.quicktime.location.ISO6709`` are the GPS carriers; the rest
#: identify the device or the moment of capture.
FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "location",
        "location-eng",
        "com.apple.quicktime.location.iso6709",
        "com.apple.quicktime.make",
        "com.apple.quicktime.model",
        "com.apple.quicktime.software",
        "com.apple.quicktime.creationdate",
        "creation_time",
        "date",
        "make",
        "model",
        "device",
        "encoder",
        "handler_name",
    }
)


class ClipError(ValueError):
    """Raised when a clip cannot be sanitised or verified."""


class FfmpegUnavailableError(ClipError):
    """Raised when ffmpeg/ffprobe are required but not installed.

    Deliberately an error rather than a skip: ingest must not proceed with
    unsanitised video.
    """


def ffmpeg_available() -> bool:
    """Is the ffmpeg toolchain present on PATH?"""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise FfmpegUnavailableError(
            "ffmpeg and ffprobe are required to ingest scenario clips, and were not "
            "found on PATH. Ingest is refused rather than proceeding with unsanitised "
            "video: MP4 carries GPS in the moov/udta '©xyz' atom, which the image "
            "EXIF stripper does not touch."
        )


def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version``, recorded for reproducibility.

    Pinned in the clip manifest the same way annotator weights are sha256-pinned:
    a sanitisation step is only reproducible if the tool that performed it is
    identified.
    """
    _require_ffmpeg()
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, timeout=30, check=True
    )
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Return the container-level metadata ffprobe reports for a file."""
    _require_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    payload = json.loads(result.stdout or "{}")

    tags: dict[str, Any] = {}
    tags.update(payload.get("format", {}).get("tags", {}) or {})
    for stream in payload.get("streams", []) or []:
        tags.update(stream.get("tags", {}) or {})
    return tags


def residual_metadata(tags: dict[str, Any]) -> dict[str, Any]:
    """Any forbidden keys still present, matched case-insensitively.

    Split out from the I/O so the policy itself is unit-testable without ffmpeg
    installed — the check is the part worth testing, not the subprocess call.
    """
    return {
        key: value for key, value in tags.items() if key.strip().lower() in FORBIDDEN_METADATA_KEYS
    }


def strip_metadata(source: Path | str, destination: Path | str) -> dict[str, Any]:
    """Re-mux a clip with all container metadata removed, then verify it.

    Uses ``-c copy`` so the video is never re-encoded: stripping metadata must
    not silently degrade the footage a scenario is validated against.

    Args:
        source:      Clip as captured.
        destination: Sanitised output. Parent directories are created.

    Returns:
        An attestation dict for the clip manifest: the ffmpeg version used, and
        the confirmation that a read-back found nothing.

    Raises:
        FfmpegUnavailableError: If the toolchain is missing.
        ClipError: If ffmpeg fails, or if forbidden metadata survives the strip.
    """
    _require_ffmpeg()
    src, dst = Path(source), Path(destination)
    if not src.exists():
        raise ClipError(f"Clip not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c",
            "copy",
            str(dst),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise ClipError(
            f"ffmpeg failed to sanitise {src} (exit {result.returncode}): "
            f"{result.stderr.strip()[-400:]}"
        )

    # Read-back assertion. Trusting -map_metadata to have worked is exactly the
    # unfalsifiable-gate pattern this project has been bitten by before.
    surviving = residual_metadata(read_metadata(dst))
    if surviving:
        raise ClipError(
            f"{dst} still carries metadata after stripping: {sorted(surviving)}. "
            f"Refusing to accept the clip — this is the field that would put a "
            f"participant's home address in a versioned bucket."
        )

    return {
        "metadata_stripped": True,
        "ffmpeg_version": ffmpeg_version(),
        "verified_by_readback": True,
    }


# ─── Clip manifest ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpectedOutcome:
    """What a clip asserts about the scenario engine.

    ``first_alert_within_s`` is what turns a clip into a test rather than a
    recording, and maps onto the response budgets already documented in
    ``docs/01_executive_implementation_plan/validation_strategy.md``.
    """

    fires: bool
    scenario_id: str | None = None
    severity: str | None = None
    first_alert_within_s: float | None = None

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> ExpectedOutcome:
        if not isinstance(raw, dict):
            raise ClipError(f"{where} must be a mapping, got {type(raw).__name__}")
        fires = bool(raw.get("fires", False))
        scenario_id = raw.get("scenario_id")
        if fires and not scenario_id:
            raise ClipError(f"{where}: a positive clip must name the scenario_id it expects")
        if fires and raw.get("first_alert_within_s") is None:
            raise ClipError(
                f"{where}: a positive clip must state first_alert_within_s — without a "
                f"deadline the clip records behaviour rather than asserting it"
            )
        return cls(
            fires=fires,
            scenario_id=str(scenario_id) if scenario_id else None,
            severity=str(raw["severity"]) if raw.get("severity") else None,
            first_alert_within_s=(
                float(raw["first_alert_within_s"])
                if raw.get("first_alert_within_s") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ClipManifest:
    """One scenario clip and everything needed to trust it."""

    clip_id: str
    session_id: str
    house_id: str
    room: str
    lighting: str
    polarity: str  # positive | negative
    scenario_id: str
    consent_reference: str
    sha256: str
    duration_s: float
    fps: float
    expected: ExpectedOutcome
    rule_hash_at_label_time: str
    negative_kind: str = ""
    metadata_stripped: bool = False
    ffmpeg_version: str = ""
    annotator: str = ""
    annotated_at: str = ""
    notes: str = ""
    ground_truth_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    VALID_POLARITIES = ("positive", "negative")
    #: Why a negative clip exists. `confuser` targets a known false-positive
    #: source (shiny floor read as wet); `absence` is the plain no-hazard case;
    #: `assistive` is an object present that must NOT alarm (a walking stick);
    #: `out_of_taxonomy` is a detector-level hard negative (a phone in hand).
    VALID_NEGATIVE_KINDS = ("confuser", "absence", "assistive", "out_of_taxonomy")

    @classmethod
    def from_mapping(cls, raw: Any, where: str = "clip") -> ClipManifest:
        if not isinstance(raw, dict):
            raise ClipError(f"{where} must be a mapping, got {type(raw).__name__}")

        def _required(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ClipError(f"{where}.{key} is required")
            return value.strip()

        polarity = _required("polarity")
        if polarity not in cls.VALID_POLARITIES:
            raise ClipError(
                f"{where}.polarity must be one of {list(cls.VALID_POLARITIES)}, got {polarity!r}"
            )

        negative_kind = str(raw.get("negative_kind", "") or "")
        if polarity == "negative":
            if not negative_kind:
                raise ClipError(
                    f"{where}.negative_kind is required for a negative clip — a negative "
                    f"without a stated reason cannot be reviewed for coverage"
                )
            if negative_kind not in cls.VALID_NEGATIVE_KINDS:
                raise ClipError(
                    f"{where}.negative_kind must be one of "
                    f"{list(cls.VALID_NEGATIVE_KINDS)}, got {negative_kind!r}"
                )

        expected = ExpectedOutcome.from_mapping(raw.get("expected", {}), f"{where}.expected")
        if polarity == "negative" and expected.fires:
            raise ClipError(f"{where}: a negative clip cannot expect an alert to fire")
        if polarity == "positive" and not expected.fires:
            raise ClipError(f"{where}: a positive clip must expect an alert to fire")

        return cls(
            clip_id=_required("clip_id"),
            session_id=_required("session_id"),
            house_id=_required("house_id"),
            room=_required("room"),
            lighting=_required("lighting"),
            polarity=polarity,
            scenario_id=_required("scenario_id"),
            consent_reference=_required("consent_reference"),
            sha256=_required("sha256"),
            duration_s=float(raw.get("duration_s", 0.0)),
            fps=float(raw.get("fps", 0.0)),
            expected=expected,
            rule_hash_at_label_time=_required("rule_hash_at_label_time"),
            negative_kind=negative_kind,
            metadata_stripped=bool(raw.get("metadata_stripped", False)),
            ffmpeg_version=str(raw.get("ffmpeg_version", "")),
            annotator=str(raw.get("annotator", "")),
            annotated_at=str(raw.get("annotated_at", "")),
            notes=str(raw.get("notes", "")),
            ground_truth_events=tuple(raw.get("ground_truth_events", []) or []),
        )

    def frame_prefix(self) -> str:
        """Prefix for extracted frames.

        Chosen so ``src/utils/dataset_utils.py``'s existing group-pattern
        extractor recognises the clip as the leakage unit — frames from one clip
        can then never be split across train and val without any new code.
        """
        return f"{self.clip_id}_frame_"

    def is_stale(self, current_rule_hash: str) -> bool:
        """Was this clip labelled against behaviour that has since changed?

        Feeds gate RG-S3. A clip whose scenario's ``rule_hash`` has moved is
        asserting an expectation that no longer exists, and a suite passing
        green against a stale expectation is worse than one that fails.
        """
        return self.rule_hash_at_label_time != current_rule_hash

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
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config_helpers import load_yaml

logger = logging.getLogger(__name__)

#: The ``clips:`` block lives beside ``capture:`` in the same file, so the
#: video protocol is read next to the image protocol it extends.
DEFAULT_CAPTURE_CONFIG = Path("configs/capture_config.yaml")

#: Mirrors ``consent.registry_path`` / ``consent.reference_pattern``. Duplicated
#: rather than imported from ``src.dataset.capture.config`` because
#: ``src/scenario_engine`` is a leaf that may not import ``src.dataset``
#: (ADR-P6-04); both are stable published formats documented in
#: ``data/consent/README.md``.
DEFAULT_CONSENT_REGISTRY = Path("data/consent/consent_registry.yaml")
DEFAULT_CONSENT_PATTERN = r"^CONSENT-h\d{2}-\d{4}-\d{3}$"

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


def _ffprobe(path: Path | str) -> dict[str, Any]:
    """Raw ``ffprobe -show_format -show_streams`` payload for a file."""
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
    return payload if isinstance(payload, dict) else {}


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Return the container-level metadata ffprobe reports for a file."""
    payload = _ffprobe(path)

    tags: dict[str, Any] = {}
    tags.update(payload.get("format", {}).get("tags", {}) or {})
    for stream in payload.get("streams", []) or []:
        tags.update(stream.get("tags", {}) or {})
    return tags


def _parse_frame_rate(raw: Any) -> float:
    """Parse ffprobe's ``"30000/1001"`` rational frame-rate notation."""
    text = str(raw or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            den = float(denominator)
            return float(numerator) / den if den else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_media(path: Path | str) -> tuple[float, float]:
    """Return ``(duration_seconds, fps)`` for a clip's first video stream.

    Both feed intake gates: a clip shorter than a scenario's dwell threshold
    cannot exercise it, and a clip below the runtime's ``target_fps`` rescales
    every frame-counted temporal predicate it is supposed to be testing.

    Returns ``(0.0, 0.0)`` for values ffprobe cannot determine, so the caller's
    range checks — not this function — decide what is acceptable.
    """
    payload = _ffprobe(path)

    try:
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    fps = 0.0
    for stream in payload.get("streams", []) or []:
        if stream.get("codec_type") != "video":
            continue
        fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
            stream.get("r_frame_rate")
        )
        if not duration:
            try:
                duration = float(stream.get("duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
        break

    return duration, fps


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


# ─── Intake requirements ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipRequirements:
    """The ``clips:`` block of ``configs/capture_config.yaml``.

    Parsed here rather than by :mod:`src.dataset.capture.config` because
    ``src/scenario_engine`` may not import ``src.dataset`` (ADR-P6-04). That
    loader ignores unknown root keys, so the two coexist in one file without
    either needing to know about the other.
    """

    inbox_dir: Path = Path("data/clip_inbox")
    clips_root: Path = Path("data/scenario_engine/clips")
    clip_id_pattern: str = r"^h\d{2}_[a-z_]+_s\d{3}_c\d{3}$"
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov")
    min_duration_s: float = 10.0
    max_duration_s: float = 120.0
    min_fps: float = 15.0
    max_file_mb: int = 500
    strip_metadata: bool = True
    consent_scope: str = "scenario-video"
    min_negative_fraction: float = 0.30

    def validate_clip_id(self, clip_id: str) -> list[str]:
        """Check a clip ID against the grammar. Returns problems (empty = ok)."""
        if not re.match(self.clip_id_pattern, clip_id):
            return [f"clip id '{clip_id}' does not match pattern {self.clip_id_pattern}"]
        return []

    def check_file(self, path: Path) -> list[str]:
        """Extension and size gates, applied before ffprobe is spent on the file."""
        problems: list[str] = []
        if path.suffix.lower() not in self.allowed_extensions:
            problems.append(
                f"{path.name}: extension '{path.suffix}' not in " f"{list(self.allowed_extensions)}"
            )
        if not path.exists():
            problems.append(f"{path}: file not found")
            return problems
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_mb:
            problems.append(f"{path.name}: {size_mb:.0f} MB exceeds max_file_mb {self.max_file_mb}")
        return problems

    def check_media(self, duration_s: float, fps: float) -> list[str]:
        """Duration and frame-rate gates.

        A clip below ``min_fps`` is rejected rather than warned about: the
        temporal predicates it exists to test are counted in frames against the
        runtime's ``target_fps``, so a 10 fps clip silently doubles every dwell
        it is meant to be checking.
        """
        problems: list[str] = []
        if duration_s <= 0:
            problems.append("duration could not be determined by ffprobe")
        elif duration_s < self.min_duration_s:
            problems.append(
                f"duration {duration_s:.1f}s is under min_duration_s "
                f"{self.min_duration_s}s — too short to exercise a dwell threshold"
            )
        elif duration_s > self.max_duration_s:
            problems.append(
                f"duration {duration_s:.1f}s exceeds max_duration_s {self.max_duration_s}s"
            )
        if fps <= 0:
            problems.append("frame rate could not be determined by ffprobe")
        elif fps < self.min_fps:
            problems.append(
                f"frame rate {fps:.1f} is under min_fps {self.min_fps} — frame-counted "
                f"temporal predicates would be rescaled relative to the runtime"
            )
        return problems


def load_clip_requirements(path: Path | str = DEFAULT_CAPTURE_CONFIG) -> ClipRequirements:
    """Load the ``clips:`` block, falling back to built-in defaults.

    A missing file or absent block is not an error — the defaults keep unit
    tests and minimal checkouts runnable, matching
    :func:`src.dataset.capture.config.load_capture_config`.

    Raises:
        ClipError: If a configured value is unusable (bad regex, empty
            extension list, inverted duration bounds).
    """
    config_path = Path(path)
    defaults = ClipRequirements()

    try:
        raw = load_yaml(config_path)
    except FileNotFoundError:
        logger.warning(f"Capture config not found at {config_path} — using built-in clip defaults")
        return defaults

    clips_raw = raw.get("clips", {}) or {}
    video_raw = clips_raw.get("video", {}) or {}

    extensions = tuple(
        ext if str(ext).startswith(".") else f".{ext}"
        for ext in (
            str(e).lower().strip()
            for e in (video_raw.get("allowed_extensions") or list(defaults.allowed_extensions))
        )
    )

    requirements = ClipRequirements(
        inbox_dir=Path(clips_raw.get("inbox_dir", defaults.inbox_dir)),
        clips_root=Path(clips_raw.get("clips_root", defaults.clips_root)),
        clip_id_pattern=str(clips_raw.get("clip_id_pattern", defaults.clip_id_pattern)),
        allowed_extensions=extensions,
        min_duration_s=float(video_raw.get("min_duration_s", defaults.min_duration_s)),
        max_duration_s=float(video_raw.get("max_duration_s", defaults.max_duration_s)),
        min_fps=float(video_raw.get("min_fps", defaults.min_fps)),
        max_file_mb=int(video_raw.get("max_file_mb", defaults.max_file_mb)),
        strip_metadata=bool(video_raw.get("strip_metadata", defaults.strip_metadata)),
        consent_scope=str(clips_raw.get("consent_scope", defaults.consent_scope)),
        min_negative_fraction=float(
            clips_raw.get("min_negative_fraction", defaults.min_negative_fraction)
        ),
    )

    try:
        re.compile(requirements.clip_id_pattern)
    except re.error as exc:
        raise ClipError(f"Invalid clips.clip_id_pattern regex in {config_path}: {exc}") from exc
    if not requirements.allowed_extensions:
        raise ClipError(f"clips.video.allowed_extensions must be non-empty in {config_path}")
    if requirements.min_duration_s >= requirements.max_duration_s:
        raise ClipError(f"clips.video.min_duration_s must be below max_duration_s in {config_path}")
    if not 0.0 <= requirements.min_negative_fraction <= 1.0:
        raise ClipError(f"clips.min_negative_fraction must be in [0, 1] in {config_path}")
    if not requirements.strip_metadata:
        raise ClipError(
            f"clips.video.strip_metadata cannot be disabled in {config_path}. MP4 carries "
            f"GPS in the moov/udta atom; ingesting unsanitised video would put a "
            f"participant's home address in a versioned S3 bucket, where deleting the "
            f"object does not remove prior versions."
        )
    return requirements


# ─── Consent ──────────────────────────────────────────────────────────────────


def verify_clip_consent(
    reference: str,
    house_id: str,
    requirements: ClipRequirements,
    registry_path: Path | str = DEFAULT_CONSENT_REGISTRY,
    reference_pattern: str = DEFAULT_CONSENT_PATTERN,
) -> list[str]:
    """Verify a clip's consent reference, including its **scope**.

    Deliberately separate from :func:`src.dataset.capture.consent.verify_consent`
    rather than importing it, for two reasons. The layering rule forbids the
    import (ADR-P6-04); and the behaviour genuinely differs in two ways that
    matter:

    1. **Scope is checked.** Image ingest ignores ``scope`` entirely. A 30-second
       clip cannot be curated frame-by-frame to avoid faces the way a still can,
       so ``dataset-training`` consent does not cover it.
    2. **A missing registry is fatal, not a warning.** Image ingest downgrades to
       a format-only check when the registry is absent. Scope cannot be checked
       without the registry, so downgrading would silently drop the very control
       this function exists to apply — the same reason a missing ffmpeg is an
       error here rather than a skip.

    Returns:
        List of problems; empty means the reference is acceptable.
    """
    problems: list[str] = []
    if not reference:
        return ["consent reference is required for every scenario clip"]
    if not re.match(reference_pattern, reference):
        problems.append(
            f"consent reference '{reference}' does not match pattern {reference_pattern}"
        )

    try:
        raw = load_yaml(Path(registry_path))
    except FileNotFoundError:
        problems.append(
            f"no consent registry at {registry_path} — clip consent scope cannot be "
            f"verified without it, and scenario clips are not ingested on scope trust. "
            f"Ingest clips on the collection machine."
        )
        return problems

    if not isinstance(raw, dict):
        raise ClipError(
            f"Consent registry {registry_path} must be a mapping of consent_id -> record"
        )

    record = raw.get(reference)
    if not isinstance(record, dict):
        problems.append(f"consent reference '{reference}' not found in {registry_path}")
        return problems

    if bool(record.get("withdrawn", False)):
        problems.append(f"consent '{reference}' has been WITHDRAWN — do not ingest")
    if str(record.get("house_id", "")) != house_id:
        problems.append(
            f"consent '{reference}' covers house '{record.get('house_id')}', "
            f"but the clip belongs to '{house_id}'"
        )
    scope = str(record.get("scope", ""))
    if scope != requirements.consent_scope:
        problems.append(
            f"consent '{reference}' has scope '{scope}', but scenario clips require "
            f"'{requirements.consent_scope}'. Video consent is not implied by image "
            f"consent — re-consent the household before ingesting clips."
        )
    return problems


def parse_clip_id(clip_id: str) -> tuple[str, str, str]:
    """Split a clip ID into ``(house_id, room, session_id)``.

    The grammar is ``h{NN}_{room}_s{NNN}_c{NNN}``, i.e. the capture session
    grammar plus a clip counter, so house and room parse the same way they do
    for image sessions.

    Raises:
        ClipError: If the ID has too few ``_``-separated tokens.
    """
    parts = clip_id.split("_")
    if len(parts) < 4:
        raise ClipError(
            f"clip id '{clip_id}' is not of the form h{{NN}}_{{room}}_s{{NNN}}_c{{NNN}}"
        )
    return parts[0], "_".join(parts[1:-2]), "_".join(parts[:-1])


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

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, key-sorted for byte-stable manifests."""
        payload: dict[str, Any] = {
            "clip_id": self.clip_id,
            "session_id": self.session_id,
            "house_id": self.house_id,
            "room": self.room,
            "lighting": self.lighting,
            "polarity": self.polarity,
            "negative_kind": self.negative_kind,
            "scenario_id": self.scenario_id,
            "consent_reference": self.consent_reference,
            "sha256": self.sha256,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "rule_hash_at_label_time": self.rule_hash_at_label_time,
            "metadata_stripped": self.metadata_stripped,
            "ffmpeg_version": self.ffmpeg_version,
            "annotator": self.annotator,
            "annotated_at": self.annotated_at,
            "notes": self.notes,
            "ground_truth_events": list(self.ground_truth_events),
            "expected": {
                "fires": self.expected.fires,
                "scenario_id": self.expected.scenario_id,
                "severity": self.expected.severity,
                "first_alert_within_s": self.expected.first_alert_within_s,
            },
        }
        return dict(sorted(payload.items()))


# ─── Clip sets ────────────────────────────────────────────────────────────────


def load_clip_manifests(clips_root: Path | str) -> list[ClipManifest]:
    """Load every ``manifests/*.json`` under a clips root, sorted by clip id.

    Raises:
        ClipError: If any manifest is unreadable or invalid. A clip suite with
            one unparseable member is not a suite that can be trusted green.
    """
    manifest_dir = Path(clips_root) / "manifests"
    if not manifest_dir.exists():
        return []

    manifests: list[ClipManifest] = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClipError(f"{path} is not readable JSON: {exc}") from exc
        manifests.append(ClipManifest.from_mapping(raw, where=str(path)))
    return sorted(manifests, key=lambda clip: clip.clip_id)


def validate_clip_set(
    clips: list[ClipManifest],
    requirements: ClipRequirements,
    rule_hashes: dict[str, str] | None = None,
) -> list[str]:
    """Set-level checks that no single clip can satisfy on its own.

    Individual manifests are validated at construction. What only the set can
    answer is whether the suite is *honest*: enough negatives to measure the
    false-positive rate, no duplicate ids, no scenario asserted against a
    ``rule_hash`` that has since moved.

    Args:
        clips:        The loaded manifests.
        requirements: Intake requirements carrying ``min_negative_fraction``.
        rule_hashes:  Optional ``scenario_id -> rule_hash`` from the compiled
                      artifact. When given, stale clips are reported.

    Returns:
        List of problems; empty means the set is acceptable.
    """
    problems: list[str] = []
    if not clips:
        return problems

    seen: set[str] = set()
    for clip in clips:
        if clip.clip_id in seen:
            problems.append(f"duplicate clip id '{clip.clip_id}'")
        seen.add(clip.clip_id)
        problems.extend(requirements.validate_clip_id(clip.clip_id))
        if requirements.strip_metadata and not clip.metadata_stripped:
            problems.append(
                f"{clip.clip_id}: metadata_stripped is false — the clip was not "
                f"sanitised, or the manifest was hand-edited"
            )

    negatives = sum(1 for clip in clips if clip.polarity == "negative")
    fraction = negatives / len(clips)
    if fraction < requirements.min_negative_fraction:
        problems.append(
            f"only {negatives}/{len(clips)} clips ({fraction:.0%}) are negatives, under "
            f"min_negative_fraction {requirements.min_negative_fraction:.0%} — a suite of "
            f"positives measures sensitivity and is blind to the false-positive rate"
        )

    if rule_hashes is not None:
        for clip in clips:
            current = rule_hashes.get(clip.scenario_id)
            if current is None:
                problems.append(
                    f"{clip.clip_id}: scenario '{clip.scenario_id}' is not in the "
                    f"compiled artifact"
                )
            elif clip.is_stale(current):
                problems.append(
                    f"{clip.clip_id}: labelled against rule_hash "
                    f"{clip.rule_hash_at_label_time} but '{clip.scenario_id}' is now "
                    f"{current} — re-review the clip before trusting its expectation"
                )
    return problems

"""
scripts.scenarios.32_ingest_scenario_clips — Scenario Clip Ingest CLI
=====================================================================

Ingests one scenario clip from the staging inbox into
``data/scenario_engine/clips``: consent-scope verification, intake gates,
**MP4 metadata stripping with a read-back assertion**, and a manifest that
turns the clip into a test (``expected.first_alert_within_s``).

Usage:
    # Ingest a positive clip
    python scripts/scenarios/32_ingest_scenario_clips.py \\
        --clip-id h01_kitchen_s001_c001 --source kitchen_unattended.mp4 \\
        --scenario-id SC-KIT-001 --lighting evening \\
        --consent-ref CONSENT-h01-2026-001 \\
        --expect fires --severity CRITICAL --first-alert-within 30

    # Ingest a negative (the shiny-floor confuser from validation_strategy.md)
    python scripts/scenarios/32_ingest_scenario_clips.py \\
        --clip-id h01_hall_s002_c003 --source shiny_floor.mp4 \\
        --scenario-id SC-BTH-001 --lighting daylight \\
        --consent-ref CONSENT-h01-2026-001 \\
        --expect no-alert --negative-kind confuser

    # Bootstrap / maintenance
    python scripts/scenarios/32_ingest_scenario_clips.py --init
    python scripts/scenarios/32_ingest_scenario_clips.py --verify-all

Exit codes: 0 = success, 1 = failure (consent, intake, sanitisation or
set-level verification).

DVC integration:
    ``--verify-all`` is the cmd of the **frozen** ``ingest_scenario_clips``
    stage — never auto-run, so ``dvc repro`` on a fresh machine can never
    overwrite human-collected footage with an empty re-run. Humans ingest, then
    ``dvc commit -f ingest_scenario_clips``. Mirrors ``ingest_custom_captures``.
    See docs/08_scenario_engineering/clip_capture_protocol.md and ADR-P6-10.

ffmpeg is required. A missing toolchain is an error, not a skip: a privacy
control that silently degrades is not a control.

Prints ASCII only — this box's console is cp1252 and the repo has a recorded
crash from Devanagari in log output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.clips import (
    ClipError,
    ClipManifest,
    ClipRequirements,
    ffmpeg_available,
    load_clip_manifests,
    load_clip_requirements,
    parse_clip_id,
    probe_media,
    strip_metadata,
    validate_clip_set,
    verify_clip_consent,
)
from src.scenario_engine.compile import DEFAULT_ARTIFACT
from src.utils.dataset_utils import compute_file_hash
from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TREE_README = """# data/scenario_engine/clips — Scenario Clips (DVC-tracked)

Ingested via `scripts/scenarios/32_ingest_scenario_clips.py`. Every clip here
has had its container metadata stripped and verified by read-back; every
manifest carries a `consent_reference` whose registry scope is `scenario-video`.

    video/      sanitised .mp4/.mov, named {clip_id}{ext}
    manifests/  one {clip_id}.json per clip -- the expectation it asserts

Do not add files by hand. Do not re-run the DVC stage to "rebuild" this
directory: it is frozen precisely because its contents cannot be regenerated.

Protocol: docs/08_scenario_engineering/clip_capture_protocol.md
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a scenario clip with metadata stripping and a manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--init", action="store_true", help="Create the clips tree and exit.")
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Re-verify every ingested manifest and the set as a whole, then exit.",
    )
    parser.add_argument("--config", type=Path, default=None, help="capture_config.yaml path.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--inbox", type=Path, default=None, help="Override clips.inbox_dir.")
    parser.add_argument("--clips-root", type=Path, default=None, help="Override clips.clips_root.")

    parser.add_argument("--clip-id", help="e.g. h01_kitchen_s001_c001")
    parser.add_argument("--source", help="Clip filename inside the inbox, or a full path.")
    parser.add_argument("--scenario-id", help="Scenario the clip exercises, e.g. SC-KIT-001.")
    parser.add_argument("--lighting", default="", help="daylight|tubelight|dim|night_flash|mixed")
    parser.add_argument("--consent-ref", default="", help="e.g. CONSENT-h01-2026-001")
    parser.add_argument(
        "--expect",
        choices=["fires", "no-alert"],
        help="What the clip asserts. 'fires' requires --first-alert-within.",
    )
    parser.add_argument("--severity", default="", help="Expected severity when --expect fires.")
    parser.add_argument(
        "--first-alert-within",
        type=float,
        default=None,
        help="Seconds within which the alert must appear (required for --expect fires).",
    )
    parser.add_argument(
        "--negative-kind",
        default="",
        help="confuser|absence|assistive|out_of_taxonomy|pre_dwell (required for --expect no-alert).",
    )
    parser.add_argument("--annotator", default="", help="Who reviewed the clip.")
    parser.add_argument("--notes", default="")

    # ── External clip provenance (ADR-P6-10 / SourceProvenance) ──────────────
    parser.add_argument(
        "--external",
        action="store_true",
        help="Mark this clip as external (third-party). Uses licence instead of consent.",
    )
    parser.add_argument("--source-url", default="", help="URL where the original was found.")
    parser.add_argument("--source-platform", default="", help="e.g. Pexels, iStock, Pixabay.")
    parser.add_argument("--creator", default="", help="Original creator/uploader.")
    parser.add_argument("--license", dest="clip_license", default="", help="e.g. Pexels-License.")
    parser.add_argument("--license-url", default="", help="URL to the licence terms.")
    parser.add_argument("--download-date", default="", help="ISO date the clip was downloaded.")
    parser.add_argument("--original-video-id", default="", help="Original platform video ID.")
    parser.add_argument(
        "--indian-home",
        action="store_true",
        default=True,
        help="Whether the clip depicts an Indian home environment.",
    )
    parser.add_argument(
        "--no-indian-home",
        action="store_false",
        dest="indian_home",
        help="Mark clip as NOT depicting an Indian home (e.g. stock footage).",
    )
    return parser.parse_args(argv)


def _resolve(args: argparse.Namespace) -> ClipRequirements:
    """Load requirements and apply CLI overrides (CLI beats YAML, house pattern)."""
    config_path = args.config
    requirements = load_clip_requirements(config_path) if config_path else load_clip_requirements()
    if args.inbox is not None:
        requirements = replace(requirements, inbox_dir=args.inbox)
    if args.clips_root is not None:
        requirements = replace(requirements, clips_root=args.clips_root)
    return requirements


def _rule_hashes(artifact_path: Path) -> dict[str, str]:
    """Map ``scenario_id -> rule_hash`` from the compiled artifact.

    Returns an empty map when the artifact is absent, so ``--verify-all``
    degrades to structural checks rather than crashing on a fresh checkout.
    """
    if not artifact_path.exists():
        logger.warning(f"No compiled artifact at {artifact_path} -- skipping staleness checks")
        return {}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for group in ("scenarios", "rejected", "deprecated"):
        for scenario in artifact.get(group, []) or []:
            if isinstance(scenario, dict) and scenario.get("scenario_id"):
                hashes[str(scenario["scenario_id"])] = str(scenario.get("rule_hash", ""))
    return hashes


def init_tree(requirements: ClipRequirements) -> None:
    """Create the clips tree. Idempotent."""
    root = requirements.clips_root
    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    requirements.inbox_dir.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(_TREE_README, encoding="utf-8", newline="\n")
    logger.info(f"Clips tree ready at {root} (inbox: {requirements.inbox_dir})")


def verify_all(requirements: ClipRequirements, artifact_path: Path) -> int:
    """Re-verify every ingested manifest plus the set-level invariants."""
    try:
        clips = load_clip_manifests(requirements.clips_root)
    except ClipError as exc:
        logger.error(str(exc))
        return 1

    if not clips:
        logger.info(
            f"No clips ingested under {requirements.clips_root}. Nothing to verify -- "
            f"this is the expected state before the first capture session."
        )
        return 0

    problems = validate_clip_set(clips, requirements, _rule_hashes(artifact_path))

    # The manifest asserts the video is present and unmodified; check it.
    for clip in clips:
        matches = list((requirements.clips_root / "video").glob(f"{clip.clip_id}.*"))
        if not matches:
            problems.append(f"{clip.clip_id}: manifest exists but no video file was found")
            continue
        digest = compute_file_hash(matches[0])
        if digest != clip.sha256:
            problems.append(
                f"{clip.clip_id}: {matches[0].name} hashes to {digest[:12]}... but the "
                f"manifest records {clip.sha256[:12]}... -- the file changed after ingest"
            )

    for problem in problems:
        logger.error(problem)
    negatives = sum(1 for clip in clips if clip.polarity == "negative")
    logger.info(
        f"Verified {len(clips)} clip(s) ({negatives} negative) under "
        f"{requirements.clips_root}: {len(problems)} problem(s)."
    )
    return 1 if problems else 0


def _build_expected(args: argparse.Namespace) -> dict[str, Any]:
    if args.expect == "fires":
        return {
            "fires": True,
            "scenario_id": args.scenario_id,
            "severity": args.severity or None,
            "first_alert_within_s": args.first_alert_within,
        }
    return {"fires": False}


def ingest(args: argparse.Namespace, requirements: ClipRequirements, artifact_path: Path) -> int:
    """Validate, sanitise and record a single clip."""
    is_external = getattr(args, "external", False)

    # Core required arguments.
    required_pairs: list[tuple[str, Any]] = [
        ("--clip-id", args.clip_id),
        ("--source", args.source),
        ("--scenario-id", args.scenario_id),
        ("--expect", args.expect),
    ]
    # Consent is required only for own-capture clips.
    if not is_external:
        required_pairs.append(("--consent-ref", args.consent_ref))

    missing = [name for name, value in required_pairs if not value]
    if missing:
        logger.error(f"Missing required argument(s): {', '.join(missing)}")
        return 1

    # Cheap structural checks first, so a typo'd clip id is reported as a typo
    # even on a machine without the toolchain.
    problems = requirements.validate_clip_id(args.clip_id)
    if problems:
        for problem in problems:
            logger.error(problem)
        return 1

    manifest_path = requirements.clips_root / "manifests" / f"{args.clip_id}.json"
    if manifest_path.exists():
        logger.error(
            f"{args.clip_id} is already ingested ({manifest_path}). Refusing to overwrite: "
            f"a clip id is what a test result refers to, so silently replacing the footage "
            f"behind one changes what a green suite means. Delete the manifest and the "
            f"video deliberately, or use the next clip number."
        )
        return 1

    if not ffmpeg_available():
        logger.error(
            "ffmpeg/ffprobe not found on PATH. Clip ingest is refused rather than "
            "proceeding with unsanitised video -- MP4 carries GPS in the moov/udta "
            "atom, which the image EXIF stripper does not touch."
        )
        return 1

    house_id, room, session_id = parse_clip_id(args.clip_id)

    if not is_external:
        consent_problems = verify_clip_consent(args.consent_ref, house_id, requirements)
        if consent_problems:
            for problem in consent_problems:
                logger.error(problem)
            return 1

    source = Path(args.source)
    if not source.exists():
        source = requirements.inbox_dir / args.source
    problems = requirements.check_file(source)
    if problems:
        for problem in problems:
            logger.error(problem)
        return 1

    duration_s, fps = probe_media(source)
    problems = requirements.check_media(duration_s, fps)
    if problems:
        for problem in problems:
            logger.error(f"{args.clip_id}: {problem}")
        return 1

    init_tree(requirements)
    destination = requirements.clips_root / "video" / f"{args.clip_id}{source.suffix.lower()}"
    try:
        attestation = strip_metadata(source, destination)
    except ClipError as exc:
        logger.error(str(exc))
        return 1

    rule_hash = _rule_hashes(artifact_path).get(args.scenario_id, "")
    if not rule_hash:
        logger.error(
            f"Scenario '{args.scenario_id}' has no rule_hash in {artifact_path}. A clip "
            f"without one cannot be detected as stale when the scenario changes; "
            f"recompile (30_compile_scenarios.py) or check the scenario id."
        )
        destination.unlink(missing_ok=True)
        return 1

    raw: dict[str, Any] = {
        "clip_id": args.clip_id,
        "session_id": session_id,
        "house_id": house_id,
        "room": room,
        "lighting": args.lighting,
        "polarity": "positive" if args.expect == "fires" else "negative",
        "negative_kind": args.negative_kind,
        "scenario_id": args.scenario_id,
        "consent_reference": args.consent_ref if not is_external else "",
        "sha256": compute_file_hash(destination),
        "duration_s": round(duration_s, 3),
        "fps": round(fps, 3),
        "rule_hash_at_label_time": rule_hash,
        "expected": _build_expected(args),
        "annotator": args.annotator,
        "annotated_at": timestamp_str(),
        "notes": args.notes,
        "metadata_stripped": bool(attestation["metadata_stripped"]),
        "ffmpeg_version": str(attestation["ffmpeg_version"]),
        "indian_home": getattr(args, "indian_home", True),
    }

    if is_external:
        raw["provenance"] = {
            "source_type": "external",
            "source_url": getattr(args, "source_url", "") or "",
            "source_platform": getattr(args, "source_platform", "") or "",
            "creator": getattr(args, "creator", "") or "",
            "license": getattr(args, "clip_license", "") or "",
            "license_url": getattr(args, "license_url", "") or "",
            "download_date": getattr(args, "download_date", "") or "",
            "original_video_id": getattr(args, "original_video_id", "") or "",
        }

    try:
        manifest = ClipManifest.from_mapping(raw, where=args.clip_id)
    except ClipError as exc:
        logger.error(str(exc))
        destination.unlink(missing_ok=True)
        return 1

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    logger.info(
        f"Ingested {args.clip_id}: {duration_s:.1f}s @ {fps:.1f} fps, metadata stripped "
        f"and verified by read-back, manifest at {manifest_path}"
    )
    logger.info("Record it in DVC with: dvc commit -f ingest_scenario_clips")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        requirements = _resolve(args)
    except ClipError as exc:
        logger.error(str(exc))
        return 1

    artifact_path = Path(args.artifact)

    if args.init:
        init_tree(requirements)
        return 0
    if args.verify_all:
        return verify_all(requirements, artifact_path)
    return ingest(args, requirements, artifact_path)


if __name__ == "__main__":
    raise SystemExit(main())

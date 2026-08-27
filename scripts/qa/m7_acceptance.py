"""
scripts.qa.m7_acceptance — M7 Real-World Ingest Acceptance Test
===============================================================

Proves the M7 clip ingest path works against a **real MP4 carrying real GPS**,
on the machine that will be used for collection.

Why this exists as a script rather than a unit test
---------------------------------------------------
The unit suite tests the sanitisation *policy* without ffmpeg, because that is
what CI can run. It cannot prove the thing that actually matters: that ffmpeg
removes what we claim it removes, and that our read-back would notice if it
did not. That requires a real container, a real GPS atom, and a real toolchain.

The test is deliberately structured so it **cannot pass vacuously**:

  A1  ffmpeg/ffprobe are present and report versions.
  A2  A fixture clip is built WITH GPS, device make/model and creation time.
  A3  ffprobe CONFIRMS those keys are present before stripping. Without this
      step the whole test would pass on a file that never had metadata --
      the classic unfalsifiable gate.
  A4  The full ingest CLI runs: consent scope -> intake gates -> strip ->
      read-back -> manifest.
  A5  ffprobe CONFIRMS the forbidden keys are gone from the ingested file.
  A6  The residual detector is shown to still FIRE on the pre-strip file, so
      A5 is a real result and not a broken detector returning empty.
  A7  An unsanitised clip cannot enter by the production path: ingest with a
      dataset-training consent scope is refused.
  A8  The manifest reloads and the recorded sha256 matches the file on disk.

Nothing here touches the real dataset. The fixture is synthetic video and gets
its own temporary clips root and its own consent registry, so no fabricated
consent record and no non-scenario clip can leak into the collection.

Usage:
    python scripts/qa/m7_acceptance.py
    python scripts/qa/m7_acceptance.py --keep    # leave the workspace for inspection

Exit codes: 0 all checks passed, 1 at least one failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.clips import (
    ClipManifest,
    device_identifying_handlers,
    ffmpeg_available,
    ffmpeg_version,
    read_metadata,
    residual_metadata,
)
from src.utils.dataset_utils import compute_file_hash
from src.utils.report_utils import git_commit_short, save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("data/qa_reports/m7_acceptance_report.json")
INGEST_CLI = Path("scripts/scenarios/32_ingest_scenario_clips.py")

#: Bengaluru. A plausible collection location, injected so the strip has
#: something real to remove.
FIXTURE_GPS = "+12.9716+077.5946/"
CLIP_ID = "h99_kitchen_s001_c001"
CONSENT_ID = "CONSENT-h99-2026-001"


@dataclass
class Check:
    """One acceptance check."""

    check_id: str
    description: str
    status: str = "pending"
    detail: str = ""

    def passed(self, detail: str = "") -> Check:
        self.status, self.detail = "pass", detail
        return self

    def failed(self, detail: str) -> Check:
        self.status, self.detail = "fail", detail
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "status": self.status,
            "detail": self.detail,
        }


def build_fixture_clip(destination: Path, seconds: int = 15) -> None:
    """Render a synthetic clip carrying the metadata a real phone would attach.

    ``testsrc2`` is a generated pattern, so the fixture contains no person, no
    household and nothing private -- only the container metadata this test is
    about.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x480:rate=30:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            f"location={FIXTURE_GPS}",
            "-metadata",
            f"location-eng={FIXTURE_GPS}",
            "-metadata",
            "make=TestPhone",
            "-metadata",
            "model=TestPhone 15 Pro",
            "-metadata",
            "date=2026-08-11",
            "-metadata",
            "creation_time=2026-08-11T10:00:00Z",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fixture build failed: {result.stderr[-500:]}")


def write_workspace(root: Path) -> tuple[Path, Path]:
    """Write an isolated capture config and consent registry.

    The registry is a **fixture for a synthetic clip**, kept out of
    ``data/consent/`` so that no consent record is ever fabricated for a real
    household, and house id ``h99`` is reserved so it cannot collide with a
    collected session.
    """
    registry = root / "consent_registry.yaml"
    registry.write_text(
        f"{CONSENT_ID}:\n"
        f"  house_id: h99\n"
        f'  granted_on: "2026-08-11"\n'
        f"  scope: scenario-video\n"
        f"  withdrawn: false\n"
        f"CONSENT-h99-2026-002:\n"
        f"  house_id: h99\n"
        f'  granted_on: "2026-08-11"\n'
        f"  scope: dataset-training\n"
        f"  withdrawn: false\n",
        encoding="utf-8",
    )

    config = root / "capture_config.yaml"
    config.write_text(
        f"consent:\n"
        f"  registry_path: {registry.as_posix()}\n"
        f'  reference_pattern: "^CONSENT-h\\\\d{{2}}-\\\\d{{4}}-\\\\d{{3}}$"\n'
        f"clips:\n"
        f"  inbox_dir: {(root / 'inbox').as_posix()}\n"
        f"  clips_root: {(root / 'clips').as_posix()}\n"
        f'  clip_id_pattern: "^h\\\\d{{2}}_[a-z_]+_s\\\\d{{3}}_c\\\\d{{3}}$"\n'
        f"  consent_scope: scenario-video\n"
        f"  video:\n"
        f"    allowed_extensions: [.mp4, .mov]\n"
        f"    min_duration_s: 10\n"
        f"    max_duration_s: 120\n"
        f"    min_fps: 15\n"
        f"    max_file_mb: 500\n"
        f"    strip_metadata: true\n",
        encoding="utf-8",
    )
    return config, registry


def run_ingest(config: Path, artifact: Path, source: Path, consent_ref: str) -> tuple[int, str]:
    """Invoke the real ingest CLI as a subprocess -- the production entry point."""
    result = subprocess.run(
        [
            sys.executable,
            str(INGEST_CLI),
            "--config",
            str(config),
            "--artifact",
            str(artifact),
            "--clip-id",
            CLIP_ID,
            "--source",
            str(source),
            "--scenario-id",
            "SC-BTH-001",
            "--lighting",
            "daylight",
            "--consent-ref",
            consent_ref,
            "--expect",
            "no-alert",
            "--negative-kind",
            "absence",
            "--annotator",
            "m7-acceptance",
            "--notes",
            "Synthetic acceptance fixture. Not a dataset member.",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, (result.stdout + result.stderr)


def run(report_path: Path, artifact: Path, keep: bool) -> int:
    checks: list[Check] = []
    workspace = Path(tempfile.mkdtemp(prefix="m7_acceptance_"))
    logger.info(f"Acceptance workspace: {workspace}")

    try:
        # A1 -- toolchain
        a1 = Check("A1", "ffmpeg and ffprobe are installed on this machine")
        if not ffmpeg_available():
            checks.append(a1.failed("ffmpeg/ffprobe not found on PATH"))
            return _finish(checks, report_path, workspace, keep)
        version = ffmpeg_version()
        checks.append(a1.passed(version))
        logger.info(f"A1 PASS {version}")

        config, _ = write_workspace(workspace)
        raw_clip = workspace / "inbox" / "fixture.mp4"

        # A2 -- fixture with metadata
        a2 = Check("A2", "A fixture clip is built carrying GPS, device and timestamp metadata")
        try:
            build_fixture_clip(raw_clip)
            checks.append(a2.passed(f"{raw_clip.stat().st_size} bytes"))
        except RuntimeError as exc:
            checks.append(a2.failed(str(exc)))
            return _finish(checks, report_path, workspace, keep)
        logger.info("A2 PASS fixture built")

        # A3 -- the metadata is genuinely there BEFORE stripping.
        # Without this the whole test would pass on a file that never had any.
        a3 = Check("A3", "ffprobe confirms forbidden metadata is present BEFORE stripping")
        before_tags = read_metadata(raw_clip)
        before_residual = residual_metadata(before_tags)
        if not before_residual:
            checks.append(
                a3.failed(
                    f"fixture carries no forbidden metadata (tags={sorted(before_tags)}); "
                    f"the strip test would be vacuous"
                )
            )
            return _finish(checks, report_path, workspace, keep)
        checks.append(a3.passed(f"present before strip: {sorted(before_residual)}"))
        logger.info(f"A3 PASS pre-strip forbidden keys: {sorted(before_residual)}")

        # A7 (run early, before the id is consumed) -- wrong consent scope is refused
        a7 = Check("A7", "Ingest is refused when the consent scope is not scenario-video")
        code, output = run_ingest(config, artifact, raw_clip, "CONSENT-h99-2026-002")
        if code == 0:
            checks.append(a7.failed("ingest accepted a dataset-training consent"))
        elif "scenario-video" not in output:
            checks.append(a7.failed(f"refused, but not for the scope reason: {output[-300:]}"))
        else:
            checks.append(a7.passed("dataset-training consent refused with the scope message"))
        logger.info(f"A7 {checks[-1].status.upper()}")

        # A4 -- the real ingest path
        a4 = Check("A4", "Full ingest path succeeds: consent, gates, strip, read-back, manifest")
        code, output = run_ingest(config, artifact, raw_clip, CONSENT_ID)
        if code != 0:
            checks.append(a4.failed(f"exit {code}: {output[-800:]}"))
            return _finish(checks, report_path, workspace, keep)
        ingested = workspace / "clips" / "video" / f"{CLIP_ID}.mp4"
        manifest_path = workspace / "clips" / "manifests" / f"{CLIP_ID}.json"
        if not ingested.exists() or not manifest_path.exists():
            checks.append(a4.failed("ingest reported success but produced no video/manifest"))
            return _finish(checks, report_path, workspace, keep)
        checks.append(a4.passed(f"{ingested.name} + {manifest_path.name}"))
        logger.info("A4 PASS ingest completed")

        # A5 -- nothing forbidden survived
        a5 = Check("A5", "ffprobe confirms forbidden metadata is ABSENT after stripping")
        after_tags = read_metadata(ingested)
        after_residual = residual_metadata(after_tags)
        if after_residual:
            checks.append(a5.failed(f"survived: {sorted(after_residual)}"))
        else:
            checks.append(
                a5.passed(f"zero residual; remaining tags: {sorted(after_tags) or 'none'}")
            )
        logger.info(f"A5 {checks[-1].status.upper()} residual={sorted(after_residual)}")

        # A6 -- the detector still works, so A5 is a real negative
        a6 = Check("A6", "The residual detector still fires on the pre-strip file")
        recheck = residual_metadata(read_metadata(raw_clip))
        if recheck:
            checks.append(a6.passed(f"still detects {sorted(recheck)} on the original"))
        else:
            checks.append(a6.failed("detector no longer fires on the original -- A5 is vacuous"))
        logger.info(f"A6 {checks[-1].status.upper()}")

        # A9 -- the handler-name gate, which absence checking cannot cover
        a9 = Check("A9", "The surviving hdlr box carries a neutral handler name")
        handlers = device_identifying_handlers(after_tags)
        neutral = {k: v for k, v in after_tags.items() if k.strip().lower() == "handler_name"}
        if handlers:
            checks.append(a9.failed(f"device-identifying handler(s): {sorted(handlers.values())}"))
        elif not neutral:
            checks.append(a9.passed("no handler_name present at all"))
        else:
            checks.append(a9.passed(f"neutral handler(s): {sorted(set(neutral.values()))}"))
        logger.info(f"A9 {checks[-1].status.upper()}")

        # A10 -- the handler check is falsifiable
        a10 = Check("A10", "A device-chosen handler name would be rejected")
        if device_identifying_handlers({"handler_name": "Samsung Video Handler"}):
            checks.append(a10.passed("'Samsung Video Handler' is detected as device-identifying"))
        else:
            checks.append(a10.failed("the handler gate does not fire on a device handler name"))
        logger.info(f"A10 {checks[-1].status.upper()}")

        # A8 -- manifest integrity
        a8 = Check("A8", "Manifest reloads and its sha256 matches the ingested file")
        manifest = ClipManifest.from_mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")), where=str(manifest_path)
        )
        digest = compute_file_hash(ingested)
        if manifest.sha256 != digest:
            checks.append(a8.failed(f"manifest {manifest.sha256[:12]} vs file {digest[:12]}"))
        elif not manifest.metadata_stripped:
            checks.append(a8.failed("manifest does not record metadata_stripped"))
        else:
            checks.append(
                a8.passed(
                    f"sha256 {digest[:12]}..., {manifest.duration_s:.1f}s @ "
                    f"{manifest.fps:.1f} fps, ffmpeg={manifest.ffmpeg_version[:40]}"
                )
            )
        logger.info(f"A8 {checks[-1].status.upper()}")

        return _finish(checks, report_path, workspace, keep)
    finally:
        if not keep and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


def _finish(checks: list[Check], report_path: Path, workspace: Path, keep: bool) -> int:
    failed = [c for c in checks if c.status != "pass"]
    report = {
        "generated_at": timestamp_str(),
        "git_commit": git_commit_short(),
        "verdict": "PASS" if not failed else "FAIL",
        "ffmpeg_version": ffmpeg_version() if ffmpeg_available() else "not installed",
        "checks": [c.to_dict() for c in checks],
        "workspace_kept": str(workspace) if keep else None,
        "notes": [
            "The fixture is synthetic (ffmpeg testsrc2) and is NOT a dataset member. "
            "It is ingested into a temporary clips root with a temporary consent "
            "registry so that no consent record is fabricated for a real household.",
            "House id h99 is reserved for acceptance fixtures.",
            "A3 and A6 exist to keep A5 falsifiable: without them a passing strip "
            "check could mean the file never carried metadata, or that the residual "
            "detector stopped working.",
        ],
    }
    save_json_report(report, report_path)
    for check in checks:
        level = logger.info if check.status == "pass" else logger.error
        level(f"{check.check_id} [{check.status.upper()}] {check.description} -- {check.detail}")
    logger.info(f"M7 acceptance verdict: {report['verdict']}. Report: {report_path}")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M7 real-world ingest acceptance test.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/scenario_engine/build/scenarios.compiled.json"),
    )
    parser.add_argument("--keep", action="store_true", help="Keep the temporary workspace.")
    args = parser.parse_args(argv)
    return run(args.report, args.artifact, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())

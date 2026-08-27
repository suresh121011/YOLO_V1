"""
scripts.scenarios.35_batch_ingest_external — SC-KIT-001 External Clip Batch Ingest
===================================================================================

Renames, validates, and ingests the collected SC-KIT-001 external clips into
the repository's clip infrastructure. Uses the existing ``32_ingest_scenario_clips.py``
CLI with the ``--external`` flag.

This script:
1. Scans the raw collection directory for SC-KIT-001 clips
2. Assigns canonical clip_id names following repository grammar (h00_kitchen_sNNN_cNNN)
3. Copies clips to the inbox directory
4. Calls the ingest CLI for each clip
5. Produces a mapping file and summary report

Usage:
    python scripts/scenarios/35_batch_ingest_external.py
    python scripts/scenarios/35_batch_ingest_external.py --dry-run
    python scripts/scenarios/35_batch_ingest_external.py \\
        --raw-dir "SC-KIT-001 -- Kitchen Cooking Risk"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.clips import (
    load_clip_requirements,
    probe_media,
)
from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Clip categories and their session numbering ─────────────────────────────

# Session 001 = clips from the POSITIVE collection folder
# Session 002 = clips from the NEGATIVE collection folder
# Session 003 = clips from the EDGE collection folder
CATEGORY_SESSIONS: dict[str, tuple[str, str, str]] = {
    "POSITIVE": (
        "s001",
        "pre_dwell",
        "Collection folder: POSITIVE. Kitchen cooking composition "
        "present but clip too short for SC-KIT-001 temporal "
        "condition (needs 1080s).",
    ),
    "NEGATIVE": (
        "s002",
        "absence",
        "Collection folder: NEGATIVE. Required objects absent or " "wrong context for SC-KIT-001.",
    ),
    "EDGE": (
        "s003",
        "pre_dwell",
        "Collection folder: EDGE. Ambiguous/partial composition; "
        "temporal condition unmet regardless.",
    ),
}

# External house_id for stock footage (h99 is reserved for test fixtures).
EXTERNAL_HOUSE_ID = "h00"
ROOM = "kitchen"
SCENARIO_ID = "SC-KIT-001"

# Provenance defaults for the current batch.
# The licence fields below are placeholders — they must be populated with
# actual values before the dataset can be frozen. The SourceProvenance model
# will reject unknown licences at manifest construction.
DEFAULT_PROVENANCE: dict[str, str] = {
    "source_platform": "Pexels",
    "license": "Pexels-License",
    "license_url": "https://www.pexels.com/license/",
    "download_date": "2026-08-15",
}


def find_raw_dir(repo_root: Path) -> Path | None:
    """Locate the SC-KIT-001 raw collection directory (em-dash in name)."""
    for d in repo_root.iterdir():
        if d.is_dir() and "SC-KIT-001" in d.name:
            return d
    return None


def scan_clips(raw_dir: Path) -> list[dict[str, Any]]:
    """Scan raw clips and assign canonical clip_ids."""
    clips: list[dict[str, Any]] = []

    for category, (session, neg_kind, default_notes) in CATEGORY_SESSIONS.items():
        cat_dir = raw_dir / category
        if not cat_dir.exists():
            logger.warning(f"Category directory not found: {cat_dir}")
            continue

        video_files = sorted(
            f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in (".mp4", ".mov")
        )

        for idx, video_path in enumerate(video_files, start=1):
            clip_id = f"{EXTERNAL_HOUSE_ID}_{ROOM}_{session}_c{idx:03d}"

            # Probe media metadata.
            try:
                duration_s, fps = probe_media(video_path)
            except Exception as exc:
                logger.error(f"Failed to probe {video_path.name}: {exc}")
                duration_s, fps = 0.0, 0.0

            # Determine if this clip is from iStock or Pexels based on filename.
            name_lower = video_path.stem.lower()
            if "istockphoto" in name_lower or "istock" in name_lower:
                source_platform = "iStock"
                # iStock videos with specific IDs.
                video_id = name_lower.split("-")[1] if "-" in name_lower else name_lower
                source_url = f"https://www.istockphoto.com/video/{video_id}"
            elif "pexels" in name_lower:
                source_platform = "Pexels"
                video_id = name_lower.replace("pexels-", "")
                source_url = f"https://www.pexels.com/video/{video_id}"
            else:
                # Assume Pexels for numeric-only filenames (Pexels download pattern).
                source_platform = "Pexels"
                video_id = video_path.stem.split("_")[0].removesuffix("_medium")
                source_url = f"https://www.pexels.com/video/{video_id}"

            clips.append(
                {
                    "clip_id": clip_id,
                    "category": category,
                    "session": session,
                    "original_filename": video_path.name,
                    "original_path": str(video_path),
                    "duration_s": round(duration_s, 1),
                    "fps": round(fps, 1),
                    "negative_kind": neg_kind,
                    "notes": default_notes,
                    "source_platform": source_platform,
                    "source_url": source_url,
                    "original_video_id": video_id,
                    "size_mb": round(video_path.stat().st_size / (1024 * 1024), 1),
                }
            )

    return clips


def copy_to_inbox(clips: list[dict[str, Any]], inbox: Path) -> list[dict[str, Any]]:
    """Copy raw clips to the inbox with canonical names. Returns updated clip list."""
    inbox.mkdir(parents=True, exist_ok=True)
    copied = []
    for clip in clips:
        src = Path(clip["original_path"])
        dst = inbox / f"{clip['clip_id']}{src.suffix.lower()}"
        if dst.exists():
            logger.info(f"Already in inbox: {dst.name}")
        else:
            shutil.copy2(src, dst)
            logger.info(f"Copied {src.name} -> {dst.name}")
        clip["inbox_path"] = str(dst)
        copied.append(clip)
    return copied


def ingest_clip(clip: dict[str, Any], repo_root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Call the ingest CLI for one clip. Returns result dict."""
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "scenarios" / "32_ingest_scenario_clips.py"),
        "--clip-id",
        clip["clip_id"],
        "--source",
        clip["inbox_path"],
        "--scenario-id",
        SCENARIO_ID,
        "--lighting",
        "mixed",
        "--expect",
        "no-alert",
        "--negative-kind",
        clip["negative_kind"],
        "--annotator",
        "automated-batch",
        "--notes",
        clip["notes"],
        "--external",
        "--source-url",
        clip["source_url"],
        "--source-platform",
        clip["source_platform"],
        "--creator",
        clip.get("source_platform", "unknown"),
        "--license",
        DEFAULT_PROVENANCE.get("license", ""),
        "--license-url",
        DEFAULT_PROVENANCE.get("license_url", ""),
        "--download-date",
        DEFAULT_PROVENANCE.get("download_date", ""),
        "--original-video-id",
        clip.get("original_video_id", ""),
        "--no-indian-home",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would ingest: {clip['clip_id']} ({clip['original_filename']})")
        return {**clip, "result": "dry_run", "exit_code": 0}

    logger.info(f"Ingesting: {clip['clip_id']} ({clip['original_filename']})")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    if result.returncode != 0:
        # Capture the error output.
        error_msg = result.stderr.strip() or result.stdout.strip()
        logger.error(f"  FAILED: {clip['clip_id']}: {error_msg[-200:]}")
        return {**clip, "result": "failed", "exit_code": result.returncode, "error": error_msg}

    logger.info(f"  OK: {clip['clip_id']}")
    return {**clip, "result": "ingested", "exit_code": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch ingest SC-KIT-001 external clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Path to the raw SC-KIT-001 collection directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and map clips without actually ingesting."
    )
    parser.add_argument(
        "--skip-copy", action="store_true", help="Skip copy to inbox (clips already there)."
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]

    # Locate raw clips.
    raw_dir = args.raw_dir or find_raw_dir(repo_root)
    if raw_dir is None or not raw_dir.exists():
        logger.error(
            "SC-KIT-001 raw collection directory not found. " "Use --raw-dir to specify the path."
        )
        return 1

    logger.info(f"Raw directory: {raw_dir}")

    # Load clip requirements for inbox path.
    try:
        requirements = load_clip_requirements()
    except Exception as exc:
        logger.error(f"Failed to load clip requirements: {exc}")
        return 1

    # Step 1: Scan and map clips.
    clips = scan_clips(raw_dir)
    logger.info(f"Found {len(clips)} clips")

    # Gate: duration < min_duration.
    accepted = []
    rejected = []
    for clip in clips:
        if clip["duration_s"] < requirements.min_duration_s:
            logger.warning(
                f"REJECTED: {clip['clip_id']} ({clip['original_filename']}) -- "
                f"duration {clip['duration_s']}s < {requirements.min_duration_s}s minimum"
            )
            clip["result"] = "rejected_duration"
            rejected.append(clip)
        elif clip["fps"] < requirements.min_fps:
            logger.warning(
                f"REJECTED: {clip['clip_id']} ({clip['original_filename']}) -- "
                f"fps {clip['fps']} < {requirements.min_fps} minimum"
            )
            clip["result"] = "rejected_fps"
            rejected.append(clip)
        else:
            accepted.append(clip)

    logger.info(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    # Save the mapping file.
    mapping_path = requirements.inbox_dir / "clip_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_data = {
        "created_at": timestamp_str(),
        "scenario_id": SCENARIO_ID,
        "raw_directory": str(raw_dir),
        "total_scanned": len(clips),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "clips": clips,
    }
    mapping_path.write_text(
        json.dumps(mapping_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info(f"Mapping saved: {mapping_path}")

    if args.dry_run:
        logger.info("[DRY RUN] No files copied or ingested.")
        print(f"\n{'='*70}")
        print(f"DRY RUN SUMMARY: {len(accepted)} clips would be ingested")
        print(
            f"  POSITIVE folder (pre_dwell): "
            f"{sum(1 for c in accepted if c['category'] == 'POSITIVE')}"
        )
        print(
            f"  NEGATIVE folder (absence):   "
            f"{sum(1 for c in accepted if c['category'] == 'NEGATIVE')}"
        )
        print(
            f"  EDGE folder (pre_dwell):     "
            f"{sum(1 for c in accepted if c['category'] == 'EDGE')}"
        )
        print(f"  Rejected: {len(rejected)}")
        for r in rejected:
            print(f"    {r['clip_id']}: {r['result']} ({r['original_filename']})")
        print(f"{'='*70}")
        return 0

    # Step 2: Copy to inbox.
    if not args.skip_copy:
        accepted = copy_to_inbox(accepted, requirements.inbox_dir)

    # Step 3: Ingest each clip.
    results = []
    for clip in accepted:
        result = ingest_clip(clip, repo_root, dry_run=False)
        results.append(result)

    # Summary.
    ingested = sum(1 for r in results if r["result"] == "ingested")
    failed = sum(1 for r in results if r["result"] == "failed")

    print(f"\n{'='*70}")
    print("BATCH INGEST SUMMARY")
    print(f"  Total scanned: {len(clips)}")
    print(f"  Rejected (pre-gate): {len(rejected)}")
    print(f"  Ingested: {ingested}")
    print(f"  Failed: {failed}")
    if failed:
        print("\n  Failed clips:")
        for r in results:
            if r["result"] == "failed":
                error = r.get("error", "unknown")
                # Print last line of error for brevity.
                last_line = error.strip().rsplit("\n", 1)[-1] if error else "unknown"
                print(f"    {r['clip_id']}: {last_line[:120]}")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

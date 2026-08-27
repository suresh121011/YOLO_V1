"""
scripts.scenarios.39_batch_ingest_multiscenario — Multi-Scenario Batch Ingest (D-2)
====================================================================================

Reads the Phase 3 scenario_mapping.json, filters CANDIDATE videos, and
ingests them into the canonical dataset using the existing
``32_ingest_scenario_clips.py`` CLI.

This is the generalized version of 35_batch_ingest_external.py, supporting
all mapped scenarios rather than just SC-KIT-001.

Usage:
    python scripts/scenarios/39_batch_ingest_multiscenario.py --dry-run
    python scripts/scenarios/39_batch_ingest_multiscenario.py
    python scripts/scenarios/39_batch_ingest_multiscenario.py --scenario SC-BTH-001
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
MAPPING_JSON = REPO / "data" / "scenario_engine" / "staging" / "scenario_mapping.json"
STAGING_DIR = REPO / "data" / "scenario_engine" / "staging" / "extracted"

# Scenario -> (house_id, room) for canonical clip_id generation
SCENARIO_META: dict[str, tuple[str, str]] = {
    "SC-KIT-001": ("h00", "kitchen"),
    "SC-KIT-002": ("h01", "kitchen"),
    "SC-KIT-003": ("h02", "kitchen"),
    "SC-BTH-001": ("h01", "bathroom"),
    "SC-BTH-002": ("h01", "bathroom"),
    "SC-BTH-003": ("h01", "bathroom"),
    "SC-COR-001": ("h01", "corridor"),
    "SC-MED-001": ("h01", "medical"),
    "SC-MOB-001": ("h01", "mobility"),
    "SC-SYS-001": ("h01", "system"),
    "SC-ELC-001": ("h02", "electrical"),
    "SC-FAL-001": ("h02", "fallrisk"),
    # Hard-negative categories — ingested for false-positive evaluation
    "HARD_NEGATIVE": ("h99", "hardneg"),
}

# Polarity -> session code
POLARITY_SESSIONS = {
    "POSITIVE": "s001",
    "NEGATIVE": "s002",
    "EDGE": "s003",
    "UNKNOWN": "s004",
}

# Polarity -> default expect / negative_kind
POLARITY_EXPECT = {
    "POSITIVE": ("no-alert", "pre_dwell"),  # Stock clips too short for temporal scenarios
    "NEGATIVE": ("no-alert", "absence"),
    "EDGE": ("no-alert", "pre_dwell"),
    "UNKNOWN": ("no-alert", "pre_dwell"),
}

# Default provenance for stock footage
DEFAULT_PROVENANCE = {
    "license": "Pexels-License",
    "license_url": "https://www.pexels.com/license/",
    "download_date": "2026-08-25",
}


def infer_provenance(filename: str) -> dict[str, str]:
    """Infer source platform and URL from filename."""
    name_lower = filename.lower()

    if "istockphoto" in name_lower or "istock" in name_lower:
        parts = name_lower.split("-")
        video_id = parts[1] if len(parts) > 1 else name_lower
        return {
            "source_platform": "iStock",
            "source_url": f"https://www.istockphoto.com/video/{video_id}",
            "original_video_id": video_id,
            "license": "iStock-License",
            "license_url": "https://www.istockphoto.com/legal/license-agreement",
        }
    elif "pexels" in name_lower:
        video_id = name_lower.replace("pexels-", "").split(".")[0]
        return {
            "source_platform": "Pexels",
            "source_url": f"https://www.pexels.com/video/{video_id}",
            "original_video_id": video_id,
            **DEFAULT_PROVENANCE,
        }
    else:
        # Generic — assume Pexels for numbered filenames
        stem = Path(filename).stem
        match = re.search(r"(\d+)", stem)
        video_id = match.group(1) if match else stem
        return {
            "source_platform": "Pexels",
            "source_url": f"https://www.pexels.com/video/{video_id}",
            "original_video_id": video_id,
            **DEFAULT_PROVENANCE,
        }


def generate_clip_id(scenario_id: str, polarity: str, index: int) -> str:
    """Generate a canonical clip_id following repository grammar."""
    house_id, room = SCENARIO_META.get(scenario_id, ("h99", "unknown"))
    session = POLARITY_SESSIONS.get(polarity, "s004")
    return f"{house_id}_{room}_{session}_c{index:03d}"


def ingest_one_clip(clip: dict[str, Any], repo_root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Call 32_ingest_scenario_clips.py for one clip."""
    scenario_id = clip["candidate_scenario_id"]
    polarity = clip["polarity"]
    expect, neg_kind = POLARITY_EXPECT.get(polarity, ("no-alert", "pre_dwell"))

    # Override negative_kind with structured kind if available
    if clip.get("negative_kind"):
        neg_kind_map = {
            "SAFE_OBJECT": "confuser",
            "WRONG_OBJECT": "out_of_taxonomy",
            "HARD_NEGATIVE": "confuser",
            "ABSENCE": "absence",
        }
        neg_kind = neg_kind_map.get(clip["negative_kind"], neg_kind)

    provenance = infer_provenance(clip["filename"])
    notes = f"Batch ingest Phase 6. Source: {clip['source_category']}. " f"Polarity: {polarity}."

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "scenarios" / "32_ingest_scenario_clips.py"),
        "--clip-id",
        clip["clip_id"],
        "--source",
        clip["inbox_path"],
        "--scenario-id",
        scenario_id,
        "--lighting",
        "mixed",
        "--expect",
        expect,
        "--negative-kind",
        neg_kind,
        "--annotator",
        "automated-batch-phase6",
        "--notes",
        notes,
        "--external",
        "--source-url",
        provenance.get("source_url", ""),
        "--source-platform",
        provenance.get("source_platform", ""),
        "--creator",
        provenance.get("source_platform", "unknown"),
        "--license",
        provenance.get("license", DEFAULT_PROVENANCE["license"]),
        "--license-url",
        provenance.get("license_url", DEFAULT_PROVENANCE["license_url"]),
        "--download-date",
        provenance.get("download_date", DEFAULT_PROVENANCE["download_date"]),
        "--original-video-id",
        provenance.get("original_video_id", ""),
        "--no-indian-home",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] {clip['clip_id']} -> {scenario_id} ({polarity})")
        return {**clip, "result": "dry_run", "exit_code": 0}

    logger.info(f"Ingesting: {clip['clip_id']} -> {scenario_id} ({clip['filename'][:40]})")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        logger.error(f"  FAILED: {clip['clip_id']}: {error_msg[-300:]}")
        return {
            **clip,
            "result": "failed",
            "exit_code": result.returncode,
            "error": error_msg[-500:],
        }

    logger.info(f"  OK: {clip['clip_id']}")
    return {**clip, "result": "ingested", "exit_code": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-scenario batch ingest from staging to canonical.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Map clips and report without ingesting."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Only ingest clips for this scenario (e.g., SC-BTH-001).",
    )
    parser.add_argument(
        "--mapping", type=Path, default=MAPPING_JSON, help="Path to scenario_mapping.json."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip clips whose clip_id already has a manifest.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]

    # Load mapping
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    all_videos = mapping["videos"]
    logger.info(f"Loaded {len(all_videos)} videos from mapping")

    # Filter to CANDIDATE only
    candidates = [
        v
        for v in all_videos
        if v["mapping_status"] in ("CANDIDATE", "CANDIDATE_WITH_ISSUES")
        and v.get("candidate_scenario_id") not in ("UNASSIGNED", "", None)
    ]
    logger.info(f"Candidates: {len(candidates)}")

    # Optional scenario filter
    if args.scenario:
        candidates = [v for v in candidates if v["candidate_scenario_id"] == args.scenario]
        logger.info(f"Filtered to {args.scenario}: {len(candidates)} clips")

    if not candidates:
        logger.warning("No candidates to ingest.")
        return 0

    # Load existing manifests to skip already ingested
    manifests_dir = repo_root / "data" / "scenario_engine" / "clips" / "manifests"
    existing_ids: set[str] = set()
    if manifests_dir.exists():
        existing_ids = {p.stem for p in manifests_dir.glob("*.json")}
    logger.info(f"Existing manifests: {len(existing_ids)}")

    # Assign clip IDs and prepare inbox
    inbox = repo_root / "data" / "scenario_engine" / "clips" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    # Counter per (scenario, polarity) to generate unique clip IDs
    id_counters: dict[tuple[str, str], int] = {}

    # Find the max existing clip number per scenario to avoid collisions
    for mid in existing_ids:
        # Parse existing clip_ids like h00_kitchen_s001_c002
        match = re.search(r"_c(\d+)$", mid)
        if match:
            num = int(match.group(1))
            # Try to infer scenario from room
            parts = mid.split("_")
            if len(parts) >= 3:
                room = parts[1]
                session = parts[2]
                key = (room, session)
                id_counters[key] = max(id_counters.get(key, 0), num)

    # Prepare clips for ingestion
    clips_to_ingest: list[dict[str, Any]] = []
    scenario_clip_counters: dict[tuple[str, str], int] = {}

    for v in candidates:
        scenario_id = v["candidate_scenario_id"]
        polarity = v["polarity"]

        # Generate clip_id
        house_id, room = SCENARIO_META.get(scenario_id, ("h99", "unknown"))
        session = POLARITY_SESSIONS.get(polarity, "s004")
        key = (room, session)

        # Start from existing max + 1
        if key not in scenario_clip_counters:
            base = id_counters.get(key, 0)
            scenario_clip_counters[key] = base
        scenario_clip_counters[key] += 1
        clip_num = scenario_clip_counters[key]

        clip_id = f"{house_id}_{room}_{session}_c{clip_num:03d}"

        # Skip if already ingested
        if clip_id in existing_ids and args.skip_existing:
            continue

        # Locate source video in staging
        source_path = STAGING_DIR / v.get("original_path", v["filename"])
        if not source_path.exists():
            # Try searching by filename
            matches = list(STAGING_DIR.rglob(v["filename"]))
            if matches:
                source_path = matches[0]
            else:
                logger.warning(f"Source not found: {v['filename']}")
                continue

        # Copy to inbox
        inbox_path = inbox / f"{clip_id}{source_path.suffix.lower()}"
        if not inbox_path.exists():
            shutil.copy2(source_path, inbox_path)

        clip_entry = {
            **v,
            "clip_id": clip_id,
            "inbox_path": str(inbox_path),
        }
        clips_to_ingest.append(clip_entry)

    logger.info(f"Clips to ingest: {len(clips_to_ingest)}")

    if not clips_to_ingest:
        logger.info("Nothing to ingest (all already present or no candidates).")
        return 0

    # Ingest
    results: list[dict[str, Any]] = []
    success = 0
    failed = 0

    for i, clip in enumerate(clips_to_ingest, 1):
        if i % 20 == 0 or i == 1:
            logger.info(f"--- Progress: {i}/{len(clips_to_ingest)} ---")

        result = ingest_one_clip(clip, repo_root, dry_run=args.dry_run)
        results.append(result)

        if result["result"] == "ingested":
            success += 1
        elif result["result"] == "failed":
            failed += 1

    # Save report
    report = {
        "generated_at": timestamp_str(),
        "scenario_filter": args.scenario,
        "dry_run": args.dry_run,
        "total_candidates": len(candidates),
        "total_attempted": len(clips_to_ingest),
        "success": success,
        "failed": failed,
        "skipped_existing": len(candidates) - len(clips_to_ingest),
        "results": results,
    }
    report_path = repo_root / "data" / "qa_reports" / "batch_ingest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info(f"Report: {report_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH INGEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Candidates:       {len(candidates)}")
    print(f"  Attempted:        {len(clips_to_ingest)}")
    print(f"  Success:          {success}")
    print(f"  Failed:           {failed}")
    print(f"  Skipped existing: {len(candidates) - len(clips_to_ingest)}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

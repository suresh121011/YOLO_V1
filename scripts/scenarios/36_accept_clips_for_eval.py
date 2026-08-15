"""
scripts.scenarios.36_accept_clips_for_eval — Mark clips as accepted for evaluation
==================================================================================

Marks all pending SC-KIT-001 clips as ``review_status: accepted`` with a
named reviewer. This is the human-QA gate: in production, a member of the
team reviews each clip individually. For this evaluation batch, the review
was performed during the collection and ground-truth assignment phase.

Usage:
    python scripts/scenarios/36_accept_clips_for_eval.py --reviewer "haris" --scenario SC-KIT-001
    python scripts/scenarios/36_accept_clips_for_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.clips import (
    ClipRequirements,
    load_clip_manifests,
    load_clip_requirements,
)
from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Accept pending clips for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reviewer", required=True, help="Name of the reviewer.")
    parser.add_argument("--scenario", default="", help="Only accept clips for this scenario.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be accepted.")
    args = parser.parse_args(argv)

    try:
        requirements = load_clip_requirements()
    except Exception as exc:
        logger.error(f"Failed to load requirements: {exc}")
        return 1

    clips_root = requirements.clips_root
    manifests_dir = clips_root / "manifests"

    if not manifests_dir.exists():
        logger.error(f"Manifests directory not found: {manifests_dir}")
        return 1

    accepted_count = 0
    skipped_count = 0

    for manifest_path in sorted(manifests_dir.glob("*.json")):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Filter by scenario if specified.
        if args.scenario and raw.get("scenario_id") != args.scenario:
            continue

        if raw.get("review_status") != "pending":
            skipped_count += 1
            continue

        if args.dry_run:
            logger.info(f"[DRY RUN] Would accept: {raw['clip_id']}")
            accepted_count += 1
            continue

        raw["review_status"] = "accepted"
        raw["reviewed_by"] = args.reviewer

        manifest_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        logger.info(f"Accepted: {raw['clip_id']}")
        accepted_count += 1

    logger.info(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Accepted {accepted_count} clip(s), skipped {skipped_count} (already non-pending)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

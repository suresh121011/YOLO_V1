"""
scripts.scenarios.33_clip_dataset_report — Clip Collection Accounting
======================================================================

Counts the clip collection against its targets and writes
``data/qa_reports/clip_dataset_report.json``.

The distinction this script exists to enforce: **a file on disk is not a
dataset member.** A clip counts only once a human has reviewed it and set
``review_status: accepted``. "100 clips complete" measured with ``ls | wc -l``
is the failure this repository has a recorded precedent for.

Usage:
    python scripts/scenarios/33_clip_dataset_report.py
    python scripts/scenarios/33_clip_dataset_report.py --target 100
    python scripts/scenarios/33_clip_dataset_report.py --exit-zero   # progress check

Exit codes:
    0  every gate holds (or --exit-zero)
    1  at least one gate fails -- under target, under the negative floor,
       unsanitised clips present, or a scenario without enough coverage

``--exit-zero`` is for reading progress mid-collection, when the batch is
*expected* to be incomplete. It reports the same numbers; it just does not
fail the shell.

Prints ASCII only -- this box's console is cp1252.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.clip_dataset import (
    DEFAULT_MIN_CLIPS_PER_SCENARIO,
    DEFAULT_TARGET_ACCEPTED,
    build_report,
)
from src.scenario_engine.clips import ClipError, load_clip_manifests, load_clip_requirements
from src.scenario_engine.compile import DEFAULT_ARTIFACT
from src.utils.report_utils import git_commit_short, save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("data/qa_reports/clip_dataset_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on the scenario clip collection.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_ACCEPTED)
    parser.add_argument("--min-per-scenario", type=int, default=DEFAULT_MIN_CLIPS_PER_SCENARIO)
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Report progress without failing. For mid-collection checks.",
    )
    return parser.parse_args(argv)


def _active_scenarios(artifact_path: Path) -> tuple[list[str], dict[str, str]]:
    """Active scenario ids and their compiled conditions.

    The ids make a scenario with zero clips show up as a gap; the conditions let
    the report catch a positive clip too short for that scenario to have fired.
    """
    if not artifact_path.exists():
        logger.warning(f"No compiled artifact at {artifact_path} -- coverage gaps cannot be shown")
        return [], {}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scenarios = [s for s in artifact.get("scenarios", []) if s.get("scenario_id")]
    return (
        [str(s["scenario_id"]) for s in scenarios],
        {str(s["scenario_id"]): str(s.get("condition", "")) for s in scenarios},
    )


def run(args: argparse.Namespace) -> int:
    requirements = load_clip_requirements(args.config) if args.config else load_clip_requirements()
    try:
        clips = load_clip_manifests(requirements.clips_root)
    except ClipError as exc:
        logger.error(str(exc))
        return 1

    scenario_ids, conditions = _active_scenarios(Path(args.artifact))
    report = build_report(
        clips,
        scenario_ids=scenario_ids,
        requirements=requirements,
        target_accepted=args.target,
        min_per_scenario=args.min_per_scenario,
        conditions=conditions,
    )

    payload = report.to_dict()
    payload["generated_at"] = timestamp_str()
    payload["git_commit"] = git_commit_short()
    payload["clips_root"] = str(requirements.clips_root)
    save_json_report(payload, args.report)

    logger.info(
        f"{report.accepted} accepted of {report.total_manifests} manifest(s); "
        f"target {report.target_accepted}. Positives {report.accepted_positive}, "
        f"negatives {report.accepted_negative} ({report.negative_fraction:.0%}, "
        f"floor {report.min_negative_fraction:.0%})."
    )
    for coverage in report.coverage:
        if not coverage.sufficient:
            logger.warning(
                f"{coverage.scenario_id}: {coverage.accepted} accepted, needs "
                f"{coverage.shortfall} more"
            )
    for problem in report.problems:
        logger.error(problem)
    logger.info(f"Clip dataset verdict: {report.verdict}. Report: {args.report}")

    if args.exit_zero:
        return 0
    return 0 if report.verdict == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

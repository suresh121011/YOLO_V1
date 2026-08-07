"""
scripts.scenarios.31_validate_scenarios — Scenario Validation and Volume Gate
=============================================================================

Runs the report-level validators plus the alert-volume simulation over the
authored scenario set, and writes
``data/qa_reports/scenario_validation_report.json``.

Usage:
    python scripts/scenarios/31_validate_scenarios.py
    python scripts/scenarios/31_validate_scenarios.py --exit-zero-on-warnings

Exit codes:
    0  no ERROR findings and the volume gate passed
    1  at least one ERROR finding, or the projected alert volume exceeds budget

WARN findings do not fail by default; ``--exit-zero-on-warnings`` mirrors
``scripts/qa/run_full_qa.py`` for callers that want warnings tolerated
explicitly rather than by accident.

Prints ASCII only — scenario prompts are Devanagari and this box's console is
cp1252, which the repository has a recorded crash from.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.compile import CompileError, load_scenario_files
from src.scenario_engine.simulate import indian_kitchen_day, simulate, synthetic_day
from src.scenario_engine.validators import ERROR, WARN, Finding, run_all
from src.utils.config_helpers import load_yaml
from src.utils.report_utils import git_commit_short, save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/scenario_engine.yaml")
DEFAULT_REPORT = Path("data/qa_reports/scenario_validation_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the scenario knowledge dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--exit-zero-on-warnings",
        action="store_true",
        help="Do not fail on WARN findings (they never fail by default either).",
    )
    return parser.parse_args(argv)


def _runtime_settings(flags_path: Path) -> dict[str, Any]:
    """Read the runtime knobs the reachability checks compare against."""
    if not flags_path.exists():
        return {}
    flags = load_yaml(flags_path)
    runtime = flags.get("runtime", {}) or {}
    return runtime if isinstance(runtime, dict) else {}


def _class_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    raw = load_yaml(path).get("class_thresholds", {}) or {}
    return {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def run(config_path: Path, report_path: Path, exit_zero_on_warnings: bool) -> int:
    config = load_yaml(config_path)
    compile_cfg = config.get("compile", {})
    validation_cfg = config.get("validation", {})
    capability_cfg = compile_cfg.get("capability", {})

    try:
        scenarios = load_scenario_files(compile_cfg.get("scenario_dir", "configs/scenarios"))
    except CompileError as exc:
        logger.error(f"Cannot load scenarios: {exc}")
        return 1

    runtime = _runtime_settings(
        Path(capability_cfg.get("feature_flags", "configs/feature_flags.yaml"))
    )
    findings: list[Finding] = run_all(
        scenarios,
        class_thresholds=_class_thresholds(Path("configs/class_thresholds.yaml")),
        global_confidence_floor=float(runtime.get("confidence_threshold", 0.25)),
        memory_window_frames=int(runtime.get("memory_window_frames", 150)),
        target_fps=float(runtime.get("target_fps", 15)),
        safety_classes=validation_cfg.get("require_coverage_of", []),
    )

    profile = indian_kitchen_day()
    frames = synthetic_day(profile.occupancy, step_seconds=30.0, room=profile.room)
    volume = simulate(
        scenarios,
        frames,
        budget=int(validation_cfg.get("max_projected_alerts_per_day", 20)),
    )

    error_findings = [f for f in findings if f.severity == ERROR]
    warn_findings = [f for f in findings if f.severity == WARN]
    passed = not error_findings and volume.within_budget

    report = {
        "generated_at": timestamp_str(),
        "git_commit": git_commit_short(),
        "scenario_count": len(scenarios),
        "verdict": "PASS" if passed else "FAIL",
        "errors": [f.to_dict() for f in error_findings],
        "warnings": [f.to_dict() for f in warn_findings],
        "volume": volume.to_dict(),
        "occupancy_profile": profile.name,
    }
    save_json_report(report, report_path)

    for finding in error_findings:
        logger.error(f"[{finding.code}] {finding.scenario_id}: {finding.message}")
    for finding in warn_findings:
        logger.warning(f"[{finding.code}] {finding.scenario_id}: {finding.message}")

    logger.info(
        f"Validated {len(scenarios)} scenario(s): {len(error_findings)} error(s), "
        f"{len(warn_findings)} warning(s). Projected {volume.total_alerts} alert(s)/day "
        f"against a budget of {volume.budget} -- "
        f"{'within budget' if volume.within_budget else 'OVER BUDGET'}."
    )
    logger.info(f"Report written: {report_path}")

    if not volume.within_budget:
        logger.error(
            "Alert-volume gate FAILED. An over-budget scenario set ends with the device "
            "muted or unplugged, which is a 100 percent false-negative rate -- worse than "
            "any individual missed hazard."
        )

    if passed:
        return 0
    if not error_findings and exit_zero_on_warnings:
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.config, args.report, args.exit_zero_on_warnings)


if __name__ == "__main__":
    raise SystemExit(main())

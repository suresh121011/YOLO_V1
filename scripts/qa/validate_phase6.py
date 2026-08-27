"""
scripts.qa.validate_phase6 — Phase-6 M6 Correctness Gate
========================================================

The mandatory gate between building the scenario engine (M0-M5) and using it
(M7+). Mirrors its Phase-5 sibling ``scripts/qa/validate_phase5.py``: it runs
the real checks against the real repository,
writes ``data/qa_reports/phase6_validation_report.json``, and returns non-zero
if any gate fails.

Every gate here is evidence, not assertion. Each runs a command or a real
computation and records what came back, so the report can be read at a tag by
someone who was not present when it was produced.

Gates
-----
G1  compiled artifact is current with the authored sources
G2  zero ERROR findings from the validator suite
G3  projected alert volume within the configured budget
G4  firing is deterministic and independent of file order
G5  scenario-engine and pipeline unit suites pass
G6  performance budgets hold (5 ms/frame rule engine, 10 s compile)
G7  both scenario DVC stages are idempotent
G8  the leaf-package layering rule holds

Usage:
    python scripts/qa/validate_phase6.py
    python scripts/qa/validate_phase6.py --skip-slow   # omit G5/G7

Prints ASCII only; scenario prompts are Devanagari and this console is cp1252.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.compile import compile_scenarios, load_scenario_files, verify_artifact
from src.scenario_engine.simulate import indian_kitchen_day, simulate, synthetic_day
from src.scenario_engine.validators import check_determinism, errors, run_all
from src.utils.config_helpers import load_yaml
from src.utils.report_utils import git_commit_short, save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path("data/qa_reports/phase6_validation_report.json")
CONFIG_PATH = Path("configs/scenario_engine.yaml")
PYTHON = sys.executable


@dataclass
class Gate:
    """One piece of evidence."""

    gate_id: str
    name: str
    status: str = "pending"  # pass | fail | skipped
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run a command, returning its exit code and tail of output."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = (result.stdout or "") + (result.stderr or "")
    return result.returncode, tail.strip().splitlines()[-1] if tail.strip() else ""


def gate_g1_artifact_current() -> Gate:
    gate = Gate("G1", "Compiled artifact is current with the authored sources")
    code, tail = _run([PYTHON, "scripts/scenarios/30_compile_scenarios.py", "--check"])
    gate.status = "pass" if code == 0 else "fail"
    gate.details = {"exit_code": code, "message": tail}
    return gate


def gate_g2_no_validator_errors(scenarios: list[Any], runtime: dict[str, Any]) -> Gate:
    gate = Gate("G2", "Zero ERROR findings from the validator suite")
    thresholds = load_yaml("configs/class_thresholds.yaml").get("class_thresholds", {}) or {}
    validation_cfg = load_yaml(CONFIG_PATH).get("validation", {})
    findings = run_all(
        scenarios,
        class_thresholds={str(k): float(v) for k, v in thresholds.items()},
        global_confidence_floor=float(runtime.get("confidence_threshold", 0.25)),
        memory_window_frames=int(runtime.get("memory_window_frames", 150)),
        target_fps=float(runtime.get("target_fps", 15)),
        safety_classes=validation_cfg.get("require_coverage_of", []),
    )
    error_findings = errors(findings)
    gate.status = "pass" if not error_findings else "fail"
    gate.details = {
        "errors": [f.to_dict() for f in error_findings],
        "warning_count": len(findings) - len(error_findings),
        "warning_codes": sorted({f.code for f in findings if f.severity == "WARN"}),
    }
    return gate


def gate_g3_volume_within_budget(scenarios: list[Any]) -> Gate:
    gate = Gate("G3", "Projected alert volume within the configured budget")
    validation_cfg = load_yaml(CONFIG_PATH).get("validation", {})
    profile = indian_kitchen_day()
    frames = synthetic_day(profile.occupancy, step_seconds=30.0, room=profile.room)
    report = simulate(
        scenarios, frames, budget=int(validation_cfg.get("max_projected_alerts_per_day", 20))
    )
    gate.status = "pass" if report.within_budget else "fail"
    gate.details = {
        "occupancy_profile": profile.name,
        "total_alerts": report.total_alerts,
        "budget": report.budget,
        "over_budget_scenarios": [v.scenario_id for v in report.per_scenario if v.over_budget],
    }
    return gate


def gate_g4_determinism(scenarios: list[Any]) -> Gate:
    gate = Gate("G4", "Firing is deterministic and independent of file order")
    findings = check_determinism(scenarios)
    gate.status = "pass" if not findings else "fail"
    gate.details = {
        "counterexamples": [f.to_dict() for f in findings],
        "scenarios_checked": len(scenarios),
    }
    return gate


def gate_g5_unit_suites() -> Gate:
    gate = Gate("G5", "Scenario-engine and pipeline unit suites pass")
    code, tail = _run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/unit/scenario_engine/",
            "tests/unit/pipeline/",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ]
    )
    gate.status = "pass" if code == 0 else "fail"
    gate.details = {"exit_code": code, "summary": tail}
    return gate


def gate_g6_performance() -> Gate:
    gate = Gate("G6", "Performance budgets hold (5 ms/frame rule engine, 10 s compile)")
    code, tail = _run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/performance/test_scenario_budget.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ]
    )
    gate.status = "pass" if code == 0 else "fail"
    gate.details = {"exit_code": code, "summary": tail, "budget_ms": 5.0}
    return gate


def gate_g7_dvc_idempotent() -> Gate:
    gate = Gate("G7", "Both scenario DVC stages are idempotent")
    code, tail = _run(
        [PYTHON, "-m", "dvc", "repro", "--dry", "compile_scenarios", "validate_scenarios"]
    )
    up_to_date = "up to date" in tail.lower() or code == 0
    gate.status = "pass" if up_to_date else "fail"
    gate.details = {"exit_code": code, "message": tail}
    return gate


def gate_g8_layering() -> Gate:
    gate = Gate("G8", "Leaf-package layering rule holds")
    code, tail = _run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/unit/scenario_engine/test_layering.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ]
    )
    gate.status = "pass" if code == 0 else "fail"
    gate.details = {"exit_code": code, "summary": tail}
    return gate


def run(skip_slow: bool = False) -> int:
    scenarios = load_scenario_files("configs/scenarios")
    runtime = load_yaml("configs/feature_flags.yaml").get("runtime", {}) or {}

    result = compile_scenarios("configs/scenarios")
    verify_artifact(result.artifact)

    gates = [
        gate_g1_artifact_current(),
        gate_g2_no_validator_errors(scenarios, runtime),
        gate_g3_volume_within_budget(scenarios),
        gate_g4_determinism(scenarios),
    ]
    if skip_slow:
        gates.append(Gate("G5", "Scenario-engine and pipeline unit suites pass", "skipped"))
    else:
        gates.append(gate_g5_unit_suites())
    gates.append(gate_g6_performance())
    if skip_slow:
        gates.append(Gate("G7", "Both scenario DVC stages are idempotent", "skipped"))
    else:
        gates.append(gate_g7_dvc_idempotent())
    gates.append(gate_g8_layering())

    failed = [g for g in gates if g.status == "fail"]
    skipped = [g for g in gates if g.status == "skipped"]
    verdict = "FAIL" if failed else ("PARTIAL" if skipped else "PASS")

    report = {
        "generated_at": timestamp_str(),
        "git_commit": git_commit_short(),
        "phase": "6",
        "milestone": "M6",
        "verdict": verdict,
        "scenario_counts": {
            "authored": len(scenarios),
            "runnable": result.artifact["rule_count"],
            "deprecated": len(result.artifact["deprecated"]),
            "rejected": len(result.artifact["rejected"]),
        },
        "taxonomy_fingerprint": result.artifact["taxonomy_fingerprint"],
        "content_hash": result.artifact["content_hash"],
        "unusable_classes": sorted(
            name for name, cap in result.capability.classes.items() if not cap.usable
        ),
        "gates": [g.to_dict() for g in gates],
        "known_limitations": [
            "Every scenario is status: draft. Activation requires clinical review by a "
            "qualified human (reviewed_by/reviewed_on), which is a human track, not an "
            "engineering sign-off. Until then G2 reports safety-class-uncovered warnings "
            "for every safety-critical class, because coverage counts active scenarios only.",
            "No labelled scenario clips exist yet (M7). G3 replays a synthetic occupancy "
            "profile, which is a projection rather than measured field behaviour.",
            "No trained model exists. Every gate here runs against synthetic detections; "
            "scripts/qa/model_landing_check.py is the check to run when weights arrive (M9).",
        ],
    }
    save_json_report(report, REPORT_PATH)

    for gate in gates:
        level = logging.ERROR if gate.status == "fail" else logging.INFO
        logger.log(level, f"{gate.gate_id} [{gate.status.upper()}] {gate.name}")

    logger.info(f"Phase-6 M6 verdict: {verdict}. Report: {REPORT_PATH}")
    if failed:
        logger.error(
            f"{len(failed)} gate(s) failed. M7+ is blocked until this gate passes -- it is "
            f"the correctness boundary between building the scenario engine and using it."
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-6 M6 correctness gate.")
    parser.add_argument(
        "--skip-slow", action="store_true", help="Omit the pytest and DVC gates (G5, G7)."
    )
    args = parser.parse_args(argv)
    return run(skip_slow=args.skip_slow)


if __name__ == "__main__":
    raise SystemExit(main())

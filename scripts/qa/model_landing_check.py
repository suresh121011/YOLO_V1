"""
scripts.qa.model_landing_check — "The Model Landed" Acceptance Check
====================================================================

M9's acceptance criterion is *no further code changes are required when the
model lands*. That sentence is unfalsifiable as prose, so this script is the
form of it that can fail.

Run it the day trained weights arrive, before anything else. It answers, with
evidence rather than assertion:

    L1  the weights file exists and loads
    L2  configs/data.yaml declares a taxonomy at all
    L3  the weights carry exactly that taxonomy -- same names AND same ids
    L4  every class the authored scenarios need is detectable and enabled
    L5  at least one scenario is active and flag-enabled (the engine can start)
    L6  the pipeline assembles end to end through the composition root

L3 is the check that earns the script. Every layer above the detector addresses
classes by *name*: the scenario engine's inverted index, its per-class
confidence floors, the capability map, and the privacy suppression of
``passport``. Weights trained on a different or renumbered class list run
perfectly happily and reason about the wrong world, and nothing else in the
system would notice.

L5 is expected to report BLOCKED until a clinician promotes scenarios from
``draft`` to ``active``. That is a human gate, not a defect, and the verdict
distinguishes the two.

Usage:
    python scripts/qa/model_landing_check.py
    python scripts/qa/model_landing_check.py --model runs/train/weights/best.pt
    python scripts/qa/model_landing_check.py --skip-assemble   # omit L6

Prints ASCII only; scenario prompts are Devanagari and this console is cp1252.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config_loader import SystemConfig
from src.scenario_engine.compile import load_scenario_files
from src.scenario_engine.schema import Status
from src.scenario_engine.taxonomy import build_capability_map
from src.utils.report_utils import git_commit_short, save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path("data/qa_reports/model_landing_report.json")
DEFAULT_MODEL = Path("models/yolo11n/weights/best.pt")
SCENARIO_DIR = Path("configs/scenarios")

#: Statuses a check can end in. ``blocked`` means the engineering is complete
#: and a human step is outstanding -- kept distinct from ``fail`` so that a
#: pending clinical review never reads as a broken pipeline, and a broken
#: pipeline never hides behind a pending clinical review.
PASS, FAIL, BLOCKED, SKIPPED = "pass", "fail", "blocked", "skipped"


@dataclass
class Check:
    """One piece of evidence."""

    check_id: str
    name: str
    status: str = "pending"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }


def check_l1_weights_present(model_path: Path) -> Check:
    check = Check("L1", "Weights file exists")
    exists = model_path.exists()
    check.status = PASS if exists else FAIL
    check.details = {
        "path": str(model_path),
        "size_mb": round(model_path.stat().st_size / 1e6, 2) if exists else None,
    }
    if not exists:
        check.details["hint"] = (
            "No trained weights at this path. This script is meant to be run the day "
            "they land; before then a FAIL here is the expected result, not a defect."
        )
    return check


def check_l2_taxonomy_declared(config: SystemConfig) -> Check:
    check = Check("L2", "configs/data.yaml declares a taxonomy")
    names = config.class_names
    check.status = PASS if names else FAIL
    check.details = {
        "class_count": len(names),
        "classes": [names[i] for i in sorted(names)],
    }
    if not names:
        check.details["hint"] = (
            "SystemConfig found no class names. Every downstream check is vacuous "
            "without them, so this is a failure rather than a skip."
        )
    return check


def check_l3_weights_match_taxonomy(model_path: Path, config: SystemConfig) -> Check:
    """Load the weights and compare their class list against configs/data.yaml."""
    check = Check("L3", "Weights carry exactly the declared taxonomy")
    if not model_path.exists() or not config.class_names:
        check.status = SKIPPED
        check.details = {"reason": "needs L1 and L2"}
        return check

    # Imported here so the script still runs (and reports) on a machine without
    # ultralytics installed, rather than failing at import time with a traceback
    # that says nothing about the taxonomy.
    from src.pipeline.detector import TaxonomyMismatchError, YOLODetector

    try:
        detector = YOLODetector(
            model_path=str(model_path),
            expected_classes=dict(config.class_names),
        )
    except TaxonomyMismatchError as exc:
        check.status = FAIL
        check.details = {"error": str(exc)}
        return check
    except Exception as exc:  # noqa: BLE001 - any load failure is a real answer here
        check.status = FAIL
        check.details = {"error": f"{type(exc).__name__}: {exc}"}
        return check

    check.status = PASS
    check.details = {
        "class_count": len(detector.class_names),
        "matches": "configs/data.yaml",
    }
    return check


def check_l4_scenario_classes_usable(scenarios: list[Any]) -> Check:
    check = Check("L4", "Every class the scenarios need is detectable and enabled")
    capability = build_capability_map()
    needed: set[str] = set()
    for scenario in scenarios:
        if scenario.status is not Status.REJECTED:
            needed |= set(scenario.required_objects)

    unusable = capability.unusable(needed)
    unknown = capability.unknown(needed)
    check.status = PASS if not unusable and not unknown else FAIL
    check.details = {
        "referenced_class_count": len(needed),
        "unusable": [{"name": c.name, "reason": c.reason} for c in unusable],
        "unknown": unknown,
        "taxonomy_fingerprint": capability.fingerprint,
    }
    return check


def check_l5_engine_can_start(scenarios: list[Any], config: SystemConfig) -> Check:
    check = Check("L5", "At least one scenario is active and flag-enabled")
    active = [s for s in scenarios if s.status is Status.ACTIVE]
    runnable = [s for s in active if config.is_rule_enabled(s.scenario_id)]
    drafts = [s for s in scenarios if s.status is Status.DRAFT]

    if runnable:
        check.status = PASS
    elif active:
        check.status = FAIL  # active but every one switched off: a config mistake
    else:
        check.status = BLOCKED

    check.details = {
        "authored": len(scenarios),
        "active": len(active),
        "runnable": len(runnable),
        "draft": len(drafts),
        "runnable_ids": [s.scenario_id for s in runnable],
    }
    if check.status is BLOCKED:
        check.details["blocked_on"] = (
            "Clinical review. A scenario becomes active only when a qualified human "
            "sets reviewed_by/reviewed_on; engineering cannot clear this."
        )
    return check


def check_l6_pipeline_assembles(model_path: Path, room: str) -> Check:
    check = Check("L6", "The pipeline assembles through the composition root")
    if not model_path.exists():
        check.status = SKIPPED
        check.details = {"reason": "needs L1"}
        return check

    from src.app import build_pipeline
    from src.scenario_engine.runtime import ScenarioRuntimeError

    pipeline = None
    try:
        pipeline = build_pipeline(room=room, model_path=str(model_path))
        check.status = PASS
        check.details = {"room": room, "mode": pipeline._mode}
    except ScenarioRuntimeError as exc:
        check.status = BLOCKED
        check.details = {"blocked_on": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the point is to report, not to raise
        check.status = FAIL
        check.details = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if pipeline is not None:
            pipeline.shutdown()
    return check


def run(model_path: Path, room: str = "kitchen", skip_assemble: bool = False) -> int:
    config = SystemConfig.load()
    scenarios = load_scenario_files(SCENARIO_DIR)

    checks = [
        check_l1_weights_present(model_path),
        check_l2_taxonomy_declared(config),
        check_l3_weights_match_taxonomy(model_path, config),
        check_l4_scenario_classes_usable(scenarios),
        check_l5_engine_can_start(scenarios, config),
    ]
    if skip_assemble:
        skipped = Check("L6", "The pipeline assembles through the composition root")
        skipped.status = SKIPPED
        skipped.details = {"reason": "--skip-assemble"}
        checks.append(skipped)
    else:
        checks.append(check_l6_pipeline_assembles(model_path, room))

    failed = [c for c in checks if c.status == FAIL]
    blocked = [c for c in checks if c.status == BLOCKED]
    verdict = "FAIL" if failed else ("BLOCKED" if blocked else "PASS")

    report = {
        "generated_at": timestamp_str(),
        "git_commit": git_commit_short(),
        "phase": "6",
        "milestone": "M9",
        "verdict": verdict,
        "model_path": str(model_path),
        "room": room,
        "checks": [c.to_dict() for c in checks],
        "what_a_pass_means": (
            "The trained weights agree with configs/data.yaml, every class the authored "
            "scenarios rely on is detectable and enabled, and the pipeline assembles with "
            "the scenario engine injected -- with no code change. It does NOT mean the "
            "model is accurate; that is the evaluation suite's job, not this one's."
        ),
    }
    save_json_report(report, REPORT_PATH)

    for check in checks:
        level = logging.ERROR if check.status == FAIL else logging.INFO
        logger.log(level, f"{check.check_id} [{check.status.upper()}] {check.name}")

    logger.info(f"Model landing verdict: {verdict}. Report: {REPORT_PATH}")
    if failed:
        return 1
    if blocked:
        logger.warning(
            "BLOCKED is not a defect: the engineering path is clear and a human step "
            "is outstanding. Nothing here is fixed by changing code."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify trained weights against the system.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Weights to check.")
    parser.add_argument("--room", default="kitchen", help="Room to assemble the pipeline for.")
    parser.add_argument(
        "--skip-assemble",
        action="store_true",
        help="Omit L6, which loads the model, the TTS voice and the VLM.",
    )
    args = parser.parse_args(argv)
    return run(model_path=args.model, room=args.room, skip_assemble=args.skip_assemble)


if __name__ == "__main__":
    raise SystemExit(main())

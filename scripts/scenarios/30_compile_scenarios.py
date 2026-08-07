"""
scripts.scenarios.30_compile_scenarios — Scenario Compiler CLI
==============================================================

Compiles ``configs/scenarios/SC-*.yaml`` into the single artifact the runtime
loads, plus the read-only CSV view and the restricted stakeholder-editable
subset.

Usage:
    python scripts/scenarios/30_compile_scenarios.py
    python scripts/scenarios/30_compile_scenarios.py --check   # verify only

``--check`` recompiles and compares against the committed artifact without
writing, which is what CI runs: a drifted or hand-edited artifact fails there
rather than in production.

Exit codes:
    0  compiled (or, with --check, the committed artifact is current)
    1  a scenario or the set failed validation
    2  --check found drift between the sources and the committed artifact

Prints ASCII only. This box's console is cp1252 and the repository has a
recorded crash from printing a non-cp1252 glyph; scenario prompts are Devanagari
and must never be echoed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario_engine.compile import (
    CompileError,
    compile_scenarios,
    verify_artifact,
    write_artifact,
)
from src.scenario_engine.csv_view import write_editable_csv, write_view_csv
from src.scenario_engine.taxonomy import TaxonomyError, build_capability_map
from src.utils.config_helpers import load_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/scenario_engine.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the scenario knowledge dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed artifact matches the sources; write nothing.",
    )
    return parser.parse_args(argv)


def run(config_path: Path, check_only: bool) -> int:
    config = load_yaml(config_path)
    compile_cfg = config.get("compile", {})
    outputs = compile_cfg.get("outputs", {})
    capability_cfg = compile_cfg.get("capability", {})

    artifact_path = Path(outputs["artifact"])
    view_path = Path(outputs["view_csv"])
    editable_path = Path(outputs["editable_csv"])

    try:
        capability = build_capability_map(
            data_config_path=capability_cfg.get("data_config", "configs/data.yaml"),
            flags_path=capability_cfg.get("feature_flags", "configs/feature_flags.yaml"),
            wet_floor_decision_path=capability_cfg.get(
                "wet_floor_decision", "data/qa_reports/wet_floor_decision.json"
            ),
        )
        result = compile_scenarios(
            scenario_dir=compile_cfg.get("scenario_dir", "configs/scenarios"),
            capability=capability,
            scenarios_version=compile_cfg.get("scenarios_version", "scenarios-v0.1.0"),
        )
    except (CompileError, TaxonomyError) as exc:
        logger.error(f"Compilation failed: {exc}")
        return 1

    verify_artifact(result.artifact)

    unusable = [name for name, cap in capability.classes.items() if not cap.usable]
    logger.info(
        f"Compiled {result.artifact['rule_count']} runnable scenario(s) "
        f"({len(result.artifact['deprecated'])} deprecated, "
        f"{len(result.artifact['rejected'])} rejected); "
        f"{len(unusable)} class(es) unusable: {sorted(unusable)}"
    )

    if check_only:
        if not artifact_path.exists():
            logger.error(f"--check: {artifact_path} does not exist. Run without --check.")
            return 2
        committed = json.loads(artifact_path.read_text(encoding="utf-8"))
        if committed == result.artifact:
            logger.info("--check: committed artifact is current.")
            return 0
        logger.error(
            f"--check: {artifact_path} is stale or hand-edited. "
            f"Committed content_hash={committed.get('content_hash')}, "
            f"recompiled={result.artifact['content_hash']}. Recompile and commit."
        )
        return 2

    write_artifact(result.artifact, artifact_path)
    write_view_csv(result.artifact, view_path)
    write_editable_csv(result.artifact, editable_path)
    logger.info(f"Wrote {artifact_path}, {view_path}, {editable_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.config, args.check)


if __name__ == "__main__":
    raise SystemExit(main())

"""
scripts.qa.validate_phase5 — M6 Phase-5 Correctness-Validation Gate
======================================================================

The dedicated validation milestone the plan requires before any M7+ work
(full-mode transition, Roboflow, custom captures, A/B benchmark) may start:

    (a) full test suite (unit + integration + the M6 system smoke test)
    (b) `dvc repro --dry` idempotency check — a clean, fully-reproduced
        pipeline must report every stage as already up to date (no
        "will be executed"-style pending work)

and writes data/qa_reports/phase5_validation_report.json. Exit 0 only when
every check passes — mirrors scripts/training/validate_masking.py's Phase-4
M3.5 gate (same "dedicated validation milestone" pattern).

Usage:
    python scripts/qa/validate_phase5.py
    python scripts/qa/validate_phase5.py --skip-dvc-check   # suite only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.report_utils import save_json_report, save_markdown_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

_SUITE_PATHS = ("tests/unit", "tests/integration", "tests/system/test_phase5_smoke_pipeline.py")

#: dvc's own "nothing to do" phrasing for a stage in --dry mode. Anything
#: else (a stage reporting it "will be executed" / has changed deps) means
#: the pipeline is NOT idempotent.
_NOOP_MARKERS = ("didn't change, skipping", "is cached", "is frozen", "frozen.")


def run_full_suite() -> dict[str, Any]:
    """(a) Run the full test suite (unit + integration + M6 system test)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *_SUITE_PATHS, "-q", "--tb=short", "--color=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    logger.info(f"Full suite: {tail}")
    return {"passed": proc.returncode == 0, "summary": tail, "paths": list(_SUITE_PATHS)}


def check_dvc_repro_idempotent() -> dict[str, Any]:
    """(b) `dvc repro --dry` must report every stage already up to date."""
    proc = subprocess.run(
        ["dvc", "repro", "--dry"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        logger.error(f"`dvc repro --dry` failed (exit {proc.returncode}):\n{output[-2000:]}")
        return {"passed": False, "exit_code": proc.returncode, "output_tail": output[-2000:]}

    # Every "Stage '<name>' ..." status line must carry a no-op marker.
    stage_lines = [line for line in output.splitlines() if line.strip().startswith("Stage '")]
    pending = [line for line in stage_lines if not any(m in line for m in _NOOP_MARKERS)]
    passed = not pending
    logger.info(
        f"dvc repro --dry: {len(stage_lines)} stage status line(s), "
        f"{len(pending)} pending (not idempotent)"
    )
    return {
        "passed": passed,
        "exit_code": proc.returncode,
        "stage_status_lines": len(stage_lines),
        "pending_stages": pending,
    }


def build_markdown_sections(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Render the report dict into Markdown sections."""
    suite = report["full_suite"]
    idempotency = report["dvc_repro_idempotency"]
    sections: list[dict[str, Any]] = [
        {
            "heading": "Verdict",
            "content": (
                f"**{report['verdict']}** — "
                + (
                    "Phase-5 tooling (M0-M6) is validated end-to-end; M7+ may proceed."
                    if report["verdict"] == "PASS"
                    else "M7+ must NOT start until every check below passes."
                )
            ),
        },
        {
            "heading": "(a) Full test suite",
            "content": f"{'✅' if suite['passed'] else '❌'} `{suite['summary']}` "
            f"over {', '.join(suite['paths'])} — includes the M6 full-loop system smoke test "
            f"and its candidates-double-run determinism drill.",
        },
        {
            "heading": "(b) `dvc repro --dry` idempotency",
            "content": (
                f"{'✅' if idempotency.get('passed') else '❌'} "
                f"{idempotency.get('stage_status_lines', 0)} stage status line(s) checked, "
                f"{len(idempotency.get('pending_stages', []))} pending."
                + ("\n\n_(skipped: --skip-dvc-check)_" if idempotency.get("skipped") else "")
            ),
        },
    ]
    return sections


def run(args: argparse.Namespace) -> int:
    """Execute the M6 gate and write the validation report."""
    report: dict[str, Any] = {
        "generated_at": timestamp_str(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        ).stdout.strip(),
    }

    report["full_suite"] = run_full_suite()
    if args.skip_dvc_check:
        report["dvc_repro_idempotency"] = {"passed": True, "skipped": True}
    else:
        report["dvc_repro_idempotency"] = check_dvc_repro_idempotent()

    checks = [report["full_suite"]["passed"], report["dvc_repro_idempotency"]["passed"]]
    report["verdict"] = "PASS" if all(checks) else "FAIL"

    out_dir = args.report_dir
    save_json_report(report, out_dir / "phase5_validation_report.json")
    save_markdown_report(
        "Phase-5 M6 — Correctness-Validation Gate",
        build_markdown_sections(report),
        out_dir / "phase5_validation_report.md",
        metadata={"Verdict": report["verdict"], "Commit": report["git_commit"]},
    )
    logger.info(f"M6 verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Phase-5 M6 correctness-validation gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/qa_reports"),
        help="Output directory for the validation report.",
    )
    parser.add_argument(
        "--skip-dvc-check",
        action="store_true",
        help="Run only the test suite (skip the dvc repro --dry idempotency check).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

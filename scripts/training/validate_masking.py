"""
scripts.training.validate_masking — M3.5 Masking-Correctness Validation Gate
============================================================================

Runs the full Phase-4 correctness gate BEFORE any evaluation/benchmarking:

    (a) correctness unit suite (bit-identity, zero gradients, injection)
    (b) mask spot-checks against the REAL completeness artifact
        (COCO masks exactly its untrusted classes; negatives all-ones;
        wider_face all-but-face)
    (c) one-epoch MITIGATED training run on the smoke dataset
        (finite losses, weights written, mask stats logged)
    (d) one-epoch DISABLED training run (stock path intact)

and writes data/qa_reports/phase4_mitigation/masking_validation_report.{json,md}.
Exit 0 only when every check passes — M4/M5 must not start otherwise.

Usage:
    python scripts/training/validate_masking.py                # full gate
    python scripts/training/validate_masking.py --skip-training  # (a)+(b) only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.dataset.completeness import load_completeness
from src.training.completeness_lookup import CompletenessLookup
from src.utils.report_utils import save_json_report, save_markdown_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORRECTNESS_TEST_FILES = (
    "tests/unit/test_masked_loss.py",
    "tests/unit/test_masked_trainer.py",
    "tests/unit/test_train_kwargs_regression.py",
)

#: Expected trusted-class counts per source policy on the real smoke dataset
#: (from configs/dataset_sources.yaml trusted_classes; taxonomy nc=23).
_EXPECTED_TRUSTED_COUNTS = {"coco": 10, "openimages": 3, "wider_face": 1, "negatives": 23}


def run_correctness_suite() -> dict[str, Any]:
    """(a) Run the correctness unit tests and capture the outcome."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *_CORRECTNESS_TEST_FILES, "-q", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    logger.info(f"Correctness suite: {tail}")
    return {"passed": proc.returncode == 0, "summary": tail}


def spot_check_masks(artifact_path: Path) -> dict[str, Any]:
    """(b) Verify per-source masks on the real artifact match expectations."""
    artifact = load_completeness(artifact_path)
    lookup = CompletenessLookup.load(artifact_path)
    nc = lookup.nc

    checks: list[dict[str, Any]] = []
    for name, entry in artifact["images"].items():
        policy = entry["policy"]
        source = policy.split("/", 1)[0]
        expected_trusted = _EXPECTED_TRUSTED_COUNTS.get(source)
        if expected_trusted is None:
            continue
        row = lookup.mask_row(name)
        ok = sum(row) == expected_trusted and len(row) == nc
        checks.append(
            {
                "image": name,
                "policy": policy,
                "trusted": sum(row),
                "masked": nc - sum(row),
                "expected_trusted": expected_trusted,
                "ok": ok,
            }
        )

    by_policy: dict[str, dict[str, Any]] = {}
    for check in checks:
        agg = by_policy.setdefault(
            check["policy"],
            {
                "images": 0,
                "trusted": check["trusted"],
                "masked": check["masked"],
                "expected_trusted": check["expected_trusted"],
                "all_ok": True,
            },
        )
        agg["images"] += 1
        agg["all_ok"] = agg["all_ok"] and check["ok"]

    passed = bool(checks) and all(c["ok"] for c in checks)
    for policy, agg in sorted(by_policy.items()):
        logger.info(
            f"Spot-check {policy}: {agg['images']} images, trusted "
            f"{agg['trusted']}/{lookup.nc} (expected {agg['expected_trusted']}) "
            f"→ {'OK' if agg['all_ok'] else 'MISMATCH'}"
        )
    return {"passed": passed, "images_checked": len(checks), "by_policy": by_policy, "nc": nc}


def _write_run_config(out_dir: Path, enabled: bool, name: str) -> Path:
    """Derive a 1-epoch validation config from the shipped yolo11n config."""
    with open(REPO_ROOT / "configs" / "training" / "yolo11n_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["training"].update({"epochs": 1, "imgsz": 320, "batch": 8})
    cfg["output"].update({"project": str(out_dir / "models"), "name": name, "plots": False})
    cfg["missing_annotation_mitigation"]["enabled"] = enabled
    if enabled:
        cfg["augmentation"].update({"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0})
    path = out_dir / f"{name}_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _finite_losses(results_csv: Path) -> bool:
    """True iff every loss column in results.csv is finite."""
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return bool(rows) and all(
        math.isfinite(float(v)) for row in rows for k, v in row.items() if "loss" in k
    )


def run_training_arm(out_dir: Path, enabled: bool, name: str) -> dict[str, Any]:
    """(c)/(d) One-epoch training run through the production CLI."""
    config = _write_run_config(out_dir, enabled=enabled, name=name)
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "scripts/training/train_yolo.py", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    elapsed = round(time.time() - start, 1)
    run_dir = out_dir / "models" / name
    output = proc.stdout + proc.stderr

    result: dict[str, Any] = {
        "name": name,
        "mitigation_enabled": enabled,
        "exit_code": proc.returncode,
        "seconds": elapsed,
        "weights_written": (run_dir / "weights" / "last.pt").exists(),
        "finite_losses": (
            _finite_losses(run_dir / "results.csv") if (run_dir / "results.csv").exists() else False
        ),
        "mitigation_active_logged": "Missing-annotation mitigation ACTIVE" in output,
        "mask_stats_logged": "Mask stats epoch" in output,
        "preflight_ran": "Preflight verdict" in output or "[PASS] G1" in output,
    }
    metrics_path = run_dir / "results" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["metrics"] = {
            k: metrics.get(k) for k in ("mAP50", "precision", "recall", "training_time_hours")
        }
        result["mitigation_block_in_metrics"] = "mitigation" in metrics

    if enabled:
        result["passed"] = bool(
            result["exit_code"] == 0
            and result["weights_written"]
            and result["finite_losses"]
            and result["mitigation_active_logged"]
            and result["mask_stats_logged"]
            and result.get("mitigation_block_in_metrics")
        )
    else:
        # Stock path: must succeed WITHOUT any mitigation machinery involved.
        result["passed"] = bool(
            result["exit_code"] == 0
            and result["weights_written"]
            and result["finite_losses"]
            and not result["mitigation_active_logged"]
            and not result["preflight_ran"]
            and not result.get("mitigation_block_in_metrics", False)
        )
    logger.info(f"Training arm '{name}': {'PASS' if result['passed'] else 'FAIL'} ({elapsed}s)")
    return result


def build_markdown_sections(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Render the report dict into Markdown sections."""
    suite = report["correctness_suite"]
    spot = report["mask_spot_checks"]
    sections: list[dict[str, Any]] = [
        {
            "heading": "Verdict",
            "content": (
                f"**{report['verdict']}** — masking correctness "
                + (
                    "proven; M4/M5 may proceed."
                    if report["verdict"] == "PASS"
                    else "NOT proven; do not start M4/M5."
                )
            ),
        },
        {
            "heading": "(a) Correctness unit suite",
            "content": f"{'✅' if suite['passed'] else '❌'} `{suite['summary']}` "
            f"(bit-identity vs stock v8DetectionLoss, exact-zero masked gradients, "
            f"criterion injection, golden train-kwargs regression).",
        },
        {
            "heading": "(b) Mask spot-checks on the real artifact",
            "content": f"{spot['images_checked']} images checked against "
            f"configs/dataset_sources.yaml expectations (nc={spot['nc']}).",
            "table": {
                "headers": ["Policy", "Images", "Trusted/nc", "Expected", "OK"],
                "rows": [
                    [
                        policy,
                        agg["images"],
                        f"{agg['trusted']}/{spot['nc']}",
                        agg["expected_trusted"],
                        "✅" if agg["all_ok"] else "❌",
                    ]
                    for policy, agg in sorted(spot["by_policy"].items())
                ],
            },
        },
    ]
    for arm_key, label in (
        ("mitigated_run", "(c) Mitigated 1-epoch run"),
        ("disabled_run", "(d) Disabled 1-epoch run"),
    ):
        arm = report.get(arm_key)
        if arm is None:
            sections.append({"heading": label, "content": "_skipped (--skip-training)_"})
            continue
        lines = [
            f"- exit code: {arm['exit_code']}",
            f"- weights written: {arm['weights_written']}",
            f"- finite losses: {arm['finite_losses']}",
            f"- runtime: {arm['seconds']} s",
        ]
        if arm["mitigation_enabled"]:
            lines += [
                f"- preflight ran: {arm['preflight_ran']}",
                f"- mitigation announced: {arm['mitigation_active_logged']}",
                f"- mask stats logged: {arm['mask_stats_logged']}",
                f"- metrics.json mitigation block: {arm.get('mitigation_block_in_metrics')}",
            ]
        else:
            lines += [
                f"- stock path clean (no preflight/trainer/mitigation traces): "
                f"{not arm['preflight_ran'] and not arm['mitigation_active_logged']}",
            ]
        status = "✅ PASS" if arm["passed"] else "❌ FAIL"
        sections.append({"heading": f"{label} — {status}", "content": "\n".join(lines)})
    return sections


def run(args: argparse.Namespace) -> int:
    """Execute the M3.5 gate and write the validation report."""
    report: dict[str, Any] = {
        "generated_at": timestamp_str(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        ).stdout.strip(),
        "artifact": args.artifact.as_posix(),
    }
    try:
        import torch
        import ultralytics

        report["environment"] = {
            "python": sys.version.split()[0],
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        }
    except ImportError as e:
        logger.error(f"torch/ultralytics unavailable: {e}")
        return 1

    report["correctness_suite"] = run_correctness_suite()
    report["mask_spot_checks"] = spot_check_masks(args.artifact)

    if args.skip_training:
        report["mitigated_run"] = None
        report["disabled_run"] = None
        checks = [report["correctness_suite"]["passed"], report["mask_spot_checks"]["passed"]]
    else:
        run_root = Path(tempfile.mkdtemp(prefix="phase4_m35_"))
        logger.info(f"Training-run workspace: {run_root}")
        try:
            report["mitigated_run"] = run_training_arm(run_root, enabled=True, name="m35_mitigated")
            report["disabled_run"] = run_training_arm(run_root, enabled=False, name="m35_disabled")
        finally:
            # Ultralytics writes label caches inside the DVC-tracked output —
            # remove them so `dvc status` stays clean after validation runs.
            for cache in (REPO_ROOT / "data" / "processed" / "labels").glob("*.cache"):
                cache.unlink()
                logger.info(f"Removed Ultralytics label cache: {cache}")
        checks = [
            report["correctness_suite"]["passed"],
            report["mask_spot_checks"]["passed"],
            report["mitigated_run"]["passed"],
            report["disabled_run"]["passed"],
        ]

    report["verdict"] = "PASS" if all(checks) else "FAIL"

    out_dir = args.report_dir
    save_json_report(report, out_dir / "masking_validation_report.json")
    save_markdown_report(
        "Phase-4 M3.5 — Masking-Correctness Validation",
        build_markdown_sections(report),
        out_dir / "masking_validation_report.md",
        metadata={
            "Verdict": report["verdict"],
            "Commit": report["git_commit"],
            "ultralytics": report["environment"]["ultralytics"],
            "torch": report["environment"]["torch"],
        },
    )
    logger.info(f"M3.5 verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Phase-4 M3.5 masking-correctness validation gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/processed/completeness.json"),
        help="Completeness artifact to spot-check.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/qa_reports/phase4_mitigation"),
        help="Output directory for the validation report.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Run only the unit suite and mask spot-checks (no training epochs).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

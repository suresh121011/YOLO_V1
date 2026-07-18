"""
scripts.dataset.18_make_release — Releases as Code (ADR-P5-07)
==================================================================

Three subcommands over ``configs/release.yaml``'s version ladder
(dataset-v0.5.0 -> v1.0.0):

    check <version>   Evaluate the track's declared gates against the
                      current build; print the report; exit 1 on FAIL.
    make <version>    Same evaluation; on a non-FAIL verdict, writes
                      ``data/releases/<version>/release_manifest.json``.
                      Refuses to write on FAIL.
    verify <version>  Re-loads an already-made manifest and checks it is
                      still self-consistent (no gate recorded FAIL — a
                      `make` never writes one, so this catches tampering).

Usage:
    python scripts/dataset/18_make_release.py check dataset-v0.5.0
    python scripts/dataset/18_make_release.py make dataset-v0.5.0
    python scripts/dataset/18_make_release.py verify dataset-v0.5.0

DVC integration: humans run ``make`` directly (not a DVC stage command),
then ``dvc commit -f record_release`` to pin ``data/releases/`` (frozen —
see docs/07 release_runbook.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.release.gates import (
    GATE_STATUS_FAIL,
    collect_license_entries,
    evaluate_release,
    load_release_config,
    read_roboflow_dataset_licenses,
)
from src.dataset.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    build_release_manifest,
)
from src.dataset.sources_config import load_sources_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_check(args: argparse.Namespace) -> int:
    """Evaluate a release track's gates and print the verdict."""
    report = evaluate_release(
        args.version,
        release_yaml_path=args.release_config,
        sources_yaml_path=args.sources_config,
        data_yaml_path=args.data,
        completeness_path=args.completeness,
        qa_report_path=args.qa_report,
        coverage_report_path=args.coverage_report,
        quality_report_path=args.quality_report,
        changelog_path=args.changelog,
        raw_root=args.raw_root,
        captures_root=args.captures_root,
        eval_report_path=args.eval_report,
        ab_benchmark_dir=args.ab_benchmark_dir,
        qa_reports_root=args.qa_reports_root,
        capture_config_path=args.capture_config,
    )
    for line in report.format_lines():
        logger.info(line)
    return 1 if report.verdict == "FAIL" else 0


def cmd_make(args: argparse.Namespace) -> int:
    """Evaluate gates; on a non-FAIL verdict, write the release manifest."""
    report = evaluate_release(
        args.version,
        release_yaml_path=args.release_config,
        sources_yaml_path=args.sources_config,
        data_yaml_path=args.data,
        completeness_path=args.completeness,
        qa_report_path=args.qa_report,
        coverage_report_path=args.coverage_report,
        quality_report_path=args.quality_report,
        changelog_path=args.changelog,
        raw_root=args.raw_root,
        captures_root=args.captures_root,
        eval_report_path=args.eval_report,
        ab_benchmark_dir=args.ab_benchmark_dir,
        qa_reports_root=args.qa_reports_root,
        capture_config_path=args.capture_config,
    )
    for line in report.format_lines():
        logger.info(line)
    if report.verdict == "FAIL":
        logger.error("Refusing to make a release with a FAIL verdict — fix the failing gate(s).")
        return 1

    quality_report = None
    if args.quality_report.exists():
        import json

        quality_report = json.loads(args.quality_report.read_text(encoding="utf-8"))

    sources_cfg = load_sources_config(args.sources_config)
    license_entries = collect_license_entries(args.raw_root, sources_cfg)
    noncommercial_sources = sorted(
        {
            str(e["source"])
            for e in license_entries
            if e.get("noncommercial") and int(e.get("image_count", 0)) > 0
        }
    )
    track = load_release_config(args.release_config)[args.version]
    roboflow_slug_licenses = {
        **read_roboflow_dataset_licenses(args.raw_root),
        **{str(k): str(v) for k, v in (track.get("roboflow_slug_licenses") or {}).items()},
    }

    manifest = build_release_manifest(
        report,
        quality_report,
        completeness_path=args.completeness,
        qa_report_path=args.qa_report,
        coverage_report_path=args.coverage_report,
        quality_report_path=args.quality_report,
        merged_manifest_path=args.merged_manifest,
        ledger_path=args.ledger_path,
        dvc_lock_path=args.dvc_lock,
        split_config_path=args.split_config,
        sources_mode=sources_cfg.mode,
        allow_noncommercial=sources_cfg.allow_noncommercial,
        noncommercial_sources=noncommercial_sources,
        roboflow_slug_licenses=roboflow_slug_licenses,
    )
    output_path = args.releases_root / args.version / RELEASE_MANIFEST_FILENAME
    manifest.save(output_path)
    logger.info(f"Release manifest written: {output_path}")
    logger.info(
        "Next: `dvc commit -f record_release`, `git add dvc.lock data/releases`, "
        "`git commit`, `git push`, `dvc push`."
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-check an already-made release manifest's self-consistency."""
    manifest_path = args.releases_root / args.version / RELEASE_MANIFEST_FILENAME
    if not manifest_path.exists():
        logger.error(f"No release manifest at {manifest_path} — run `make` first.")
        return 1

    manifest = ReleaseManifest.load(manifest_path)
    if manifest.version != args.version:
        logger.error(f"Manifest version {manifest.version!r} != requested {args.version!r}")
        return 1

    failed_gates = [g for g in manifest.gates if g.get("status") == GATE_STATUS_FAIL]
    if failed_gates:
        logger.error(f"Manifest records {len(failed_gates)} FAILed gate(s) — tampered or corrupt")
        for g in failed_gates:
            logger.error(f"  {g['gate_id']} {g['name']}: {g['details']}")
        return 1

    current_dvc_lock = args.dvc_lock
    if current_dvc_lock.exists():
        from src.utils.dataset_utils import compute_file_hash

        current_hash = compute_file_hash(current_dvc_lock)
        if current_hash != manifest.dvc_lock_sha256:
            logger.warning(
                "Current dvc.lock differs from the one recorded at release time — "
                "expected unless you've checked out this release's tag."
            )
        else:
            logger.info("dvc.lock matches the release manifest exactly.")

    logger.info(
        f"Release manifest {manifest_path} is self-consistent "
        f"({len(manifest.gates)} gates recorded)."
    )
    return 0


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("version", help="Release version, e.g. dataset-v0.5.0")
    parser.add_argument("--release-config", type=Path, default=Path("configs/release.yaml"))
    parser.add_argument("--sources-config", type=Path, default=Path("configs/dataset_sources.yaml"))
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--completeness", type=Path, default=Path("data/processed/completeness.json")
    )
    parser.add_argument(
        "--qa-report", type=Path, default=Path("data/qa_reports/annotation_qa_report.json")
    )
    parser.add_argument(
        "--coverage-report", type=Path, default=Path("data/qa_reports/coverage_report.json")
    )
    parser.add_argument(
        "--quality-report", type=Path, default=Path("data/qa_reports/dataset_quality_report.json")
    )
    parser.add_argument("--changelog", type=Path, default=Path("data/DATASET_CHANGELOG.md"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--captures-root", type=Path, default=Path("data/raw/custom_captures"))
    parser.add_argument(
        "--eval-report", type=Path, default=Path("data/qa_reports/eval_report.json")
    )
    parser.add_argument(
        "--ab-benchmark-dir", type=Path, default=Path("data/qa_reports/ab_benchmark")
    )
    parser.add_argument(
        "--merged-manifest", type=Path, default=Path("data/merged/merged_manifest.json")
    )
    parser.add_argument(
        "--ledger-path", type=Path, default=Path("data/annotation/verification_ledger.json")
    )
    parser.add_argument("--dvc-lock", type=Path, default=Path("dvc.lock"))
    parser.add_argument(
        "--split-config", type=Path, default=Path("configs/dataset_split_config.yaml")
    )
    parser.add_argument("--releases-root", type=Path, default=Path("data/releases"))
    parser.add_argument("--qa-reports-root", type=Path, default=Path("data/qa_reports"))
    parser.add_argument("--capture-config", type=Path, default=Path("configs/capture_config.yaml"))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Releases as code: check/make/verify a dataset release (Phase-5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Evaluate gates; do not write a manifest.")
    _add_common_paths(check_parser)
    check_parser.set_defaults(func=cmd_check)

    make_parser = subparsers.add_parser(
        "make", help="Evaluate gates; write the manifest on PASS/WARN."
    )
    _add_common_paths(make_parser)
    make_parser.set_defaults(func=cmd_make)

    verify_parser = subparsers.add_parser("verify", help="Re-check an existing release manifest.")
    _add_common_paths(verify_parser)
    verify_parser.set_defaults(func=cmd_verify)

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())

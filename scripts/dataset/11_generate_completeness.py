"""
scripts.dataset.11_generate_completeness — Per-Image Completeness Artifact
==========================================================================

Phase-4 (Missing Annotation Mitigation): compiles per-source/per-session
trusted-class declarations into ``data/processed/completeness.json`` — the
artifact the mitigation trainer uses to mask the classification loss — plus
a JSON/CSV/Markdown report triplet under ``data/qa_reports/``.

Inputs (all must exist; the generator hard-fails on any ambiguity):
    data/merged/merged_manifest.json                 — provenance + label_completeness
    data/processed/images/{train,val,test}           — what actually trains
    data/processed/split_report/split_summary.json   — split lineage
    configs/data.yaml                                — 23-class taxonomy
    configs/dataset_sources.yaml                     — completeness.policies section
    data/raw/custom_captures/manifests/*.json        — per-session trusted classes

Usage:
    python scripts/dataset/11_generate_completeness.py
    python scripts/dataset/11_generate_completeness.py --output out.json --report-dir reports/

DVC integration:
    Invoked by the generate_completeness stage (between split_train_val_test
    and the frozen train_yolo11n stage).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.completeness import (
    build_completeness,
    find_unused_policies,
    save_completeness,
    summarize_completeness,
    validate_completeness,
)
from src.dataset.completeness_policies import CompletenessError
from src.utils.report_utils import format_count_pct, write_all_formats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_report_sections(artifact: dict, summary: dict) -> list[dict]:
    """Assemble Markdown report sections from an artifact summary.

    Args:
        artifact: The completeness artifact.
        summary:  Output of summarize_completeness().

    Returns:
        Section list for save_markdown_report.
    """
    stats = summary["stats"]
    total = int(stats.get("images_total", 0))
    by_split = stats.get("by_split", {})
    split_rows = [[split, format_count_pct(count, total)] for split, count in by_split.items()]

    policy_table_rows = [
        [
            row["policy"],
            row["mode"],
            row["images"],
            f"{row['trusted_count']}/{row['trusted_count'] + row['untrusted_count']}",
            row["trusted_classes"] or "—",
        ]
        for row in summary["policy_rows"]
    ]

    sections: list[dict] = [
        {
            "heading": "Coverage",
            "content": (
                f"{total} processed images mapped to "
                f"{len(summary['policy_rows'])} completeness policies. "
                f"Mean trusted classes per image: "
                f"{stats.get('mean_trusted_classes_per_image', 0.0)}."
            ),
            "table": {"headers": ["Split", "Images"], "rows": split_rows},
        },
        {
            "heading": "Policies",
            "content": (
                "`trusted/total` counts trusted taxonomy classes per policy. "
                "Classes outside the trusted set are masked out of the "
                "classification loss when mitigation is enabled."
            ),
            "table": {
                "headers": ["Policy", "Mode", "Images", "Trusted/Total", "Trusted classes"],
                "rows": policy_table_rows,
            },
        },
    ]

    unused = summary["unused_policies"]
    if unused:
        sections.append(
            {
                "heading": "Warnings",
                "content": (
                    f"{len(unused)} policy(ies) are defined but referenced by no image "
                    f"(e.g. sessions fully deduplicated at merge): {', '.join(unused)}"
                ),
            }
        )
    return sections


def run(args: argparse.Namespace) -> int:
    """Build, validate, save, and report the completeness artifact.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success (warnings allowed), 1 on any error.
    """
    try:
        artifact = build_completeness(
            merged_manifest_path=args.merged_manifest,
            processed_images_root=args.images_root,
            split_summary_path=args.split_summary,
            data_yaml_path=args.data,
            sources_yaml_path=args.sources,
            capture_manifests_dir=args.capture_manifests,
        )
    except (CompletenessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Completeness generation failed: {e}")
        return 1

    # Defense in depth: the builder never emits an invalid artifact, but the
    # validator is the same code the training preflight trusts — run it here
    # so a generator bug can never ship a green artifact.
    errors = validate_completeness(artifact, data_yaml_path=args.data)
    if errors:
        for err in errors:
            logger.error(f"Artifact validation: {err}")
        return 1

    save_completeness(artifact, args.output)

    summary = summarize_completeness(artifact)
    for key in find_unused_policies(artifact):
        logger.warning(f"Unused completeness policy (no images reference it): {key}")

    paths = write_all_formats(
        report_data={
            "generated_at": artifact["generated_at"],
            "artifact_path": args.output.as_posix(),
            "taxonomy_fingerprint": artifact["taxonomy"]["fingerprint"],
            "stats": summary["stats"],
            "policies": summary["policy_rows"],
            "unused_policies": summary["unused_policies"],
        },
        csv_rows=summary["policy_rows"],
        md_title="Label-Completeness Report",
        md_sections=build_report_sections(artifact, summary),
        output_dir=args.report_dir,
        base_name="completeness_report",
        md_metadata={
            "Artifact": args.output.as_posix(),
            "Images": summary["stats"].get("images_total", 0),
            "Policies": len(summary["policy_rows"]),
            "Taxonomy": artifact["taxonomy"]["fingerprint"][:23] + "…",
        },
    )
    logger.info(f"Completeness report written: {paths['markdown']}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate per-image label-completeness metadata (Phase-4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--merged-manifest",
        type=Path,
        default=Path("data/merged/merged_manifest.json"),
        help="Merged dataset manifest (provenance + label_completeness).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/processed/images"),
        help="Processed images root containing train/val/test subdirs.",
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path("data/processed/split_report/split_summary.json"),
        help="Split summary JSON (lineage input).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Dataset taxonomy YAML.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/dataset_sources.yaml"),
        help="Dataset sources YAML with the completeness.policies section.",
    )
    parser.add_argument(
        "--capture-manifests",
        type=Path,
        default=Path("data/raw/custom_captures/manifests"),
        help="Per-session capture manifest directory (per_session policies).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/completeness.json"),
        help="Output artifact path.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/qa_reports"),
        help="Directory for the completeness_report JSON/CSV/MD triplet.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

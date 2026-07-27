"""
scripts.qa.build_comparison_report — Before/After Dataset Comparison
====================================================================

Diffs two dataset builds from the artifacts each one already produced —
``annotation_qa_report.json``, ``dataset_statistics.json``, the merged
manifest, the split summary and the completeness report. It recomputes
nothing (mirrors the L5 discipline in ``src/dataset/annotation/quality.py``):
a wrong number here means an upstream artifact was wrong, not this script.

Produced for the Phase-E ``mode: smoke`` → ``mode: full`` migration, but the
inputs are generic — any two builds whose artifacts were snapshotted can be
compared, which is what makes it release evidence rather than a one-off.

Sections: dataset size · per-class counts · public vs local contribution ·
class imbalance · completeness · QA results · leakage · and the classes still
short of the ``dataset-v1.0.0`` per-class floor (RG9 ``min_instances_per_class``).

Usage:
    python scripts/qa/build_comparison_report.py \
        --before-dir data/qa_reports/_snapshots/phase_e_before \
        --label BEFORE=smoke-public+local AFTER=full-public+local
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_helpers import load_yaml
from src.utils.report_utils import save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Sources that are public downloads; anything else is locally collected.
PUBLIC_SOURCES = frozenset({"coco", "openimages", "roboflow", "wider_face", "negatives"})

#: Classes docs/04 flags as needing Indian-home capture (configs/data.yaml
#: "Custom Data Required"). Read from config where possible, this is the
#: fallback for a config that predates the key.
CUSTOM_REQUIRED_FALLBACK = (
    "gas_cylinder",
    "medicine_strip",
    "wet_floor",
    "walking_stick",
    "support_handle",
    "stove",
    "passport",
    "cupboard",
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning(f"missing artifact: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _delta(before: float | None, after: float | None) -> str:
    """Signed change, or a marker when a side is absent."""
    if before is None or after is None:
        return "n/a"
    diff = after - before
    if isinstance(before, int) and isinstance(after, int):
        return f"{diff:+,}"
    return f"{diff:+.4f}"


def class_counts_from_manifest(manifest: dict[str, Any] | None) -> dict[str, int]:
    """Per-class box counts as recorded at merge time."""
    if not manifest:
        return {}
    return {str(k): int(v) for k, v in (manifest.get("class_counts") or {}).items()}


def source_split(manifest: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    """Accepted image counts grouped into public vs local contributions."""
    if not manifest:
        return {}
    per_source = {
        str(entry["source"]): int(entry.get("accepted", 0))
        for entry in (manifest.get("sources") or [])
    }
    public = {n: c for n, c in per_source.items() if n in PUBLIC_SOURCES}
    local = {n: c for n, c in per_source.items() if n not in PUBLIC_SOURCES}
    return {
        "per_source": per_source,
        "public": public,
        "local": local,
        "public_total": sum(public.values()),
        "local_total": sum(local.values()),
    }


def imbalance_of(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Gini / ratio block as recorded by dataset_stats.py."""
    return dict((stats or {}).get("imbalance") or {})


def qa_of(report: dict[str, Any] | None) -> dict[str, Any]:
    """The QA numbers a release gate actually reads."""
    if not report:
        return {}
    summary = report.get("summary") or {}
    checks = report.get("checks") or {}
    orchestrator = report.get("orchestrator") or {}
    return {
        "total_images": summary.get("total_images"),
        "total_labels": summary.get("total_labels"),
        "total_boxes": summary.get("total_boxes"),
        "critical_issues": summary.get("critical_issues"),
        "warning_issues": summary.get("warning_issues"),
        "train_val_leakage": (checks.get("train_val_leakage") or {}).get("count"),
        "train_test_leakage": (checks.get("train_test_leakage") or {}).get("count"),
        "license_critical": orchestrator.get("license_critical"),
        "image_quality_warnings": orchestrator.get("image_quality_warnings"),
        "annotation_sweep_warnings": orchestrator.get("annotation_sweep_warnings"),
        "l4_l5_report_warnings": orchestrator.get("l4_l5_report_warnings"),
    }


def completeness_of(report: dict[str, Any] | None) -> dict[str, Any]:
    """Coverage of the per-image completeness artifact.

    Counts live under ``stats`` (11_generate_completeness.py), not at the top
    level — reading the root returns None for every field and renders a report
    that looks generated but says nothing.
    """
    if not report:
        return {}
    stats = report.get("stats") or {}
    return {
        "images_total": stats.get("images_total"),
        "by_split": stats.get("by_split") or {},
        "by_policy": stats.get("by_policy") or {},
        "policy_count": len(report.get("policies") or []),
        "unused_policies": report.get("unused_policies") or [],
    }


def shortfall(
    class_counts: dict[str, int],
    taxonomy: list[str],
    floor: int,
    custom_required: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Classes below the v1.0 per-class instance floor, worst first.

    ``custom_required`` classes are called out separately: for those, a
    shortfall cannot be closed by a public source at all — it is an Indian-home
    capture requirement (RG9), which is the distinction the caller needs.
    """
    rows = []
    for name in taxonomy:
        count = class_counts.get(name, 0)
        if count >= floor:
            continue
        rows.append(
            {
                "class": name,
                "count": count,
                "floor": floor,
                "short_by": floor - count,
                "needs_custom_capture": name in custom_required,
            }
        )
    return sorted(rows, key=lambda r: r["count"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the comparison payload from both builds' artifacts."""
    before_dir: Path = args.before_dir
    b_qa = _load(before_dir / "annotation_qa_report.json")
    b_stats = _load(before_dir / "dataset_statistics.json")
    b_manifest = _load(before_dir / "merged_manifest.json")
    b_completeness = _load(before_dir / "completeness_report.json")

    a_qa = _load(args.qa_report)
    a_stats = _load(args.stats)
    a_manifest = _load(args.merged_manifest)
    a_completeness = _load(args.completeness_report)

    data_cfg = load_yaml(args.data)
    taxonomy = [str(n) for _, n in sorted((data_cfg.get("names") or {}).items())]

    release_cfg = load_yaml(args.release_config).get("releases", {})
    floor = int((release_cfg.get(args.release_track) or {}).get("min_instances_per_class", 0))

    b_classes = class_counts_from_manifest(b_manifest)
    a_classes = class_counts_from_manifest(a_manifest)

    custom_required = tuple(
        (b_stats or a_stats or {}).get("custom_required_counts", {}) or CUSTOM_REQUIRED_FALLBACK
    )

    return {
        "generated_at": timestamp_str(),
        "release_track": args.release_track,
        "per_class_floor": floor,
        "labels": {"before": args.before_label, "after": args.after_label},
        "size": {
            "before": qa_of(b_qa),
            "after": qa_of(a_qa),
            "delta_images": _delta(
                qa_of(b_qa).get("total_images"), qa_of(a_qa).get("total_images")
            ),
            "delta_boxes": _delta(qa_of(b_qa).get("total_boxes"), qa_of(a_qa).get("total_boxes")),
        },
        "per_class": {
            "before": b_classes,
            "after": a_classes,
            "delta": {
                name: a_classes.get(name, 0) - b_classes.get(name, 0)
                for name in sorted(set(b_classes) | set(a_classes))
            },
        },
        "contribution": {"before": source_split(b_manifest), "after": source_split(a_manifest)},
        "imbalance": {"before": imbalance_of(b_stats), "after": imbalance_of(a_stats)},
        "completeness": {
            "before": completeness_of(b_completeness),
            "after": completeness_of(a_completeness),
        },
        "shortfall_vs_v1": shortfall(a_classes, taxonomy, floor, custom_required),
        "custom_required_classes": list(custom_required),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Human-readable rendering; the JSON stays the machine-readable form."""
    before_label = payload["labels"]["before"]
    after_label = payload["labels"]["after"]
    lines = [
        "# Phase E — Before/After Dataset Comparison",
        "",
        f"Generated {payload['generated_at']} · " f"**{before_label}** → **{after_label}**",
        "",
        "## 1. Dataset size",
        "",
        f"| Metric | {before_label} | {after_label} | Δ |",
        "|---|---:|---:|---:|",
    ]
    size_b, size_a = payload["size"]["before"], payload["size"]["after"]
    for key in ("total_images", "total_labels", "total_boxes"):
        lines.append(
            f"| {key} | {size_b.get(key, 'n/a'):,} | {size_a.get(key, 'n/a'):,} | "
            f"{_delta(size_b.get(key), size_a.get(key))} |"
            if isinstance(size_b.get(key), int) and isinstance(size_a.get(key), int)
            else f"| {key} | {size_b.get(key)} | {size_a.get(key)} | n/a |"
        )

    lines += [
        "",
        "## 2. Per-class box counts",
        "",
        f"| Class | {before_label} | {after_label} | Δ |",
        "|---|---:|---:|---:|",
    ]
    for name, delta in sorted(payload["per_class"]["delta"].items(), key=lambda kv: -kv[1]):
        before = payload["per_class"]["before"].get(name, 0)
        after = payload["per_class"]["after"].get(name, 0)
        lines.append(f"| {name} | {before:,} | {after:,} | {delta:+,} |")

    lines += ["", "## 3. Public vs local contribution (accepted images)", ""]
    for side, label in (("before", before_label), ("after", after_label)):
        contribution = payload["contribution"][side]
        if not contribution:
            continue
        lines.append(f"**{label}**")
        lines.append("")
        lines.append("| Source | Accepted | Kind |")
        lines.append("|---|---:|---|")
        for source, count in sorted(contribution["per_source"].items(), key=lambda kv: -kv[1]):
            kind = "public" if source in PUBLIC_SOURCES else "local"
            lines.append(f"| {source} | {count:,} | {kind} |")
        lines.append(f"| **public total** | **{contribution['public_total']:,}** | |")
        lines.append(f"| **local total** | **{contribution['local_total']:,}** | |")
        lines.append("")

    lines += [
        "## 4. Class imbalance",
        "",
        f"| Metric | {before_label} | {after_label} | Δ |",
        "|---|---:|---:|---:|",
    ]
    imb_b, imb_a = payload["imbalance"]["before"], payload["imbalance"]["after"]
    for key in sorted(set(imb_b) | set(imb_a)):
        lines.append(
            f"| {key} | {imb_b.get(key, 'n/a')} | {imb_a.get(key, 'n/a')} | "
            f"{_delta(imb_b.get(key), imb_a.get(key))} |"
        )

    qa_a = payload["size"]["after"]
    lines += [
        "",
        "## 5. QA and leakage",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| critical issues | {qa_a.get('critical_issues')} |",
        f"| warnings | {qa_a.get('warning_issues')} |",
        f"| train/val leakage | {qa_a.get('train_val_leakage')} |",
        f"| train/test leakage | {qa_a.get('train_test_leakage')} |",
        f"| license critical | {qa_a.get('license_critical')} |",
        f"| image-quality warnings | {qa_a.get('image_quality_warnings')} |",
        f"| annotation sweep | {qa_a.get('annotation_sweep_warnings')} |",
        f"| L4/L5 report sweep | {qa_a.get('l4_l5_report_warnings')} |",
        "",
        "## 6. Completeness",
        "",
    ]
    comp_b, comp_a = payload["completeness"]["before"], payload["completeness"]["after"]
    lines += [
        f"| Metric | {before_label} | {after_label} |",
        "|---|---:|---:|",
        f"| images covered | {comp_b.get('images_total')} | {comp_a.get('images_total')} |",
        f"| policies resolved | {comp_b.get('policy_count')} | {comp_a.get('policy_count')} |",
        f"| unused policies | {len(comp_b.get('unused_policies') or [])} "
        f"| {len(comp_a.get('unused_policies') or [])} |",
        "",
        "Per-source coverage (`by_policy`):",
        "",
        f"| Source | {before_label} | {after_label} |",
        "|---|---:|---:|",
    ]
    by_policy_b = comp_b.get("by_policy") or {}
    by_policy_a = comp_a.get("by_policy") or {}
    for source in sorted(set(by_policy_b) | set(by_policy_a)):
        lines.append(
            f"| {source} | {by_policy_b.get(source, 0):,} | {by_policy_a.get(source, 0):,} |"
        )
    lines += [
        "",
        f"## 7. Classes short of the {payload['release_track']} floor "
        f"({payload['per_class_floor']} instances)",
        "",
    ]
    rows = payload["shortfall_vs_v1"]
    if not rows:
        lines.append("None — every class meets the floor.")
    else:
        lines += ["| Class | Count | Short by | Needs Indian-home capture |", "|---|---:|---:|---|"]
        for row in rows:
            mark = "**yes**" if row["needs_custom_capture"] else "no"
            lines.append(f"| {row['class']} | {row['count']:,} | {row['short_by']:,} | {mark} |")
        lines += [
            "",
            "Classes marked **yes** cannot be closed by any public source — they are "
            "the RG9 Indian-home capture requirement.",
        ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two dataset builds from their recorded artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument(
        "--qa-report", type=Path, default=Path("data/qa_reports/annotation_qa_report.json")
    )
    parser.add_argument(
        "--stats", type=Path, default=Path("data/qa_reports/dataset_statistics.json")
    )
    parser.add_argument(
        "--merged-manifest", type=Path, default=Path("data/merged/merged_manifest.json")
    )
    parser.add_argument(
        "--completeness-report",
        type=Path,
        default=Path("data/qa_reports/completeness_report.json"),
    )
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--release-config", type=Path, default=Path("configs/release.yaml"))
    parser.add_argument("--release-track", default="dataset-v1.0.0")
    parser.add_argument("--output", type=Path, default=Path("data/qa_reports"))
    parser.add_argument("--name", default="phase_e_comparison")
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    payload = build(args)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = save_json_report(payload, args.output / f"{args.name}.json")
    markdown_path = args.output / f"{args.name}.md"
    # Explicit LF: this lands beside DVC-tracked reports, and a CRLF artifact
    # makes its hash OS-dependent (the Phase-5 reproducibility remediation).
    markdown_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    for path in (json_path, markdown_path):
        logger.info(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

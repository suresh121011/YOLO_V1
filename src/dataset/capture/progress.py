"""
src.dataset.capture.progress — Collection Progress Tracking
=============================================================

Machine-checks the custom-capture collection against the Phase-3
governance targets (configs/capture_config.yaml `targets:`): per-class
instance counts vs the ≥200/class minimum, total image count vs the
2,000-image target, annotation-lifecycle breakdown, house/room/lighting
coverage, consent anomalies, and eval-set status.

Report written via src/utils/report_utils.write_all_formats to
data/qa_reports/capture_progress.{json,csv,md}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset.capture.config import CollectionTargets
from src.dataset.capture.consent import ConsentRecord, find_withdrawn_consents
from src.dataset.capture.ingest import is_eval_locked, load_session_manifests
from src.utils.report_utils import write_all_formats

logger = logging.getLogger(__name__)


@dataclass
class ProgressReport:
    """Collection progress against Phase-3 targets."""

    total_images: int = 0
    total_target: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    class_target: int = 0
    houses: set[str] = field(default_factory=set)
    houses_target: int = 0
    rooms_by_house: dict[str, set[str]] = field(default_factory=dict)
    lighting_covered: set[str] = field(default_factory=set)
    annotation_status_counts: dict[str, int] = field(default_factory=dict)
    withdrawn_sessions: dict[str, str] = field(default_factory=dict)
    eval_image_count: int = 0
    eval_locked: bool = False

    @property
    def classes_met(self) -> list[str]:
        """Custom classes that have reached the per-class target."""
        return sorted(
            name for name, count in self.class_counts.items() if count >= self.class_target
        )

    @property
    def classes_pending(self) -> dict[str, int]:
        """Custom classes still short of the target → instances remaining."""
        return {
            name: max(0, self.class_target - self.class_counts.get(name, 0))
            for name in sorted(self.class_counts)
            if self.class_counts.get(name, 0) < self.class_target
        }

    @property
    def targets_met(self) -> bool:
        """True once total images, all custom classes and house count are met."""
        return (
            self.total_images >= self.total_target
            and len(self.houses) >= self.houses_target
            and not self.classes_pending
            and not self.withdrawn_sessions
        )


def build_progress_report(
    captures_root: Path,
    eval_root: Path,
    targets: CollectionTargets,
    consent_registry: dict[str, ConsentRecord],
) -> ProgressReport:
    """Build a progress report from the on-disk capture tree.

    Args:
        captures_root:    Training captures root (data/raw/custom_captures).
        eval_root:        Locked eval set root.
        targets:          Collection targets (configs/capture_config.yaml).
        consent_registry: Loaded consent registry (possibly empty).

    Returns:
        :class:`ProgressReport`.
    """
    sessions = load_session_manifests(captures_root)

    report = ProgressReport(
        total_target=targets.total_images,
        class_target=targets.min_instances_per_class,
        houses_target=targets.min_houses,
        class_counts={name: 0 for name in targets.custom_classes},
    )

    session_refs: dict[str, str] = {}
    for session in sessions:
        report.total_images += session.image_count
        for name, count in session.class_counts.items():
            if name in report.class_counts:
                report.class_counts[name] += count
        if session.house_id:
            report.houses.add(session.house_id)
            report.rooms_by_house.setdefault(session.house_id, set()).add(session.room)
        if session.lighting:
            report.lighting_covered.add(session.lighting)
        report.annotation_status_counts[session.annotation_status] = (
            report.annotation_status_counts.get(session.annotation_status, 0) + 1
        )
        session_refs[session.session_id] = session.consent_reference

    report.withdrawn_sessions = find_withdrawn_consents(session_refs, consent_registry)

    if eval_root.exists():
        eval_sessions = load_session_manifests(eval_root)
        report.eval_image_count = sum(s.image_count for s in eval_sessions)
        report.eval_locked = is_eval_locked(eval_root)

    return report


def write_progress_report(report: ProgressReport, output_dir: Path) -> dict[str, Path]:
    """Write the progress report in JSON/CSV/Markdown.

    Returns:
        Dict of format → written Path (see write_all_formats).
    """
    json_data = {
        "total_images": report.total_images,
        "total_target": report.total_target,
        "class_counts": report.class_counts,
        "class_target": report.class_target,
        "classes_met": report.classes_met,
        "classes_pending": report.classes_pending,
        "houses": sorted(report.houses),
        "houses_target": report.houses_target,
        "rooms_by_house": {h: sorted(r) for h, r in sorted(report.rooms_by_house.items())},
        "lighting_covered": sorted(report.lighting_covered),
        "annotation_status_counts": report.annotation_status_counts,
        "withdrawn_sessions": report.withdrawn_sessions,
        "eval_image_count": report.eval_image_count,
        "eval_locked": report.eval_locked,
        "targets_met": report.targets_met,
    }

    csv_rows = [
        {
            "class": name,
            "count": count,
            "target": report.class_target,
            "met": count >= report.class_target,
        }
        for name, count in sorted(report.class_counts.items())
    ]

    class_table_rows = [
        [name, count, report.class_target, "✅" if count >= report.class_target else "⏳"]
        for name, count in sorted(report.class_counts.items())
    ]

    sections: list[dict[str, Any]] = [
        {
            "heading": "Overview",
            "content": (
                f"Total images: **{report.total_images} / {report.total_target}**  \n"
                f"Houses: **{len(report.houses)} / {report.houses_target}** "
                f"({', '.join(sorted(report.houses)) or 'none'})  \n"
                f"Targets met: **{'YES' if report.targets_met else 'no'}**"
            ),
        },
        {
            "heading": "Per-class instance counts",
            "table": {"headers": ["Class", "Count", "Target", "Met"], "rows": class_table_rows},
        },
        {
            "heading": "Annotation status",
            "content": "\n".join(
                f"- {status}: {count}"
                for status, count in sorted(report.annotation_status_counts.items())
            )
            or "(no sessions ingested)",
        },
        {
            "heading": "Coverage",
            "content": "\n".join(
                f"- {house}: {', '.join(sorted(rooms))}"
                for house, rooms in sorted(report.rooms_by_house.items())
            )
            + f"\n\nLighting covered: {', '.join(sorted(report.lighting_covered)) or '(none)'}",
        },
        {
            "heading": "Eval set",
            "content": (
                f"Images: {report.eval_image_count}  \n"
                f"Locked: {'yes' if report.eval_locked else 'no'}"
            ),
        },
        {
            "heading": "Consent anomalies",
            "content": (
                "\n".join(
                    f"- {sid}: {ref} — WITHDRAWN, remove data"
                    for sid, ref in sorted(report.withdrawn_sessions.items())
                )
                or "(none)"
            ),
        },
    ]

    return write_all_formats(
        report_data=json_data,
        csv_rows=csv_rows,
        md_title="Custom Capture Collection Progress",
        md_sections=sections,
        output_dir=output_dir,
        base_name="capture_progress",
        csv_fieldnames=["class", "count", "target", "met"],
    )

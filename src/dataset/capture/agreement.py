"""
src.dataset.capture.agreement — Dual-Annotator Agreement (IAA)
==============================================================

Inter-annotator agreement for the dual-annotator CVAT workflow
(dataset governance) and the ONLY pre-training instrument for the
wet_floor R24 decision gate (docs/04 §8): the first wet_floor pilot
session must reach the configured per-class agreement or the class is
demoted from bbox to scene-level.

Method: per image and class, boxes of the two annotators are matched
greedily by descending IoU (pairs with IoU ≥ ``iou_threshold`` match).
Agreement per class = matched / (matched + only_a + only_b). Greedy
matching is an approximation of optimal (Hungarian) assignment — fine
for a QA signal; disagreements are adjudicated in CVAT, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.capture.config import IaaSettings
from src.utils.annotation_utils import Annotation, parse_label_file

logger = logging.getLogger(__name__)


@dataclass
class ClassAgreement:
    """Agreement stats for one class across a session."""

    class_name: str
    matched: int = 0
    only_a: int = 0
    only_b: int = 0
    iou_sum: float = 0.0

    @property
    def total(self) -> int:
        """All box observations that should have matched."""
        return self.matched + self.only_a + self.only_b

    @property
    def agreement(self) -> float:
        """matched / total (1.0 when neither annotator drew this class)."""
        return self.matched / self.total if self.total else 1.0

    @property
    def mean_iou(self) -> float:
        """Mean IoU over matched pairs (0.0 when nothing matched)."""
        return self.iou_sum / self.matched if self.matched else 0.0


@dataclass
class AgreementReport:
    """Session-level dual-annotator agreement."""

    annotator_a: str
    annotator_b: str
    images_compared: int = 0
    per_class: dict[str, ClassAgreement] = field(default_factory=dict)
    per_image: dict[str, float] = field(default_factory=dict)

    @property
    def overall_agreement(self) -> float:
        """Micro-averaged agreement across all classes."""
        matched = sum(c.matched for c in self.per_class.values())
        total = sum(c.total for c in self.per_class.values())
        return matched / total if total else 1.0

    def worst_images(self, count: int = 5) -> list[tuple[str, float]]:
        """The images with the lowest per-image agreement, ascending."""
        ranked = sorted(self.per_image.items(), key=lambda kv: (kv[1], kv[0]))
        return ranked[:count]


def compute_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU of two normalized YOLO boxes (cx, cy, w, h).

    Returns:
        Intersection-over-union in [0, 1].
    """
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_staged_labels(
    staging_dir: Path, session_id: str, annotator: str
) -> dict[str, list[Annotation]]:
    """Parse one annotator's staged label files (stem → annotations)."""
    source = staging_dir / session_id / annotator
    if not source.is_dir():
        raise FileNotFoundError(f"No staged labels at {source}")
    return {path.stem: parse_label_file(path) for path in sorted(source.glob("*.txt"))}


def compare_annotators(
    labels_a: dict[str, list[Annotation]],
    labels_b: dict[str, list[Annotation]],
    iou_threshold: float,
    class_names: dict[int, str],
    annotator_a: str = "a",
    annotator_b: str = "b",
) -> AgreementReport:
    """Compute dual-annotator agreement over a session's labels.

    Args:
        labels_a:      Annotator A's labels (stem → annotations).
        labels_b:      Annotator B's labels.
        iou_threshold: Minimum IoU for two boxes to count as the same object.
        class_names:   Taxonomy mapping (id → name); unknown IDs are skipped.
        annotator_a:   Handle for reporting.
        annotator_b:   Handle for reporting.

    Returns:
        :class:`AgreementReport` with per-class and per-image breakdowns.
    """
    report = AgreementReport(annotator_a=annotator_a, annotator_b=annotator_b)
    stems = sorted(set(labels_a) | set(labels_b))
    report.images_compared = len(stems)

    for stem in stems:
        image_matched = 0
        image_total = 0
        anns_a = labels_a.get(stem, [])
        anns_b = labels_b.get(stem, [])
        class_ids = {a.class_id for a in anns_a} | {b.class_id for b in anns_b}

        for class_id in sorted(class_ids):
            name = class_names.get(class_id)
            if name is None:
                logger.warning(f"{stem}: skipping unknown class id {class_id}")
                continue
            stats = report.per_class.setdefault(name, ClassAgreement(class_name=name))

            boxes_a = [a for a in anns_a if a.class_id == class_id]
            boxes_b = [b for b in anns_b if b.class_id == class_id]
            matched, iou_sum = _greedy_match(boxes_a, boxes_b, iou_threshold)

            stats.matched += matched
            stats.iou_sum += iou_sum
            stats.only_a += len(boxes_a) - matched
            stats.only_b += len(boxes_b) - matched
            image_matched += matched
            image_total += matched + (len(boxes_a) - matched) + (len(boxes_b) - matched)

        report.per_image[stem] = image_matched / image_total if image_total else 1.0

    return report


def _greedy_match(
    boxes_a: list[Annotation],
    boxes_b: list[Annotation],
    iou_threshold: float,
) -> tuple[int, float]:
    """Greedy descending-IoU matching. Returns (matched count, IoU sum)."""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            iou = compute_iou((a.cx, a.cy, a.w, a.h), (b.cx, b.cy, b.w, b.h))
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    iou_sum = 0.0
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
        iou_sum += iou
    return matched, iou_sum


def agreement_verdict(report: AgreementReport, settings: IaaSettings) -> tuple[str, list[str]]:
    """Evaluate a report against the configured per-class agreement gates.

    Args:
        report:   Computed agreement report.
        settings: IAA thresholds (wet_floor has an R24 override).

    Returns:
        ("pass" | "fail", list of failure descriptions).
    """
    failures: list[str] = []
    for name, stats in sorted(report.per_class.items()):
        if stats.total == 0:
            continue
        gate = settings.min_agreement_for(name)
        if stats.agreement < gate:
            failures.append(
                f"{name}: agreement {stats.agreement:.2f} < required {gate:.2f} "
                f"(matched {stats.matched}, only-{report.annotator_a} {stats.only_a}, "
                f"only-{report.annotator_b} {stats.only_b})"
            )
    return ("fail" if failures else "pass", failures)


def report_as_dict(report: AgreementReport, verdict: str, failures: list[str]) -> dict[str, object]:
    """JSON-serializable form of a report (for data/qa_reports/iaa_*.json)."""
    return {
        "annotators": [report.annotator_a, report.annotator_b],
        "images_compared": report.images_compared,
        "overall_agreement": round(report.overall_agreement, 4),
        "verdict": verdict,
        "failures": failures,
        "per_class": {
            name: {
                "matched": stats.matched,
                "only_a": stats.only_a,
                "only_b": stats.only_b,
                "agreement": round(stats.agreement, 4),
                "mean_iou": round(stats.mean_iou, 4),
            }
            for name, stats in sorted(report.per_class.items())
        },
        "worst_images": [
            {"image": stem, "agreement": round(value, 4)} for stem, value in report.worst_images()
        ],
    }

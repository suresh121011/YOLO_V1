"""
src.dataset.annotation.gt_eval — Annotation-Quality Ground-Truth Eval
=====================================================================

P9 (annotation V2 plan). The annotation pipeline previously had **no**
ground-truth quality signal — "precision/recall" existed only as a
ledger-calibrated proxy in :mod:`.coverage`. This module scores predicted
detections against the held-out, human-verified eval set
(``data/eval/indian_home_v0``) to produce *real* per-class precision / recall /
F1 / mean-IoU via greedy IoU matching.

Pure geometry — no model, no GPU. Callers supply predicted boxes and GT boxes
(both normalized xywhn, per image, per class); this reuses
:func:`src.dataset.annotation.coverage.iou_xywhn`. The result feeds two things:
the audit's "no GT annotation quality" gap, and P4's prompt-gating decision
(enable an open-vocab prompt for a class only once its measured precision
clears a bar).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.dataset.annotation.coverage import iou_xywhn

Box = tuple[float, float, float, float]
# image_id -> class_id -> list of boxes
BoxesByImageClass = dict[str, dict[int, list[Box]]]

GT_EVAL_SCHEMA_VERSION = 1


@dataclass
class ClassScore:
    """Per-class detection metrics at a fixed IoU threshold."""

    class_id: int
    class_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_iou_sum: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        """Mean IoU over matched (true-positive) pairs; 0 when no TP."""
        return self.matched_iou_sum / self.tp if self.tp else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "support": self.tp + self.fn,  # GT box count for the class
        }


@dataclass
class GtEvalReport:
    """Full report: per-class scores + micro/macro aggregates."""

    iou_threshold: float
    per_class: dict[int, ClassScore] = field(default_factory=dict)
    images_scored: int = 0

    def micro(self) -> dict[str, float]:
        """Instance-weighted aggregate (sum TP/FP/FN across classes)."""
        tp = sum(c.tp for c in self.per_class.values())
        fp = sum(c.fp for c in self.per_class.values())
        fn = sum(c.fn for c in self.per_class.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    def macro(self) -> dict[str, float]:
        """Class-averaged aggregate over classes that have GT support."""
        scored = [c for c in self.per_class.values() if (c.tp + c.fn) > 0]
        if not scored:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        n = len(scored)
        return {
            "precision": round(sum(c.precision for c in scored) / n, 4),
            "recall": round(sum(c.recall for c in scored) / n, 4),
            "f1": round(sum(c.f1 for c in scored) / n, 4),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": GT_EVAL_SCHEMA_VERSION,
            "iou_threshold": self.iou_threshold,
            "images_scored": self.images_scored,
            "micro": self.micro(),
            "macro": self.macro(),
            "per_class": [self.per_class[k].as_dict() for k in sorted(self.per_class)],
        }


def _match_with_iou(
    pred_boxes: list[Box], gt_boxes: list[Box], iou_threshold: float
) -> tuple[int, int, int, float]:
    """Greedy IoU match → (tp, fp, fn, sum of matched IoUs).

    Same greedy strategy as :func:`coverage.match_candidates_to_verified`
    (each prediction takes its best unmatched GT above threshold), extended to
    also return the matched IoUs so mean-IoU can be reported.
    """
    matched_gt: set[int] = set()
    tp = 0
    iou_sum = 0.0
    for pred in pred_boxes:
        best_iou, best_g = 0.0, -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = iou_xywhn(pred, g_box)
            if iou > best_iou:
                best_iou, best_g = iou, g_idx
        if best_iou >= iou_threshold:
            matched_gt.add(best_g)
            tp += 1
            iou_sum += best_iou
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, iou_sum


def score_predictions_against_gt(
    predictions: BoxesByImageClass,
    ground_truth: BoxesByImageClass,
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> GtEvalReport:
    """Score predicted boxes against GT boxes → per-class + aggregate metrics.

    Args:
        predictions:   Predicted boxes, image_id → class_id → [xywhn boxes].
        ground_truth:  GT boxes in the same shape (the held-out eval set).
        class_names:   Taxonomy id → name (every class gets a row, even 0-support).
        iou_threshold: Minimum IoU for a prediction↔GT match (default 0.5).

    Returns:
        A :class:`GtEvalReport`. Classes with neither predictions nor GT still
        appear (all-zero) so the report is a stable, full-taxonomy artifact.
    """
    report = GtEvalReport(iou_threshold=iou_threshold)
    for cid, name in class_names.items():
        report.per_class[cid] = ClassScore(class_id=cid, class_name=name)

    image_ids = set(predictions) | set(ground_truth)
    report.images_scored = len(image_ids)
    for image_id in image_ids:
        pred_by_class = predictions.get(image_id, {})
        gt_by_class = ground_truth.get(image_id, {})
        for cid in set(pred_by_class) | set(gt_by_class):
            score = report.per_class.get(cid)
            if score is None:  # class id outside the taxonomy — skip defensively
                continue
            tp, fp, fn, iou_sum = _match_with_iou(
                pred_by_class.get(cid, []), gt_by_class.get(cid, []), iou_threshold
            )
            score.tp += tp
            score.fp += fp
            score.fn += fn
            score.matched_iou_sum += iou_sum
    return report

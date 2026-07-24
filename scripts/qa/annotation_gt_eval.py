"""
scripts.qa.annotation_gt_eval — Annotation-Quality GT Evaluation (P9)
=====================================================================

Scores a directory of predicted YOLO labels against the held-out, human-verified
eval set (``data/eval/indian_home_v0``) and writes real per-class
precision / recall / F1 / mean-IoU — the ground-truth annotation-quality signal
the pipeline lacked (audit QA-lens finding).

Pure geometry, no model / no GPU: point ``--pred-labels`` at any YOLO label dir
(e.g. an auto-annotation run over the eval images) and ``--gt-labels`` at the
eval set's labels. Produces ``data/qa_reports/annotation_gt_eval.json``.

This is a REPORT, not a release gate (non-blocking, like coverage/quality). It
also produces the per-class precision that P4's prompt-gating consumes: enable an
open-vocab prompt for a class only once its measured precision clears a bar.

Example:
    python scripts/qa/annotation_gt_eval.py \
        --gt-labels data/eval/indian_home_v0/labels \
        --pred-labels data/annotation/eval_predictions/labels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.annotation.gt_eval import BoxesByImageClass, score_predictions_against_gt
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yolo_labels(labels_dir: Path) -> BoxesByImageClass:
    """Parse a YOLO label directory into image_id → class_id → [xywhn boxes].

    Detection lines are ``class cx cy w h``. Non-conforming lines (comments,
    blanks, wrong field count, non-numeric) are skipped — QA's structural
    checks are the place to fail on malformed labels, not the scorer.
    """
    result: BoxesByImageClass = {}
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")
    for txt in sorted(labels_dir.glob("*.txt")):
        image_id = txt.stem
        by_class: dict[int, list[tuple[float, float, float, float]]] = {}
        for raw in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            by_class.setdefault(cid, []).append((cx, cy, w, h))
        result[image_id] = by_class
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score predicted YOLO labels against the eval-set GT (P9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt-labels",
        type=Path,
        default=Path("data/eval/indian_home_v0/labels"),
        help="Ground-truth (human-verified) YOLO labels directory.",
    )
    parser.add_argument(
        "--pred-labels",
        type=Path,
        required=True,
        help="Predicted YOLO labels directory (e.g. an auto-annotation run).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qa_reports/annotation_gt_eval.json"),
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="If set, print the classes whose measured precision is below this "
        "bar (P4 prompt-gating aid). Does not change the exit code.",
    )
    return parser.parse_args()


def _class_names(config_path: Path) -> dict[int, str]:
    data_cfg = load_data_config(config_path)
    names = get_class_names_from_data_yaml(data_cfg)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def main() -> int:
    args = parse_args()
    class_names = _class_names(args.config)
    gt = load_yolo_labels(args.gt_labels)
    pred = load_yolo_labels(args.pred_labels)

    report = score_predictions_against_gt(pred, gt, class_names, iou_threshold=args.iou)
    payload = report.as_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    micro, macro = report.micro(), report.macro()
    logger.info(
        f"GT eval @IoU {args.iou}: {report.images_scored} images · "
        f"micro P/R/F1 {micro['precision']}/{micro['recall']}/{micro['f1']} · "
        f"macro P/R/F1 {macro['precision']}/{macro['recall']}/{macro['f1']}"
    )
    logger.info(f"Report written to {args.output}")

    if args.min_precision is not None:
        below = [
            c.as_dict()
            for c in report.per_class.values()
            if (c.tp + c.fn) > 0 and c.precision < args.min_precision
        ]
        if below:
            names = ", ".join(f"{c['class_name']}({c['precision']})" for c in below)
            logger.warning(f"Classes below precision {args.min_precision} (do NOT prompt): {names}")
        else:
            logger.info(f"All supported classes meet precision {args.min_precision}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
scripts.training.evaluate_model — Standalone Checkpoint Evaluation CLI (M10)
==============================================================================

Evaluates ONE trained checkpoint and writes its per-class/aggregate metrics
+ confusion matrix. This answers "how good is this checkpoint" — distinct
from evaluate_mitigation.py's baseline-vs-mitigated A/B comparison (M4,
"did the mitigation help").

`--split eval` evaluates against the locked Phase-5 custom eval set
(data/eval/indian_home_v0, configs/eval_data.yaml) instead of
configs/data.yaml's train/val/test split, and additionally writes
data/qa_reports/eval_report.json — the flat artifact release gate RG10
(dataset-v1.0.0 track) checks for the existence of.

Usage:
    python scripts/training/evaluate_model.py --weights models/yolo11n/weights/best.pt --split val
    python scripts/training/evaluate_model.py --weights models/yolo11n/weights/best.pt --split eval

DVC integration: the (frozen) evaluate_yolo11n stage runs this with
--split eval once dataset-v1.0.0 + a real trained checkpoint exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.evaluation import EvalRunSpec, run_single_eval, wet_floor_ap50_checkpoint
from src.utils.config_helpers import resolve_device
from src.utils.report_utils import save_json_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

#: configs/data.yaml only declares train/val/test — the locked custom eval
#: set lives in a separate flat directory with its own data yaml.
EVAL_SPLIT_DATA_YAML = Path("configs/eval_data.yaml")
DEFAULT_DATA_YAML = Path("configs/data.yaml")
EVAL_REPORT_PATH = Path("data/qa_reports/eval_report.json")


def run(args: argparse.Namespace) -> int:
    """Evaluate one checkpoint and write its report(s).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    device = resolve_device(args.device)
    is_eval_split = args.split == "eval"
    data_yaml = args.data or (EVAL_SPLIT_DATA_YAML if is_eval_split else DEFAULT_DATA_YAML)
    # configs/eval_data.yaml declares the locked set under train/val/test
    # identically (module docstring) — "test" reads as "final held-out".
    ultralytics_split = "test" if is_eval_split else args.split
    label = args.label or args.split

    spec = EvalRunSpec(
        weights=args.weights,
        label=label,
        data_yaml=data_yaml,
        split=ultralytics_split,
        device=device,
        imgsz=args.imgsz,
        seed=args.seed,
    )
    try:
        summary = run_single_eval(spec, args.out_dir)
    except FileNotFoundError as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

    # R24 checkpoint 2 (docs/04 capture_annotation_runbook.md §8): a low
    # wet_floor AP50 on ANY real evaluation reopens the demotion path,
    # independent of --split — merged into the summary so it's visible
    # wherever this run's report lands, not only on --split eval.
    summary["wet_floor_checkpoint"] = wet_floor_ap50_checkpoint(summary["per_class"])
    if summary["wet_floor_checkpoint"]["reopen_demotion"]:
        logger.warning(
            f"R24 checkpoint 2: wet_floor AP50={summary['wet_floor_checkpoint']['ap50']} < "
            f"{summary['wet_floor_checkpoint']['ap50_threshold']} — demotion path reopened "
            "(docs/04_dataset_engineering/capture_annotation_runbook.md §8)."
        )

    if is_eval_split:
        save_json_report(summary, args.eval_report_out)
        logger.info(f"Locked eval-set report written: {args.eval_report_out}")

    aggregate = summary["aggregate"]
    logger.info(
        f"'{label}' aggregate: P={aggregate['precision']} R={aggregate['recall']} "
        f"mAP50={aggregate['mAP50']} mAP50-95={aggregate['mAP50_95']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate one trained checkpoint (Phase-5 M10).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "eval"), default="val")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override the data yaml (default: configs/data.yaml, or "
        "configs/eval_data.yaml for --split eval).",
    )
    parser.add_argument(
        "--label", type=str, default=None, help="Report label (default: the split name)."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/qa_reports/evaluation"))
    parser.add_argument(
        "--eval-report-out",
        type=Path,
        default=EVAL_REPORT_PATH,
        help="Also write the flat eval_report.json RG10 checks for (--split eval only).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

"""
scripts.training.evaluate_mitigation — Baseline vs Mitigated Evaluation CLI
===========================================================================

Evaluates two trained checkpoints (baseline and mitigated) on the same split
and writes the comparison report triplet (JSON/CSV/MD) plus per-run
confusion matrices under data/qa_reports/phase4_mitigation/.

Reading the numbers: public-source validation labels are partially
annotated, so absolute metrics underestimate untrusted classes; the
per-class DELTAS between the two arms are the meaningful signal at smoke
scale. See docs/06_training_engineering/.

Usage:
    python scripts/training/evaluate_mitigation.py \
        --baseline-weights models/benchmarks/baseline/weights/best.pt \
        --mitigated-weights models/benchmarks/mitigated/weights/best.pt
    python scripts/training/evaluate_mitigation.py --split test --imgsz 320
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.evaluation import EvalRunSpec, run_evaluation
from src.utils.config_helpers import resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    """Run both evaluations and write the comparison report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    device = resolve_device(args.device)
    specs = [
        EvalRunSpec(
            weights=args.baseline_weights,
            label="baseline",
            data_yaml=args.data,
            split=args.split,
            device=device,
            imgsz=args.imgsz,
        ),
        EvalRunSpec(
            weights=args.mitigated_weights,
            label="mitigated",
            data_yaml=args.data,
            split=args.split,
            device=device,
            imgsz=args.imgsz,
        ),
    ]
    try:
        report_path = run_evaluation(specs, args.out_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    logger.info(f"Comparison report: {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs mitigated checkpoints (Phase-4 M4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--mitigated-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/qa_reports/phase4_mitigation"),
        help="Directory for reports and confusion matrices.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

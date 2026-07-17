"""
scripts.training.benchmark_mitigation — Baseline vs Mitigated Benchmark CLI
===========================================================================

Runs the Phase-4 A/B benchmark (stock vs masked-loss training) and writes
the report triplet with per-budget PASS/FAIL under
data/qa_reports/phase4_mitigation/. Weights land in --workspace so the
evaluation CLI can score them afterwards:

    python scripts/training/benchmark_mitigation.py --smoke
    python scripts/training/evaluate_mitigation.py \
        --baseline-weights models/benchmarks/models/baseline_r0/weights/best.pt \
        --mitigated-weights models/benchmarks/models/mitigated_r0/weights/best.pt

Exit codes: 0 = benchmark PASS (all budgets met), 1 = budget breach or error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.benchmark import BenchmarkConfig, run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    """Run the benchmark.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on PASS, 1 on budget breach or error.
    """
    if args.smoke:
        config = BenchmarkConfig(device=args.device)
    else:
        config = BenchmarkConfig(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            repeats=args.repeats,
        )
    try:
        report_path = run_benchmark(config, out_dir=args.out_dir, workspace=args.workspace)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    verdict = json.loads(report_path.read_text(encoding="utf-8"))["verdict"]
    return 0 if verdict == "PASS" else 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark stock vs masked-loss training (Phase-4 M5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the smoke preset (2 epochs @ 320px, batch 8, 2 repeats).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("models/benchmarks"),
        help="Where run configs and weights are kept (for later evaluation).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/qa_reports/phase4_mitigation"),
        help="Report directory.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

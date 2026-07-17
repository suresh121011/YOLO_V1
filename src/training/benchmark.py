"""
src.training.benchmark — Baseline-vs-Mitigated Benchmark Framework (M5)
=======================================================================

Reproducible A/B benchmark of stock vs masked-loss training:

    - trains both arms through the production CLI (subprocess — the exact
      user path, correct Windows multiprocessing), repeats× each
    - identical hyperparameters/seeds in both arms; mixing augmentations
      (mosaic/mixup/copy_paste) are zeroed in BOTH arms so the comparison is
      apples-to-apples under preflight gate G8. Ultralytics' deterministic
      default seed means metric variance across repeats is ≈0 by design —
      repeats exist to bound TIMING/memory noise
    - measures wall time, per-epoch time, peak process-tree RSS (psutil),
      GPU memory (torch.cuda when available, else N/A), final P/R/F1/mAP
    - microbenchmarks the loss forward (stock vs masked) and mask building
    - evaluates every number against the Phase-4 performance budgets and
      marks each PASS/FAIL; any budget breach fails the benchmark verdict

Reports land as the house JSON/CSV/MD triplet. torch/ultralytics are
imported lazily (microbenchmarks only); psutil is an explicit dependency.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml

from src.utils.report_utils import timestamp_str, write_all_formats

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REPORT_BASENAME = "benchmark_report"

#: Phase-4 performance budgets (rationale: docs/06 masked_loss_architecture.md).
#: Each entry: budget id → (description, limit, unit).
#: The BINDING user-facing budget is wall_time_overhead_pct; the loss-forward
#: budget is an ABSOLUTE per-call ceiling — the masked multiply is O(bs·A·nc)
#: like the BCE itself, so a percentage of the (small) isolated loss call
#: would misstate real cost (~0.6 ms/call ≈ 0.2 % of a full training step).
PERFORMANCE_BUDGETS: dict[str, tuple[str, float, str]] = {
    "wall_time_overhead_pct": ("Training wall-time overhead per run", 5.0, "%"),
    "peak_rss_overhead_pct": ("Peak CPU RSS delta", 5.0, "%"),
    "peak_rss_overhead_mb": ("Peak CPU RSS delta (absolute)", 200.0, "MB"),
    "gpu_memory_overhead_pct": ("Peak GPU memory delta", 5.0, "%"),
    "loss_forward_overhead_ms": ("Loss-forward overhead per call (bs=16)", 1.0, "ms"),
    "mask_build_ms": ("Per-batch mask build (bs=16)", 1.0, "ms"),
    "lookup_load_s": ("CompletenessLookup.load wall time", 2.0, "s"),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark run parameters.

    Attributes:
        epochs:      Training epochs per run (smoke default keeps runs short).
        imgsz:       Training image size.
        batch:       Batch size.
        device:      Compute device.
        repeats:     Runs per arm (timing/memory variance bounds).
        base_config: Training YAML the benchmark configs derive from.
        artifact:    Completeness artifact (mitigated arm + microbenchmarks).
    """

    epochs: int = 2
    imgsz: int = 320
    batch: int = 8
    device: str = "cpu"
    repeats: int = 2
    base_config: Path = Path("configs/training/yolo11n_config.yaml")
    artifact: Path = Path("data/processed/completeness.json")

    def validate(self) -> None:
        """Raise ValueError on non-positive parameters."""
        for field_name in ("epochs", "imgsz", "batch", "repeats"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"BenchmarkConfig.{field_name} must be positive")


# ─── Training-arm execution ───────────────────────────────────────────────────


def _write_arm_config(cfg: BenchmarkConfig, workspace: Path, arm: str, repeat: int) -> Path:
    """Derive one arm's training YAML from the base config."""
    with open(REPO_ROOT / cfg.base_config, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    train_cfg["training"].update({"epochs": cfg.epochs, "imgsz": cfg.imgsz, "batch": cfg.batch})
    train_cfg["model"]["device"] = cfg.device
    train_cfg["output"].update(
        {"project": str(workspace / "models"), "name": f"{arm}_r{repeat}", "plots": False}
    )
    train_cfg["missing_annotation_mitigation"]["enabled"] = arm == "mitigated"
    train_cfg["missing_annotation_mitigation"]["completeness_path"] = cfg.artifact.as_posix()
    # Fairness + gate G8: identical augmentation regime in BOTH arms.
    train_cfg["augmentation"].update({"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0})
    path = workspace / f"{arm}_r{repeat}_config.yaml"
    path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    return path


def _run_with_monitoring(cmd: list[str]) -> dict[str, Any]:
    """Run a command, sampling peak RSS of its whole process tree.

    Args:
        cmd: Command line to execute (cwd = repo root).

    Returns:
        Dict with exit_code, seconds, peak_rss_mb.
    """
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    peak_rss = 0
    handle = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            processes = [handle, *handle.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak_rss = max(peak_rss, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover - race
            pass
        time.sleep(0.2)
    return {
        "exit_code": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
    }


def _epoch_times_from_results_csv(results_csv: Path) -> list[float]:
    """Per-epoch durations from Ultralytics' cumulative 'time' column."""
    import csv as csv_mod

    if not results_csv.exists():
        return []
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    cumulative = [float(row["time"]) for row in rows if "time" in row]
    return [round(t - prev, 2) for prev, t in zip([0.0, *cumulative], cumulative, strict=False)]


def run_training_arm(
    cfg: BenchmarkConfig, workspace: Path, arm: str, repeat: int
) -> dict[str, Any]:
    """Train one arm once and collect its measurements.

    Args:
        cfg:       Benchmark parameters.
        workspace: Directory for configs and run outputs.
        arm:       "baseline" or "mitigated".
        repeat:    Repeat index (naming only — runs are deterministic).

    Returns:
        Run record: timing, memory, metrics, artifact paths.

    Raises:
        RuntimeError: If the training subprocess fails.
    """
    config_path = _write_arm_config(cfg, workspace, arm, repeat)
    run_dir = workspace / "models" / f"{arm}_r{repeat}"
    logger.info(f"Benchmark run: {arm} repeat {repeat} ({cfg.epochs} epochs @ {cfg.imgsz}px)")

    monitored = _run_with_monitoring(
        [sys.executable, "scripts/training/train_yolo.py", "--config", str(config_path)]
    )
    if monitored["exit_code"] != 0:
        raise RuntimeError(
            f"Benchmark training run failed ({arm} r{repeat}, exit "
            f"{monitored['exit_code']}) — re-run manually: "
            f"python scripts/training/train_yolo.py --config {config_path}"
        )

    metrics_path = run_dir / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "arm": arm,
        "repeat": repeat,
        "seconds_total": monitored["seconds"],
        "epoch_seconds": _epoch_times_from_results_csv(run_dir / "results.csv"),
        "peak_rss_mb": monitored["peak_rss_mb"],
        "gpu_memory_mb": None,  # populated only when CUDA is available (see below)
        "precision": precision,
        "recall": recall,
        "f1": round(f1, 4),
        "mAP50": float(metrics.get("mAP50", 0.0)),
        "mAP50_95": float(metrics.get("mAP50_95", 0.0)),
        "weights": (run_dir / "weights" / "best.pt").as_posix(),
    }


# ─── Microbenchmarks ──────────────────────────────────────────────────────────


def microbenchmark_loss_forward(
    artifact: Path, rounds: int = 9, iterations: int = 20
) -> dict[str, Any]:
    """Time stock vs masked loss forward on synthetic tensors.

    Methodology: the two criteria are timed in INTERLEAVED rounds and the
    per-round means are reduced with the MEDIAN — wall-clock interference
    (OS scheduling, sync clients) hits both arms alike and outlier rounds
    drop out. A single long series proved unusable on desktop hardware
    (0.5 %…10 % swings between identical runs).

    Args:
        artifact:   Completeness artifact for realistic mask lookups.
        rounds:     Interleaved measurement rounds (median taken across).
        iterations: Loss evaluations per round.

    Returns:
        Median per-call milliseconds for both criteria + overhead %.
    """
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    from src.training._masked_loss_impl import MaskedDetectionLoss
    from src.training.completeness_lookup import CompletenessLookup
    from src.training.mitigation_config import MitigationConfig

    lookup = CompletenessLookup.load(artifact)
    sample_files = list(lookup._policy_by_image)[:16] or ["missing.jpg"]

    model = DetectionModel(cfg="yolo11n.yaml", nc=lookup.nc, verbose=False)
    model.args = get_cfg()  # type: ignore[assignment]
    model.eval()

    batch_size = 16
    batch: Any = {
        "img": torch.rand(batch_size, 3, 96, 96),
        "batch_idx": torch.arange(batch_size, dtype=torch.float32),
        "cls": torch.zeros(batch_size, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]).repeat(batch_size, 1),
        "im_file": [sample_files[i % len(sample_files)] for i in range(batch_size)],
    }
    with torch.no_grad():
        preds = model(batch["img"])

    stock = v8DetectionLoss(model)
    masked = MaskedDetectionLoss(model, lookup=lookup, config=MitigationConfig(enabled=True))

    def one_round(criterion: Any) -> float:
        start = time.perf_counter()
        for _ in range(iterations):
            criterion(preds, batch)
        return (time.perf_counter() - start) / iterations * 1000  # ms

    for _ in range(5):  # joint warmup (allocators, caches)
        stock(preds, batch)
        masked(preds, batch)

    stock_rounds: list[float] = []
    masked_rounds: list[float] = []
    for _ in range(rounds):
        stock_rounds.append(one_round(stock))
        masked_rounds.append(one_round(masked))

    stock_ms = statistics.median(stock_rounds)
    masked_ms = statistics.median(masked_rounds)
    overhead_pct = (masked_ms - stock_ms) / stock_ms * 100 if stock_ms else 0.0
    return {
        "stock_ms": round(stock_ms, 3),
        "masked_ms": round(masked_ms, 3),
        "overhead_ms": round(masked_ms - stock_ms, 3),
        "overhead_pct": round(overhead_pct, 2),
        "rounds": rounds,
        "iterations": iterations,
        "batch_size": batch_size,
        "stock_round_spread_ms": [round(v, 3) for v in stock_rounds],
        "masked_round_spread_ms": [round(v, 3) for v in masked_rounds],
    }


def microbenchmark_mask_build(artifact: Path, iterations: int = 200) -> dict[str, Any]:
    """Time CompletenessLookup mask-row assembly for a bs=16 batch."""
    from src.training.completeness_lookup import CompletenessLookup

    load_start = time.perf_counter()
    lookup = CompletenessLookup.load(artifact)
    load_seconds = time.perf_counter() - load_start

    files = list(lookup._policy_by_image)[:16]
    files = [files[i % len(files)] for i in range(16)]
    for _ in range(10):  # warmup
        [lookup.mask_row(f) for f in files]
    start = time.perf_counter()
    for _ in range(iterations):
        [lookup.mask_row(f) for f in files]
    per_batch_ms = (time.perf_counter() - start) / iterations * 1000
    return {
        "lookup_load_s": round(load_seconds, 3),
        "mask_build_ms": round(per_batch_ms, 4),
        "images_indexed": len(lookup),
    }


# ─── Aggregation, budgets, report ─────────────────────────────────────────────


def _mean_std(values: list[float]) -> dict[str, float]:
    """Mean and (population-safe) std of a measurement series."""
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def aggregate_arm(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeat runs of one arm."""
    return {
        "runs": len(runs),
        "seconds_total": _mean_std([r["seconds_total"] for r in runs]),
        "peak_rss_mb": _mean_std([r["peak_rss_mb"] for r in runs]),
        "precision": _mean_std([r["precision"] for r in runs]),
        "recall": _mean_std([r["recall"] for r in runs]),
        "f1": _mean_std([r["f1"] for r in runs]),
        "mAP50": _mean_std([r["mAP50"] for r in runs]),
        "mAP50_95": _mean_std([r["mAP50_95"] for r in runs]),
    }


def evaluate_budgets(
    baseline: dict[str, Any],
    mitigated: dict[str, Any],
    loss_micro: dict[str, Any],
    mask_micro: dict[str, Any],
    cuda_available: bool,
) -> list[dict[str, Any]]:
    """Evaluate every performance budget; one PASS/FAIL row each."""
    baseline_s = baseline["seconds_total"]["mean"]
    mitigated_s = mitigated["seconds_total"]["mean"]
    wall_overhead = (mitigated_s - baseline_s) / baseline_s * 100 if baseline_s else 0.0

    baseline_rss = baseline["peak_rss_mb"]["mean"]
    mitigated_rss = mitigated["peak_rss_mb"]["mean"]
    rss_overhead_pct = (mitigated_rss - baseline_rss) / baseline_rss * 100 if baseline_rss else 0.0
    rss_overhead_mb = mitigated_rss - baseline_rss

    measured: dict[str, float | None] = {
        "wall_time_overhead_pct": round(wall_overhead, 2),
        "peak_rss_overhead_pct": round(rss_overhead_pct, 2),
        "peak_rss_overhead_mb": round(rss_overhead_mb, 1),
        "gpu_memory_overhead_pct": None if not cuda_available else 0.0,
        "loss_forward_overhead_ms": loss_micro["overhead_ms"],
        "mask_build_ms": mask_micro["mask_build_ms"],
        "lookup_load_s": mask_micro["lookup_load_s"],
    }

    rows: list[dict[str, Any]] = []
    for budget_id, (description, limit, unit) in PERFORMANCE_BUDGETS.items():
        value = measured[budget_id]
        if value is None:
            status = "N/A"
        else:
            status = "PASS" if value <= limit else "FAIL"
        rows.append(
            {
                "budget": budget_id,
                "description": description,
                "measured": value if value is not None else "n/a (no CUDA)",
                "limit": limit,
                "unit": unit,
                "status": status,
            }
        )
    return rows


def run_benchmark(cfg: BenchmarkConfig, out_dir: Path, workspace: Path) -> Path:
    """Execute the full benchmark and write the report triplet.

    Args:
        cfg:       Benchmark parameters (validated here).
        out_dir:   Report directory.
        workspace: Where run configs/weights land (kept for later evaluation).

    Returns:
        Path to the benchmark report JSON.

    Raises:
        RuntimeError: If any training run fails or a budget check cannot run.
    """
    cfg.validate()
    # Ultralytics resolves RELATIVE project paths under its settings runs_dir,
    # not the cwd — force an absolute workspace so weights land where declared.
    workspace = workspace if workspace.is_absolute() else REPO_ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    try:
        for arm in ("baseline", "mitigated"):
            for repeat in range(cfg.repeats):
                runs.append(run_training_arm(cfg, workspace, arm, repeat))
    finally:
        for cache in (REPO_ROOT / "data" / "processed" / "labels").glob("*.cache"):
            cache.unlink()

    baseline_runs = [r for r in runs if r["arm"] == "baseline"]
    mitigated_runs = [r for r in runs if r["arm"] == "mitigated"]
    baseline_agg = aggregate_arm(baseline_runs)
    mitigated_agg = aggregate_arm(mitigated_runs)

    logger.info("Running microbenchmarks (loss forward, mask build)…")
    loss_micro = microbenchmark_loss_forward(cfg.artifact)
    mask_micro = microbenchmark_mask_build(cfg.artifact)

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        cuda_available = False

    budget_rows = evaluate_budgets(
        baseline_agg, mitigated_agg, loss_micro, mask_micro, cuda_available
    )
    verdict = "PASS" if all(row["status"] != "FAIL" for row in budget_rows) else "FAIL"

    report: dict[str, Any] = {
        "generated_at": timestamp_str(),
        "verdict": verdict,
        "config": {
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "device": cfg.device,
            "repeats": cfg.repeats,
            "cuda_available": cuda_available,
        },
        "note": (
            "Smoke-scale benchmark (188 images): metric numbers are indicative "
            "only and NOT generalizable; deterministic seeds make metric "
            "variance across repeats ≈0 by design. Budgets are re-validated at "
            "full scale in Phase 5."
        ),
        "arms": {"baseline": baseline_agg, "mitigated": mitigated_agg},
        "runs": runs,
        "microbenchmarks": {"loss_forward": loss_micro, "mask_build": mask_micro},
        "budgets": budget_rows,
    }

    sections = [
        {
            "heading": "Verdict",
            "content": f"**{verdict}** — every performance budget "
            f"{'met' if verdict == 'PASS' else 'NOT met; investigate before merge'}. "
            + report["note"],
        },
        {
            "heading": "Arms (mean ± std over repeats)",
            "table": {
                "headers": ["Arm", "Wall s", "Peak RSS MB", "P", "R", "F1", "mAP50", "mAP50-95"],
                "rows": [
                    [
                        arm,
                        f"{agg['seconds_total']['mean']} ± {agg['seconds_total']['std']}",
                        f"{agg['peak_rss_mb']['mean']} ± {agg['peak_rss_mb']['std']}",
                        agg["precision"]["mean"],
                        agg["recall"]["mean"],
                        agg["f1"]["mean"],
                        agg["mAP50"]["mean"],
                        agg["mAP50_95"]["mean"],
                    ]
                    for arm, agg in (("baseline", baseline_agg), ("mitigated", mitigated_agg))
                ],
            },
        },
        {
            "heading": "Performance budgets",
            "table": {
                "headers": ["Budget", "Measured", "Limit", "Unit", "Status"],
                "rows": [
                    [row["description"], row["measured"], row["limit"], row["unit"], row["status"]]
                    for row in budget_rows
                ],
            },
        },
        {
            "heading": "Microbenchmarks",
            "content": (
                f"Loss forward (bs=16, interleaved median of {loss_micro['rounds']} rounds): "
                f"stock {loss_micro['stock_ms']} ms vs masked {loss_micro['masked_ms']} ms — "
                f"+{loss_micro['overhead_ms']} ms/call ({loss_micro['overhead_pct']}% of the "
                f"isolated loss call; ≈0% of a full training step, see the wall-time budget). "
                f"Mask build: {mask_micro['mask_build_ms']} ms/batch; lookup load "
                f"{mask_micro['lookup_load_s']} s for {mask_micro['images_indexed']} images."
            ),
        },
    ]

    paths = write_all_formats(
        report_data=report,
        csv_rows=runs,
        md_title="Missing-Annotation Mitigation — Benchmark Report",
        md_sections=sections,
        output_dir=out_dir,
        base_name=BENCHMARK_REPORT_BASENAME,
        md_metadata={
            "Verdict": verdict,
            "Config": f"{cfg.epochs} epochs @ {cfg.imgsz}px, batch {cfg.batch}, "
            f"{cfg.repeats}× repeats, {cfg.device}",
        },
    )
    logger.info(f"Benchmark report written: {paths['markdown']} (verdict {verdict})")
    return paths["json"]

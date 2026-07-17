"""Unit tests for src.training.benchmark — aggregation, budgets, report math."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.benchmark import (
    PERFORMANCE_BUDGETS,
    BenchmarkConfig,
    _epoch_times_from_results_csv,
    _mean_std,
    aggregate_arm,
    evaluate_budgets,
)


def make_run(
    arm: str,
    seconds: float = 100.0,
    rss: float = 1000.0,
    map50: float = 0.5,
) -> dict:
    """Minimal run record as produced by run_training_arm."""
    return {
        "arm": arm,
        "repeat": 0,
        "seconds_total": seconds,
        "epoch_seconds": [seconds / 2, seconds / 2],
        "peak_rss_mb": rss,
        "gpu_memory_mb": None,
        "precision": 0.6,
        "recall": 0.5,
        "f1": 0.5455,
        "mAP50": map50,
        "mAP50_95": 0.3,
        "weights": "w.pt",
    }


def budgets_for(
    baseline_s: float,
    mitigated_s: float,
    baseline_rss: float = 1000.0,
    mitigated_rss: float = 1000.0,
    loss_overhead_ms: float = 0.2,
    mask_ms: float = 0.1,
    load_s: float = 0.05,
) -> dict[str, dict]:
    """Run evaluate_budgets on synthetic aggregates; return rows by budget id."""
    baseline = aggregate_arm([make_run("baseline", seconds=baseline_s, rss=baseline_rss)])
    mitigated = aggregate_arm([make_run("mitigated", seconds=mitigated_s, rss=mitigated_rss)])
    rows = evaluate_budgets(
        baseline,
        mitigated,
        loss_micro={"overhead_ms": loss_overhead_ms},
        mask_micro={"mask_build_ms": mask_ms, "lookup_load_s": load_s},
        cuda_available=False,
    )
    return {row["budget"]: row for row in rows}


@pytest.mark.unit
class TestAggregation:
    """mean ± std math."""

    def test_mean_std_single_value_has_zero_std(self) -> None:
        assert _mean_std([5.0]) == {"mean": 5.0, "std": 0.0}

    def test_mean_std_multiple_values(self) -> None:
        stats = _mean_std([10.0, 12.0])
        assert stats["mean"] == 11.0
        assert stats["std"] > 0

    def test_aggregate_arm_covers_all_measures(self) -> None:
        agg = aggregate_arm([make_run("baseline"), make_run("baseline", seconds=110.0)])
        assert agg["runs"] == 2
        assert agg["seconds_total"]["mean"] == 105.0
        assert set(agg) == {
            "runs",
            "seconds_total",
            "peak_rss_mb",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50_95",
        }


@pytest.mark.unit
class TestBudgets:
    """Per-budget PASS/FAIL evaluation."""

    def test_all_within_limits_pass(self) -> None:
        rows = budgets_for(baseline_s=100.0, mitigated_s=103.0)
        assert rows["wall_time_overhead_pct"]["status"] == "PASS"
        assert rows["wall_time_overhead_pct"]["measured"] == 3.0
        assert all(row["status"] in ("PASS", "N/A") for row in rows.values()), {
            k: v["status"] for k, v in rows.items()
        }

    def test_wall_time_budget_breach_fails(self) -> None:
        rows = budgets_for(baseline_s=100.0, mitigated_s=106.0)
        assert rows["wall_time_overhead_pct"]["status"] == "FAIL"

    def test_rss_absolute_budget_breach_fails(self) -> None:
        rows = budgets_for(
            baseline_s=100.0, mitigated_s=100.0, baseline_rss=8000.0, mitigated_rss=8250.0
        )
        # +250 MB is only ~3% (percent budget passes) but breaches the 200 MB cap.
        assert rows["peak_rss_overhead_pct"]["status"] == "PASS"
        assert rows["peak_rss_overhead_mb"]["status"] == "FAIL"

    def test_loss_forward_budget_is_absolute_per_call(self) -> None:
        assert (
            budgets_for(100, 100, loss_overhead_ms=0.9)["loss_forward_overhead_ms"]["status"]
            == "PASS"
        )
        assert (
            budgets_for(100, 100, loss_overhead_ms=1.1)["loss_forward_overhead_ms"]["status"]
            == "FAIL"
        )

    def test_gpu_budget_na_without_cuda(self) -> None:
        rows = budgets_for(100, 100)
        assert rows["gpu_memory_overhead_pct"]["status"] == "N/A"

    def test_negative_overhead_passes(self) -> None:
        # Mitigated faster than baseline (noise) must not fail the budget.
        rows = budgets_for(baseline_s=100.0, mitigated_s=97.0)
        assert rows["wall_time_overhead_pct"]["status"] == "PASS"

    def test_every_declared_budget_is_evaluated(self) -> None:
        rows = budgets_for(100, 100)
        assert set(rows) == set(PERFORMANCE_BUDGETS)


@pytest.mark.unit
class TestEpochTimes:
    """results.csv cumulative-time parsing."""

    def test_cumulative_to_per_epoch(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("epoch,time,train/box_loss\n1,10.0,1.0\n2,25.0,0.9\n", "utf-8")
        assert _epoch_times_from_results_csv(csv_path) == [10.0, 15.0]

    def test_missing_file_yields_empty(self, tmp_path: Path) -> None:
        assert _epoch_times_from_results_csv(tmp_path / "absent.csv") == []


@pytest.mark.unit
class TestBenchmarkConfig:
    """Config validation."""

    def test_defaults_valid(self) -> None:
        BenchmarkConfig().validate()

    def test_non_positive_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="epochs"):
            BenchmarkConfig(epochs=0).validate()
        with pytest.raises(ValueError, match="repeats"):
            BenchmarkConfig(repeats=-1).validate()

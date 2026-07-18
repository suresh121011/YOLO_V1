"""Unit tests for src.dataset.release.gates (RG1-RG10 + MODE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.dataset.release.gates import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    check_build_mode,
    evaluate_release,
    load_release_config,
    rg1_qa_check,
    rg2_completeness_freshness,
    rg3_coverage_quality,
    rg4_changelog_entry,
    rg5_working_tree_tagged,
    rg6_dvc_push_verified,
    rg7_license_gate,
    rg8_split_eval_leakage,
    rg9_capture_targets,
    rg10_ab_benchmark_eval,
)

pytestmark = pytest.mark.unit


# ─── MODE ──────────────────────────────────────────────────────────────────────


class TestCheckBuildMode:
    def test_matching_mode_passes(self) -> None:
        assert check_build_mode("full", "full").status == GATE_STATUS_PASS

    def test_mismatched_mode_fails(self) -> None:
        result = check_build_mode("smoke", "full")
        assert result.status == GATE_STATUS_FAIL
        assert "smoke" in result.details and "full" in result.details


# ─── RG1 ───────────────────────────────────────────────────────────────────────


class TestRg1QaCheck:
    def test_missing_report_fails(self) -> None:
        assert rg1_qa_check(None).status == GATE_STATUS_FAIL

    def test_clean_report_passes(self) -> None:
        report = {
            "summary": {"critical_issues": 0},
            "orchestrator": {
                "license_critical": False,
                "eval_overlap_critical": False,
                "annotation_sweep_warnings": 0,
                "l4_l5_report_warnings": 0,
            },
        }
        assert rg1_qa_check(report).status == GATE_STATUS_PASS

    def test_critical_issue_fails(self) -> None:
        report = {"summary": {"critical_issues": 2}, "orchestrator": {}}
        assert rg1_qa_check(report).status == GATE_STATUS_FAIL

    def test_license_critical_fails(self) -> None:
        report = {"summary": {"critical_issues": 0}, "orchestrator": {"license_critical": True}}
        assert rg1_qa_check(report).status == GATE_STATUS_FAIL

    def test_sweep_warnings_fail(self) -> None:
        report = {
            "summary": {"critical_issues": 0},
            "orchestrator": {"annotation_sweep_warnings": 1},
        }
        assert rg1_qa_check(report).status == GATE_STATUS_FAIL


# ─── RG2 ───────────────────────────────────────────────────────────────────────


class TestRg2CompletenessFreshness:
    def test_no_problems_passes(self) -> None:
        assert rg2_completeness_freshness([], []).status == GATE_STATUS_PASS

    def test_validation_error_fails(self) -> None:
        assert rg2_completeness_freshness(["bad taxonomy"], []).status == GATE_STATUS_FAIL

    def test_freshness_error_fails(self) -> None:
        assert rg2_completeness_freshness([], ["stale hash"]).status == GATE_STATUS_FAIL


# ─── RG3 ───────────────────────────────────────────────────────────────────────


class TestRg3CoverageQuality:
    def test_missing_reports_fail(self) -> None:
        assert rg3_coverage_quality(None, None, 0, {}).status == GATE_STATUS_FAIL

    def test_thresholds_met_passes(self) -> None:
        coverage = {"per_class": {"charger": {"coverage_score": 0.8}}}
        quality = {"verification_progress": {"ledger_stats": {"cells_verified": 100}}}
        result = rg3_coverage_quality(coverage, quality, 50, {"charger": 0.5})
        assert result.status == GATE_STATUS_PASS

    def test_insufficient_verified_cells_fails(self) -> None:
        coverage: dict[str, Any] = {"per_class": {}}
        quality = {"verification_progress": {"ledger_stats": {"cells_verified": 10}}}
        result = rg3_coverage_quality(coverage, quality, 50, {})
        assert result.status == GATE_STATUS_FAIL

    def test_below_threshold_class_fails(self) -> None:
        coverage = {"per_class": {"charger": {"coverage_score": 0.2}}}
        quality = {"verification_progress": {"ledger_stats": {"cells_verified": 100}}}
        result = rg3_coverage_quality(coverage, quality, 0, {"charger": 0.5})
        assert result.status == GATE_STATUS_FAIL

    def test_missing_class_score_fails(self) -> None:
        coverage: dict[str, Any] = {"per_class": {}}
        quality = {"verification_progress": {"ledger_stats": {"cells_verified": 100}}}
        result = rg3_coverage_quality(coverage, quality, 0, {"charger": 0.5})
        assert result.status == GATE_STATUS_FAIL


# ─── RG4 ───────────────────────────────────────────────────────────────────────


class TestRg4ChangelogEntry:
    def test_entry_present_passes(self) -> None:
        text = "# Changelog\n\n## dataset-v0.5.0 — 2026-01-01\n\nDetails.\n"
        assert rg4_changelog_entry(text, "dataset-v0.5.0").status == GATE_STATUS_PASS

    def test_missing_entry_fails(self) -> None:
        text = "# Changelog\n\n## dataset-v0.1.0-smoke — 2026-01-01\n"
        assert rg4_changelog_entry(text, "dataset-v0.5.0").status == GATE_STATUS_FAIL


# ─── RG5 ───────────────────────────────────────────────────────────────────────


class TestRg5WorkingTreeTagged:
    def test_clean_and_tagged_passes(self) -> None:
        result = rg5_working_tree_tagged("", ["dataset-v0.5.0"], "dataset-v0.5.0")
        assert result.status == GATE_STATUS_PASS

    def test_dirty_tree_fails(self) -> None:
        result = rg5_working_tree_tagged(" M foo.py\n", ["dataset-v0.5.0"], "dataset-v0.5.0")
        assert result.status == GATE_STATUS_FAIL

    def test_missing_tag_fails(self) -> None:
        result = rg5_working_tree_tagged("", [], "dataset-v0.5.0")
        assert result.status == GATE_STATUS_FAIL


# ─── RG6 ───────────────────────────────────────────────────────────────────────


class TestRg6DvcPushVerified:
    def test_empty_output_passes(self) -> None:
        assert rg6_dvc_push_verified("").status == GATE_STATUS_PASS

    def test_nonempty_output_fails(self) -> None:
        assert rg6_dvc_push_verified("modified: data/merged\n").status == GATE_STATUS_FAIL


# ─── RG7 ───────────────────────────────────────────────────────────────────────


class TestRg7LicenseGate:
    def test_no_noncommercial_data_passes(self) -> None:
        entries = [{"source": "coco", "noncommercial": False, "image_count": 53}]
        result = rg7_license_gate(entries, allow_noncommercial=True, roboflow_slug_licenses={})
        assert result.status == GATE_STATUS_PASS

    def test_noncommercial_gate_violation_fails(self) -> None:
        entries = [{"source": "wider_face", "noncommercial": True, "image_count": 60}]
        result = rg7_license_gate(entries, allow_noncommercial=False, roboflow_slug_licenses={})
        assert result.status == GATE_STATUS_FAIL

    def test_noncommercial_allowed_passes(self) -> None:
        entries = [{"source": "wider_face", "noncommercial": True, "image_count": 60}]
        result = rg7_license_gate(entries, allow_noncommercial=True, roboflow_slug_licenses={})
        assert result.status == GATE_STATUS_PASS

    def test_roboflow_without_slug_licenses_fails(self) -> None:
        entries = [{"source": "roboflow", "noncommercial": False, "image_count": 40}]
        result = rg7_license_gate(entries, allow_noncommercial=True, roboflow_slug_licenses={})
        assert result.status == GATE_STATUS_FAIL

    def test_roboflow_with_slug_licenses_passes(self) -> None:
        entries = [{"source": "roboflow", "noncommercial": False, "image_count": 40}]
        result = rg7_license_gate(
            entries, allow_noncommercial=True, roboflow_slug_licenses={"slug-a": "CC-BY-4.0"}
        )
        assert result.status == GATE_STATUS_PASS


# ─── RG8 ───────────────────────────────────────────────────────────────────────


class TestRg8SplitEvalLeakage:
    def test_zero_leakage_zero_overlap_passes(self) -> None:
        checks = {"train_val_leakage": {"count": 0}, "train_test_leakage": {"count": 0}}
        eval_set = {"overlap": {"available": False}, "house_exclusivity": {"available": False}}
        assert rg8_split_eval_leakage(checks, eval_set).status == GATE_STATUS_PASS

    def test_train_val_leakage_fails(self) -> None:
        checks = {"train_val_leakage": {"count": 3}, "train_test_leakage": {"count": 0}}
        assert rg8_split_eval_leakage(checks, {}).status == GATE_STATUS_FAIL

    def test_eval_overlap_fails(self) -> None:
        checks = {"train_val_leakage": {"count": 0}, "train_test_leakage": {"count": 0}}
        eval_set = {
            "overlap": {"available": True, "exact_overlap_count": 1, "near_overlap_count": 0}
        }
        assert rg8_split_eval_leakage(checks, eval_set).status == GATE_STATUS_FAIL

    def test_shared_house_fails(self) -> None:
        checks = {"train_val_leakage": {"count": 0}, "train_test_leakage": {"count": 0}}
        eval_set = {"house_exclusivity": {"available": True, "shared_houses": ["h01"]}}
        assert rg8_split_eval_leakage(checks, eval_set).status == GATE_STATUS_FAIL


# ─── RG9 ───────────────────────────────────────────────────────────────────────


class TestRg9CaptureTargets:
    def test_targets_met_passes(self) -> None:
        assert rg9_capture_targets(2000, 3, 2000, 3).status == GATE_STATUS_PASS

    def test_insufficient_images_fails(self) -> None:
        assert rg9_capture_targets(500, 3, 2000, 3).status == GATE_STATUS_FAIL

    def test_insufficient_houses_fails(self) -> None:
        assert rg9_capture_targets(2000, 1, 2000, 3).status == GATE_STATUS_FAIL


# ─── RG10 ──────────────────────────────────────────────────────────────────────


class TestRg10AbBenchmarkEval:
    def test_both_present_passes(self) -> None:
        assert rg10_ab_benchmark_eval(True, True).status == GATE_STATUS_PASS

    def test_missing_eval_report_fails(self) -> None:
        assert rg10_ab_benchmark_eval(False, True).status == GATE_STATUS_FAIL

    def test_missing_ab_benchmark_fails(self) -> None:
        assert rg10_ab_benchmark_eval(True, False).status == GATE_STATUS_FAIL


# ─── load_release_config ───────────────────────────────────────────────────────


class TestLoadReleaseConfig:
    def test_missing_releases_section_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "release.yaml"
        path.write_text("other: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="releases"):
            load_release_config(path)

    def test_loads_declared_tracks(self, tmp_path: Path) -> None:
        path = tmp_path / "release.yaml"
        path.write_text(
            yaml.safe_dump({"releases": {"dataset-v0.5.0": {"mode": "full", "gates": ["RG1"]}}}),
            encoding="utf-8",
        )
        releases = load_release_config(path)
        assert "dataset-v0.5.0" in releases


# ─── evaluate_release (integration over the pure gates) ──────────────────────


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestEvaluateRelease:
    def test_unknown_version_raises(self, tmp_path: Path) -> None:
        release_yaml = _write_yaml(tmp_path / "release.yaml", {"releases": {"dataset-v0.5.0": {}}})
        sources_yaml = _write_yaml(
            tmp_path / "sources.yaml", {"mode": "smoke", "sources": {"coco": {}}}
        )
        with pytest.raises(ValueError, match="Unknown release version"):
            evaluate_release(
                "dataset-v9.9.9",
                release_yaml_path=release_yaml,
                sources_yaml_path=sources_yaml,
            )

    def test_mode_mismatch_fails_on_smoke_build(self, tmp_path: Path) -> None:
        """The plan's negative-case acceptance: v0.5.0 (mode: full) on a
        smoke build must fail — before any other gate even matters."""
        release_yaml = _write_yaml(
            tmp_path / "release.yaml",
            {"releases": {"dataset-v0.5.0": {"mode": "full", "gates": ["RG1"]}}},
        )
        sources_yaml = _write_yaml(
            tmp_path / "sources.yaml",
            {"mode": "smoke", "allow_noncommercial": True, "sources": {"coco": {}}},
        )
        report = evaluate_release(
            "dataset-v0.5.0",
            release_yaml_path=release_yaml,
            sources_yaml_path=sources_yaml,
            qa_report_path=tmp_path / "missing_qa.json",
        )
        assert report.verdict == "FAIL"
        assert any(r.gate_id == "MODE" and r.status == GATE_STATUS_FAIL for r in report.results)

    def test_only_required_gates_are_evaluated(self, tmp_path: Path) -> None:
        release_yaml = _write_yaml(
            tmp_path / "release.yaml",
            {"releases": {"dataset-v0.5.0": {"mode": "smoke", "gates": ["RG1"]}}},
        )
        sources_yaml = _write_yaml(
            tmp_path / "sources.yaml",
            {"mode": "smoke", "allow_noncommercial": True, "sources": {"coco": {}}},
        )
        qa_report_path = tmp_path / "qa_report.json"
        qa_report_path.write_text(
            json.dumps(
                {
                    "summary": {"critical_issues": 0},
                    "orchestrator": {
                        "license_critical": False,
                        "eval_overlap_critical": False,
                        "annotation_sweep_warnings": 0,
                        "l4_l5_report_warnings": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        report = evaluate_release(
            "dataset-v0.5.0",
            release_yaml_path=release_yaml,
            sources_yaml_path=sources_yaml,
            qa_report_path=qa_report_path,
        )
        gate_ids = {r.gate_id for r in report.results}
        assert gate_ids == {"MODE", "RG1"}
        assert report.verdict == "PASS"

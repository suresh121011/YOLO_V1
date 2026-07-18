"""
Integration test: Phase-5 M5 release flow — gates -> manifest -> round-trip.

End-to-end on a synthetic, fully isolated fixture tree (no real repo state
touched — the M2 lesson: never smoke-test stateful CLIs against committed
project files). Drives evaluate_release() -> build_release_manifest() ->
ReleaseManifest.save()/load() exactly as scripts/dataset/18_make_release.py's
`make` command does internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.dataset.completeness import taxonomy_fingerprint
from src.dataset.release.gates import ReleaseReport, evaluate_release
from src.dataset.release.manifest import ReleaseManifest, build_release_manifest
from src.utils.dataset_utils import compute_file_hash

pytestmark = pytest.mark.integration

_NAMES = {0: "person", 1: "charger", 2: "wire"}
_NC = 3
_LIVE_FP = taxonomy_fingerprint(_NC, _NAMES)


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class _Fixture:
    """A synthetic, release-ready artifact tree under one tmp_path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_yaml = _write_yaml(
            root / "data.yaml", {"nc": _NC, "names": {str(k): v for k, v in _NAMES.items()}}
        )
        self.release_yaml = _write_yaml(
            root / "release.yaml",
            {
                "releases": {
                    "dataset-v0.5.0": {
                        "mode": "full",
                        "gates": ["RG1", "RG2", "RG3", "RG4"],
                        "min_verified_cells": 0,
                    }
                }
            },
        )
        self.sources_yaml = _write_yaml(
            root / "sources.yaml",
            {"mode": "full", "allow_noncommercial": True, "sources": {"coco": {}}},
        )
        self.qa_report = _write_json(
            root / "qa_report.json",
            {
                "summary": {"critical_issues": 0},
                "orchestrator": {
                    "license_critical": False,
                    "eval_overlap_critical": False,
                    "annotation_sweep_warnings": 0,
                    "l4_l5_report_warnings": 0,
                },
            },
        )
        merged_manifest_input = _write_json(root / "merged_manifest_input.json", {"sources": []})
        split_summary_input = _write_json(root / "split_summary_input.json", {"seed": 42})
        self.completeness = _write_json(
            root / "completeness.json",
            {
                "schema_version": 1,
                "taxonomy": {
                    "nc": _NC,
                    "names": {str(k): v for k, v in _NAMES.items()},
                    "fingerprint": _LIVE_FP,
                },
                "policies": {"coco": {"mode": "trusted_list", "trusted_class_ids": [0]}},
                "images": {"a.jpg": {"policy": "coco", "split": "train"}},
                "stats": {"images_total": 1},
                "inputs": {
                    "merged_manifest": {
                        "path": merged_manifest_input.as_posix(),
                        "sha256": compute_file_hash(merged_manifest_input),
                    },
                    "split_summary": {
                        "path": split_summary_input.as_posix(),
                        "sha256": compute_file_hash(split_summary_input),
                    },
                },
            },
        )
        self.coverage_report = _write_json(
            root / "coverage_report.json",
            {
                "schema_version": 1,
                "taxonomy_fingerprint": _LIVE_FP,
                "per_class": {"charger": {"coverage_score": 0.9}},
                "per_image": {},
                "per_image_summary": {},
                "dataset": {"residual_missing_total": 0.0, "unknown_objects_total": 0},
            },
        )
        self.quality_report_path = _write_json(
            root / "quality_report.json",
            {
                "schema_version": 1,
                "taxonomy_fingerprint": _LIVE_FP,
                "dataset_scale": {
                    "images_total": 1,
                    "images_by_split": {"train": 1},
                    "images_by_source": {"coco": 1},
                    "instances_per_class": {"person": 1},
                },
                "completeness_summary": {"masked_cell_fraction": 0.5},
                "coverage_summary": {},
                "per_class_risk": {"charger": {"coverage_score": 0.9}},
                "verification_progress": {"ledger_stats": {"cells_verified": 0}},
            },
        )
        self.changelog = root / "changelog.md"
        self.changelog.write_text(
            "# Changelog\n\n## dataset-v0.5.0 — 2026-01-01\n\nFirst full-mode release.\n",
            encoding="utf-8",
        )
        self.raw_root = root / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.captures_root = root / "custom_captures"
        self.eval_report = root / "eval_report.json"
        self.ab_benchmark_dir = root / "ab_benchmark"
        self.merged_manifest = root / "merged_manifest.json"
        self.ledger_path = root / "ledger.json"
        self.dvc_lock = root / "dvc.lock"
        self.split_config = _write_yaml(
            root / "split_config.yaml", {"split": {"strategy": "group_aware", "seed": 42}}
        )

    def evaluate(self, version: str = "dataset-v0.5.0") -> ReleaseReport:
        return evaluate_release(
            version,
            release_yaml_path=self.release_yaml,
            sources_yaml_path=self.sources_yaml,
            data_yaml_path=self.data_yaml,
            completeness_path=self.completeness,
            qa_report_path=self.qa_report,
            coverage_report_path=self.coverage_report,
            quality_report_path=self.quality_report_path,
            changelog_path=self.changelog,
            raw_root=self.raw_root,
            captures_root=self.captures_root,
            eval_report_path=self.eval_report,
            ab_benchmark_dir=self.ab_benchmark_dir,
        )


class TestReleaseFlowEndToEnd:
    def test_clean_fixture_passes_and_produces_a_valid_manifest(self, tmp_path: Path) -> None:
        fixture = _Fixture(tmp_path)
        report = fixture.evaluate()
        assert report.verdict == "PASS"

        quality_report = json.loads(fixture.quality_report_path.read_text(encoding="utf-8"))
        manifest = build_release_manifest(
            report,
            quality_report,
            completeness_path=fixture.completeness,
            qa_report_path=fixture.qa_report,
            coverage_report_path=fixture.coverage_report,
            quality_report_path=fixture.quality_report_path,
            merged_manifest_path=fixture.merged_manifest,
            ledger_path=fixture.ledger_path,
            dvc_lock_path=fixture.dvc_lock,
            split_config_path=fixture.split_config,
            sources_mode="full",
            allow_noncommercial=True,
            noncommercial_sources=[],
            roboflow_slug_licenses={},
            params_files=(),
        )

        manifest_path = tmp_path / "releases" / "dataset-v0.5.0" / "release_manifest.json"
        manifest.save(manifest_path)

        reloaded = ReleaseManifest.load(manifest_path)
        assert reloaded.version == "dataset-v0.5.0"
        assert reloaded.mode == "full"
        assert reloaded.taxonomy_fingerprint == _LIVE_FP
        assert reloaded.counts["images_total"] == 1
        assert all(g["status"] == "pass" for g in reloaded.gates)

    def test_qa_critical_fails_the_whole_release(self, tmp_path: Path) -> None:
        fixture = _Fixture(tmp_path)
        fixture.qa_report.write_text(
            json.dumps({"summary": {"critical_issues": 1}, "orchestrator": {}}), encoding="utf-8"
        )
        report = fixture.evaluate()
        assert report.verdict == "FAIL"
        assert any(f.gate_id == "RG1" for f in report.failures())

    def test_mode_mismatch_fails_regardless_of_other_gates(self, tmp_path: Path) -> None:
        fixture = _Fixture(tmp_path)
        fixture.sources_yaml = _write_yaml(
            tmp_path / "sources.yaml",
            {"mode": "smoke", "allow_noncommercial": True, "sources": {"coco": {}}},
        )
        report = fixture.evaluate()
        assert report.verdict == "FAIL"
        assert any(f.gate_id == "MODE" for f in report.failures())

    def test_missing_changelog_entry_fails_rg4_only(self, tmp_path: Path) -> None:
        fixture = _Fixture(tmp_path)
        fixture.changelog.write_text("# Changelog\n\n## dataset-v0.1.0-smoke\n", encoding="utf-8")
        report = fixture.evaluate()
        assert report.verdict == "FAIL"
        failing_ids = {f.gate_id for f in report.failures()}
        assert failing_ids == {"RG4"}

    def test_below_threshold_coverage_fails_rg3(self, tmp_path: Path) -> None:
        fixture = _Fixture(tmp_path)
        fixture.release_yaml = _write_yaml(
            tmp_path / "release.yaml",
            {
                "releases": {
                    "dataset-v0.5.0": {
                        "mode": "full",
                        "gates": ["RG1", "RG2", "RG3", "RG4"],
                        "min_coverage_score": {"charger": 0.95},
                    }
                }
            },
        )
        report = fixture.evaluate()
        assert report.verdict == "FAIL"
        assert any(f.gate_id == "RG3" for f in report.failures())

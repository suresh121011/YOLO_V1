"""Unit tests for src.dataset.release.manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.dataset.release.gates import GateResult, ReleaseReport
from src.dataset.release.manifest import ReleaseManifest, build_release_manifest

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _passing_report(version: str = "dataset-v0.5.0") -> ReleaseReport:
    return ReleaseReport(
        version=version,
        required_gate_ids=("RG1", "RG4", "RG6"),
        results=(
            GateResult("MODE", "build-mode", "pass", "ok"),
            GateResult("RG1", "qa-check", "pass", "ok"),
            GateResult("RG4", "changelog-entry", "pass", "ok"),
            GateResult("RG6", "dvc-push-verified", "pass", "ok"),
        ),
    )


class TestBuildReleaseManifest:
    def test_basic_fields_populated(self, tmp_path: Path) -> None:
        report = _passing_report()
        quality_report = {
            "taxonomy_fingerprint": "sha256:abc",
            "dataset_scale": {
                "images_total": 188,
                "images_by_split": {"train": 164, "val": 12, "test": 12},
                "images_by_source": {"coco": 53},
                "instances_per_class": {"person": 202},
            },
        }
        split_config = _write(
            tmp_path / "split.yaml",
            yaml.safe_dump({"split": {"strategy": "group_aware", "seed": 42}}),
        )

        manifest = build_release_manifest(
            report,
            quality_report,
            completeness_path=tmp_path / "missing_completeness.json",
            qa_report_path=tmp_path / "missing_qa.json",
            coverage_report_path=tmp_path / "missing_coverage.json",
            quality_report_path=tmp_path / "missing_quality.json",
            merged_manifest_path=tmp_path / "missing_merged.json",
            ledger_path=tmp_path / "missing_ledger.json",
            dvc_lock_path=tmp_path / "missing_dvc.lock",
            split_config_path=split_config,
            sources_mode="full",
            allow_noncommercial=True,
            noncommercial_sources=["wider_face"],
            roboflow_slug_licenses={},
            params_files=(),
        )

        assert isinstance(manifest, ReleaseManifest)
        assert manifest.version == "dataset-v0.5.0"
        assert manifest.git_tag == "dataset-v0.5.0"
        assert manifest.mode == "full"
        assert manifest.split_strategy == "group_aware"
        assert manifest.taxonomy_fingerprint == "sha256:abc"
        assert manifest.counts["images_total"] == 188
        assert manifest.counts["by_split"] == {"train": 164, "val": 12, "test": 12}
        assert manifest.licenses["noncommercial_sources"] == ["wider_face"]
        assert manifest.changelog_entry_present is True
        assert manifest.dvc_push_verified is True
        assert manifest.reproducibility["seed"] == 42
        assert len(manifest.gates) == 4

    def test_missing_artifacts_hash_to_empty_string(self, tmp_path: Path) -> None:
        manifest = build_release_manifest(
            _passing_report(),
            None,
            completeness_path=tmp_path / "nope.json",
            qa_report_path=tmp_path / "nope2.json",
            coverage_report_path=tmp_path / "nope3.json",
            quality_report_path=tmp_path / "nope4.json",
            merged_manifest_path=tmp_path / "nope5.json",
            ledger_path=tmp_path / "nope6.json",
            dvc_lock_path=tmp_path / "nope7.lock",
            split_config_path=tmp_path / "nope8.yaml",
            params_files=(),
        )
        assert all(v == "" for v in manifest.artifact_hashes.values())

    def test_ungated_flags_are_false(self, tmp_path: Path) -> None:
        """A gate never evaluated (e.g. RG4/RG6 absent from the track) leaves
        the corresponding boolean flag False, not a stale True."""
        report = ReleaseReport(
            version="dataset-v0.5.0",
            required_gate_ids=("RG1",),
            results=(
                GateResult("MODE", "build-mode", "pass", "ok"),
                GateResult("RG1", "qa-check", "pass", "ok"),
            ),
        )
        manifest = build_release_manifest(
            report,
            None,
            completeness_path=tmp_path / "nope.json",
            qa_report_path=tmp_path / "nope2.json",
            coverage_report_path=tmp_path / "nope3.json",
            quality_report_path=tmp_path / "nope4.json",
            merged_manifest_path=tmp_path / "nope5.json",
            ledger_path=tmp_path / "nope6.json",
            dvc_lock_path=tmp_path / "nope7.lock",
            split_config_path=tmp_path / "nope8.yaml",
            params_files=(),
        )
        assert manifest.changelog_entry_present is False
        assert manifest.dvc_push_verified is False


class TestReleaseManifestRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        manifest = build_release_manifest(
            _passing_report(),
            {"taxonomy_fingerprint": "sha256:abc", "dataset_scale": {}},
            completeness_path=tmp_path / "nope.json",
            qa_report_path=tmp_path / "nope2.json",
            coverage_report_path=tmp_path / "nope3.json",
            quality_report_path=tmp_path / "nope4.json",
            merged_manifest_path=tmp_path / "nope5.json",
            ledger_path=tmp_path / "nope6.json",
            dvc_lock_path=tmp_path / "nope7.lock",
            split_config_path=tmp_path / "nope8.yaml",
            params_files=(),
        )
        path = tmp_path / "release_manifest.json"
        manifest.save(path)

        loaded = ReleaseManifest.load(path)
        assert loaded.version == manifest.version
        assert loaded.gates == manifest.gates
        assert loaded.schema_version == manifest.schema_version

    def test_saved_json_is_well_formed(self, tmp_path: Path) -> None:
        manifest = build_release_manifest(
            _passing_report(),
            None,
            completeness_path=tmp_path / "nope.json",
            qa_report_path=tmp_path / "nope2.json",
            coverage_report_path=tmp_path / "nope3.json",
            quality_report_path=tmp_path / "nope4.json",
            merged_manifest_path=tmp_path / "nope5.json",
            ledger_path=tmp_path / "nope6.json",
            dvc_lock_path=tmp_path / "nope7.lock",
            split_config_path=tmp_path / "nope8.yaml",
            params_files=(),
        )
        path = tmp_path / "release_manifest.json"
        manifest.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "dataset-v0.5.0"

"""Unit tests for src.training.preflight — mitigation gates G1–G8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset.annotation.ledger import new_ledger, recompute_stats, record_verdict, save_ledger
from src.dataset.completeness import build_completeness, save_completeness, taxonomy_fingerprint
from src.dataset.manifest import MergedManifest
from src.training.completeness_lookup import CompletenessLookup, UnknownImageError
from src.training.mitigation_config import MitigationConfig
from src.training.preflight import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_SKIPPED,
    GATE_STATUS_WARN,
    PreflightReport,
    effective_mixing_augs,
    run_preflight,
)

_DATA_YAML = """\
nc: 3
names:
  0: person
  1: face
  2: knife
"""

_SOURCES_YAML = """\
mode: smoke
completeness:
  policies:
    coco: trusted_list
    negatives: verified_absence_all
sources:
  coco:
    trusted_classes: [person, knife]
  negatives:
    trusted_classes: []
"""

#: train_cfg with mixing augmentations explicitly off (G8-clean).
_CLEAN_TRAIN_CFG: dict[str, Any] = {
    "augmentation": {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}
}


class Env:
    """Synthetic processed-dataset environment for gate tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_yaml = root / "data.yaml"
        self.artifact_path = root / "processed" / "completeness.json"
        self.processed_root = root / "processed"
        self.merged_manifest_path = root / "merged" / "merged_manifest.json"
        self.ledger_path = root / "annotation" / "verification_ledger.json"
        self.verified_labels_dir = root / "annotation" / "verified_labels"
        self.mitigation = MitigationConfig(enabled=True, completeness_path=self.artifact_path)

    def gates(self, train_cfg: dict[str, Any] | None = None) -> PreflightReport:
        """Run preflight against this environment (tmp-scoped, isolated from
        the real repo's ledger/verified_labels/merged manifest)."""
        return run_preflight(
            self.mitigation,
            data_yaml_path=self.data_yaml,
            train_cfg=_CLEAN_TRAIN_CFG if train_cfg is None else train_cfg,
            processed_root=self.processed_root,
            ledger_path=self.ledger_path,
            verified_labels_dir=self.verified_labels_dir,
            merged_manifest_path=self.merged_manifest_path,
        )

    def gate(self, gate_id: str, train_cfg: dict[str, Any] | None = None) -> Any:
        """Run preflight and return one gate's result."""
        report = self.gates(train_cfg)
        return next(r for r in report.results if r.gate_id == gate_id)


def make_env(tmp_path: Path) -> Env:
    """Create a valid two-source environment (coco + negatives)."""
    env = Env(tmp_path)
    env.data_yaml.write_text(_DATA_YAML, encoding="utf-8")
    sources_yaml = tmp_path / "dataset_sources.yaml"
    sources_yaml.write_text(_SOURCES_YAML, encoding="utf-8")

    images = {
        "coco_0001.jpg": ("train", "coco"),
        "coco_0002.jpg": ("val", "coco"),
        "negatives_0001.jpg": ("train", "negatives"),
    }
    for name, (split, _src) in images.items():
        img = env.processed_root / "images" / split / name
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8fake")

    merged_path = env.merged_manifest_path
    MergedManifest(
        image_provenance={n: s for n, (_sp, s) in images.items()},
        label_completeness={"coco": ["person", "knife"], "negatives": []},
    ).save(merged_path)

    split_summary = env.processed_root / "split_report" / "split_summary.json"
    split_summary.parent.mkdir(parents=True, exist_ok=True)
    split_summary.write_text(json.dumps({"seed": 7}), encoding="utf-8")

    artifact = build_completeness(
        merged_manifest_path=merged_path,
        processed_images_root=env.processed_root / "images",
        split_summary_path=split_summary,
        data_yaml_path=env.data_yaml,
        sources_yaml_path=sources_yaml,
        capture_manifests_dir=None,
    )
    save_completeness(artifact, env.artifact_path)
    return env


@pytest.mark.unit
class TestVerdictAggregation:
    """Report verdict semantics."""

    def test_all_gates_pass_on_valid_env(self, tmp_path: Path) -> None:
        report = make_env(tmp_path).gates()
        by_id = {r.gate_id: r.status for r in report.results}
        # G5 depends on the local ultralytics install; every filesystem gate
        # must pass outright.
        for gate_id in ("G1", "G2", "G3", "G4", "G6", "G7", "G8"):
            assert by_id[gate_id] == GATE_STATUS_PASS, f"{gate_id}: {by_id}"
        assert report.verdict in ("PASS", "FAIL")  # FAIL only if no ultralytics

    def test_failures_lists_only_failed_gates(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.artifact_path.unlink()
        report = env.gates()
        assert report.verdict == "FAIL"
        assert any(r.gate_id == "G1" for r in report.failures())

    def test_to_dict_is_json_serializable(self, tmp_path: Path) -> None:
        report = make_env(tmp_path).gates()
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["verdict"] == report.verdict
        assert len(payload["gates"]) == len(report.results)

    def test_format_lines_end_with_verdict(self, tmp_path: Path) -> None:
        report = make_env(tmp_path).gates()
        lines = report.format_lines()
        assert lines[-1] == f"Preflight verdict: {report.verdict}"


@pytest.mark.unit
class TestGateG1Artifact:
    """G1: artifact existence and parseability."""

    def test_missing_artifact_fails_and_skips_dependents(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.artifact_path.unlink()
        report = env.gates()
        by_id = {r.gate_id: r.status for r in report.results}
        assert by_id["G1"] == GATE_STATUS_FAIL
        for dependent in ("G2", "G3", "G4", "G7"):
            assert by_id[dependent] == GATE_STATUS_SKIPPED

    def test_corrupt_json_fails_g1(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.artifact_path.write_text("{not json", encoding="utf-8")
        assert env.gate("G1").status == GATE_STATUS_FAIL


@pytest.mark.unit
class TestGateG2Taxonomy:
    """G2: live taxonomy fingerprint match."""

    def test_taxonomy_drift_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.data_yaml.write_text(_DATA_YAML.replace("2: knife", "2: blade"), encoding="utf-8")
        result = env.gate("G2")
        assert result.status == GATE_STATUS_FAIL
        assert "generate_completeness" in result.details


@pytest.mark.unit
class TestGateG3Coverage:
    """G3: train/val disk images vs artifact records."""

    def test_uncovered_image_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        stray = env.processed_root / "images" / "train" / "coco_9999.jpg"
        stray.write_bytes(b"\xff\xd8fake")
        result = env.gate("G3")
        assert result.status == GATE_STATUS_FAIL
        assert "coco_9999.jpg" in result.details

    def test_split_mismatch_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        src = env.processed_root / "images" / "train" / "coco_0001.jpg"
        dst = env.processed_root / "images" / "val" / "coco_0001.jpg"
        src.replace(dst)
        result = env.gate("G3")
        assert result.status == GATE_STATUS_FAIL
        assert "split mismatch" in result.details

    def test_stale_record_warns(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        (env.processed_root / "images" / "train" / "coco_0001.jpg").unlink()
        result = env.gate("G3")
        assert result.status == GATE_STATUS_WARN
        assert "stale" in result.details

    def test_empty_dataset_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        for split in ("train", "val"):
            for img in (env.processed_root / "images" / split).glob("*.jpg"):
                img.unlink()
        assert env.gate("G3").status == GATE_STATUS_FAIL


@pytest.mark.unit
class TestGateG4Consistency:
    """G4: artifact self-consistency."""

    def test_orphan_reference_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        artifact = json.loads(env.artifact_path.read_text(encoding="utf-8"))
        artifact["images"]["coco_0001.jpg"]["policy"] = "ghost"
        env.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        result = env.gate("G4")
        assert result.status == GATE_STATUS_FAIL
        assert "orphan" in result.details

    def test_unused_policy_warns(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        artifact = json.loads(env.artifact_path.read_text(encoding="utf-8"))
        artifact["policies"]["spare"] = {"mode": "trusted_list", "trusted_class_ids": [0]}
        env.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        result = env.gate("G4")
        assert result.status == GATE_STATUS_WARN
        assert "spare" in result.details


@pytest.mark.unit
class TestGateG7Freshness:
    """G7: recorded input hashes vs disk."""

    def test_changed_merged_manifest_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        merged = tmp_path / "merged" / "merged_manifest.json"
        merged.write_text(merged.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = env.gate("G7")
        assert result.status == GATE_STATUS_FAIL
        assert "changed since generation" in result.details

    def test_missing_input_file_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        (env.processed_root / "split_report" / "split_summary.json").unlink()
        result = env.gate("G7")
        assert result.status == GATE_STATUS_FAIL
        assert "missing on disk" in result.details


@pytest.mark.unit
class TestGateG8MixingAugs:
    """G8: mosaic/mixup/copy_paste vs the configured policy."""

    def test_explicit_zeros_pass(self, tmp_path: Path) -> None:
        assert make_env(tmp_path).gate("G8").status == GATE_STATUS_PASS

    def test_active_mosaic_fails_under_forbid(self, tmp_path: Path) -> None:
        result = make_env(tmp_path).gate("G8", {"augmentation": {"mosaic": 0.8}})
        assert result.status == GATE_STATUS_FAIL
        assert "mosaic" in result.details

    def test_missing_aug_section_uses_ultralytics_defaults(self, tmp_path: Path) -> None:
        # No augmentation section ⇒ Ultralytics defaults apply (mosaic=1.0).
        result = make_env(tmp_path).gate("G8", {})
        assert result.status == GATE_STATUS_FAIL

    def test_warn_policy_downgrades_to_warning(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.mitigation = env.mitigation.with_overrides(mixing_augmentation_policy="warn")
        result = env.gate("G8", {"augmentation": {"mosaic": 0.8}})
        assert result.status == GATE_STATUS_WARN

    def test_ignore_policy_passes(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        env.mitigation = env.mitigation.with_overrides(mixing_augmentation_policy="ignore")
        assert env.gate("G8", {"augmentation": {"mosaic": 0.8}}).status == GATE_STATUS_PASS

    def test_effective_augs_partial_section_fills_script_defaults(self) -> None:
        effective = effective_mixing_augs({"augmentation": {"mosaic": 0.0}})
        # mixup/copy_paste fall back to train_yolo.py's defaults (0.1).
        assert effective == {"mosaic": 0.0, "mixup": 0.1, "copy_paste": 0.1}

    def test_effective_augs_no_section_is_ultralytics_defaults(self) -> None:
        assert effective_mixing_augs({}) == {"mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0}


@pytest.mark.unit
class TestGateG9LedgerConsistency:
    """G9: ledger <-> verified_labels <-> provenance + taxonomy fp."""

    def test_missing_ledger_file_is_pass(self, tmp_path: Path) -> None:
        assert make_env(tmp_path).gate("G9").status == GATE_STATUS_PASS

    def test_empty_ledger_is_pass(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        save_ledger(new_ledger(), env.ledger_path)
        assert env.gate("G9").status == GATE_STATUS_PASS

    def test_consistent_entry_passes(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "coco",
            "face",
            "present_labeled",
            [(0.5, 0.5, 0.1, 0.1)],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        save_ledger(ledger, env.ledger_path)
        env.verified_labels_dir.mkdir(parents=True, exist_ok=True)
        (env.verified_labels_dir / "coco_0001.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
        assert env.gate("G9").status == GATE_STATUS_PASS

    def test_missing_verified_labels_file_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "coco",
            "face",
            "present_labeled",
            [(0.5, 0.5, 0.1, 0.1)],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        save_ledger(ledger, env.ledger_path)
        result = env.gate("G9")
        assert result.status == GATE_STATUS_FAIL
        assert "no verified_labels file" in result.details

    def test_box_count_mismatch_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "coco",
            "face",
            "present_labeled",
            [(0.5, 0.5, 0.1, 0.1)],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        save_ledger(ledger, env.ledger_path)
        env.verified_labels_dir.mkdir(parents=True, exist_ok=True)
        (env.verified_labels_dir / "coco_0001.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8"
        )
        result = env.gate("G9")
        assert result.status == GATE_STATUS_FAIL
        assert "box count" in result.details

    def test_source_mismatch_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "negatives",  # actually provenanced to 'coco'
            "face",
            "verified_absent",
            [],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        save_ledger(ledger, env.ledger_path)
        result = env.gate("G9")
        assert result.status == GATE_STATUS_FAIL
        assert "!= provenance" in result.details

    def test_unknown_image_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "ghost.jpg",
            "coco",
            "face",
            "verified_absent",
            [],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        save_ledger(ledger, env.ledger_path)
        result = env.gate("G9")
        assert result.status == GATE_STATUS_FAIL
        assert "absent from merged manifest" in result.details

    def test_taxonomy_drift_fails(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "coco",
            "face",
            "verified_absent",
            [],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        recompute_stats(ledger, "sha256:stale")
        save_ledger(ledger, env.ledger_path)
        result = env.gate("G9")
        assert result.status == GATE_STATUS_FAIL
        assert "fingerprint drift" in result.details

    def test_matching_taxonomy_fingerprint_passes(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        ledger = new_ledger()
        record_verdict(
            ledger,
            "coco_0001.jpg",
            "coco",
            "face",
            "verified_absent",
            [],
            "vb001",
            "anno_1",
            "cvat",
            "",
        )
        live_fp = taxonomy_fingerprint(3, {0: "person", 1: "face", 2: "knife"})
        recompute_stats(ledger, live_fp)
        save_ledger(ledger, env.ledger_path)
        assert env.gate("G9").status == GATE_STATUS_PASS


@pytest.mark.unit
class TestCompletenessLookup:
    """Runtime lookup reader."""

    def test_mask_rows_and_policies(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        lookup = CompletenessLookup.load(env.artifact_path)
        assert len(lookup) == 3
        assert lookup.nc == 3
        assert lookup.policy_for("coco_0001.jpg") == "coco"
        assert lookup.mask_row("coco_0001.jpg") == (1, 0, 1)  # person, knife
        assert lookup.mask_row("negatives_0001.jpg") == (1, 1, 1)  # all-ones

    def test_full_paths_resolve_by_basename(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        lookup = CompletenessLookup.load(env.artifact_path)
        assert lookup.mask_row(str(tmp_path / "anywhere" / "coco_0002.jpg")) == (1, 0, 1)

    def test_unknown_image_raises(self, tmp_path: Path) -> None:
        lookup = CompletenessLookup.load(make_env(tmp_path).artifact_path)
        with pytest.raises(UnknownImageError, match="generate_completeness"):
            lookup.mask_row("never_seen.jpg")

    def test_fingerprint_mismatch_raises(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        with pytest.raises(ValueError, match="taxonomy"):
            CompletenessLookup.load(env.artifact_path, expected_fingerprint="sha256:other")

    def test_invalid_artifact_rejected_at_load(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        artifact = json.loads(env.artifact_path.read_text(encoding="utf-8"))
        artifact["images"]["coco_0001.jpg"]["policy"] = "ghost"
        env.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="failed validation"):
            CompletenessLookup.load(env.artifact_path)

    def test_coverage_reports_missing_and_stale(self, tmp_path: Path) -> None:
        lookup = CompletenessLookup.load(make_env(tmp_path).artifact_path)
        missing, stale = lookup.coverage(["coco_0001.jpg", "new_9999.jpg"])
        assert missing == ["new_9999.jpg"]
        assert set(stale) == {"coco_0002.jpg", "negatives_0001.jpg"}

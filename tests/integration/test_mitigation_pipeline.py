"""Integration tests: mitigation pipeline against the real repo configs.

Unit tests cover the pipeline on synthetic configs; these tests pin the REAL
shipped configuration files to the pipeline's contracts, so config drift in
the repository itself fails CI (no DVC data needed — data-dependent checks
live in the unit suite's synthetic environments and the M3.5 gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.completeness import build_completeness, taxonomy_fingerprint
from src.dataset.completeness_policies import registered_policy_modes
from src.dataset.manifest import MergedManifest
from src.training.completeness_lookup import CompletenessLookup
from src.training.mitigation_config import MitigationConfig
from src.training.preflight import run_preflight
from src.utils.config_helpers import (
    get_class_names_from_data_yaml,
    load_data_config,
    load_training_config,
    load_yaml,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def sources_cfg() -> dict:
    return load_yaml(PROJECT_ROOT / "configs" / "dataset_sources.yaml")


@pytest.fixture
def data_cfg() -> dict:
    return load_data_config(PROJECT_ROOT / "configs" / "data.yaml")


@pytest.mark.integration
class TestShippedConfigConsistency:
    """The real config files must satisfy the completeness contracts."""

    def test_every_source_has_a_registered_completeness_policy(self, sources_cfg: dict) -> None:
        policies = sources_cfg["completeness"]["policies"]
        modes = set(registered_policy_modes())
        for source in sources_cfg["sources"]:
            assert source in policies, f"source '{source}' missing from completeness.policies"
            assert (
                policies[source] in modes
            ), f"source '{source}' uses unregistered mode '{policies[source]}'"

    def test_trusted_classes_resolve_against_live_taxonomy(
        self, sources_cfg: dict, data_cfg: dict
    ) -> None:
        names = set(get_class_names_from_data_yaml(data_cfg).values())
        for source, spec in sources_cfg["sources"].items():
            unknown = set(spec.get("trusted_classes", [])) - names
            assert not unknown, f"source '{source}' trusts unknown classes {sorted(unknown)}"

    def test_negatives_policy_is_verified_absence(self, sources_cfg: dict) -> None:
        # The all-ones inversion must stay declared, never name-inferred.
        assert sources_cfg["completeness"]["policies"]["negatives"] == "verified_absence_all"
        assert sources_cfg["sources"]["negatives"]["trusted_classes"] == []

    @pytest.mark.parametrize("config_name", ["yolo11n_config.yaml", "yolo11s_config.yaml"])
    def test_training_configs_parse_with_mitigation_disabled(self, config_name: str) -> None:
        train_cfg = load_training_config(PROJECT_ROOT / "configs" / "training" / config_name)
        mitigation = MitigationConfig.from_training_config(train_cfg)
        assert mitigation.enabled is False  # backward-compat default
        assert mitigation.mixing_augmentation_policy == "forbid"
        mitigation.validate()

    def test_live_taxonomy_fingerprint_is_stable(self, data_cfg: dict) -> None:
        names = get_class_names_from_data_yaml(data_cfg)
        fp1 = taxonomy_fingerprint(int(data_cfg["nc"]), names)
        fp2 = taxonomy_fingerprint(int(data_cfg["nc"]), dict(reversed(list(names.items()))))
        assert fp1 == fp2
        assert fp1.startswith("sha256:")


@pytest.mark.integration
class TestEndToEndSyntheticPipeline:
    """generate → save → lookup → preflight on a synthetic dataset using the
    REAL 23-class taxonomy (no repo data/ dependence — CI has no DVC data)."""

    def _build_env(self, tmp_path: Path) -> tuple[Path, Path, dict]:
        data_yaml = PROJECT_ROOT / "configs" / "data.yaml"
        names = get_class_names_from_data_yaml(load_data_config(data_yaml))

        sources_yaml = tmp_path / "sources.yaml"
        sources_yaml.write_text(
            "mode: smoke\n"
            "completeness:\n"
            "  policies:\n"
            "    coco: trusted_list\n"
            "    negatives: verified_absence_all\n"
            "sources:\n"
            "  coco:\n"
            "    trusted_classes: [person, face]\n"
            "  negatives:\n"
            "    trusted_classes: []\n",
            encoding="utf-8",
        )

        images = {
            "coco_0001.jpg": ("train", "coco"),
            "coco_0002.jpg": ("val", "coco"),
            "negatives_0001.jpg": ("train", "negatives"),
        }
        images_root = tmp_path / "processed" / "images"
        for name, (split, _src) in images.items():
            target = images_root / split / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xd8synthetic")

        merged = tmp_path / "merged_manifest.json"
        MergedManifest(
            image_provenance={n: s for n, (_sp, s) in images.items()},
            label_completeness={"coco": ["person", "face"], "negatives": []},
        ).save(merged)

        split_summary = tmp_path / "processed" / "split_report" / "split_summary.json"
        split_summary.parent.mkdir(parents=True, exist_ok=True)
        split_summary.write_text(json.dumps({"seed": 42}), encoding="utf-8")

        artifact = build_completeness(
            merged_manifest_path=merged,
            processed_images_root=images_root,
            split_summary_path=split_summary,
            data_yaml_path=data_yaml,
            sources_yaml_path=sources_yaml,
            capture_manifests_dir=None,
        )
        artifact_path = tmp_path / "processed" / "completeness.json"
        from src.dataset.completeness import save_completeness

        save_completeness(artifact, artifact_path)
        return artifact_path, tmp_path / "processed", names

    def test_full_flow_reaches_green_preflight(self, tmp_path: Path) -> None:
        artifact_path, processed_root, names = self._build_env(tmp_path)

        lookup = CompletenessLookup.load(artifact_path)
        assert lookup.nc == len(names) == 23
        assert sum(lookup.mask_row("coco_0001.jpg")) == 2  # person + face
        assert sum(lookup.mask_row("negatives_0001.jpg")) == 23  # all-ones

        mitigation = MitigationConfig(enabled=True, completeness_path=artifact_path)
        report = run_preflight(
            mitigation,
            data_yaml_path=PROJECT_ROOT / "configs" / "data.yaml",
            train_cfg={"augmentation": {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}},
            processed_root=processed_root,
        )
        by_id = {r.gate_id: r.status for r in report.results}
        for gate_id in ("G1", "G2", "G3", "G4", "G6", "G7", "G8"):
            assert by_id[gate_id] == "pass", report.format_lines()

    def test_taxonomy_drift_detected_end_to_end(self, tmp_path: Path) -> None:
        artifact_path, processed_root, _names = self._build_env(tmp_path)
        drifted_yaml = tmp_path / "drifted_data.yaml"
        original = (PROJECT_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8")
        drifted_yaml.write_text(original.replace("0: person", "0: human"), encoding="utf-8")

        mitigation = MitigationConfig(enabled=True, completeness_path=artifact_path)
        report = run_preflight(
            mitigation,
            data_yaml_path=drifted_yaml,
            train_cfg={"augmentation": {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}},
            processed_root=processed_root,
        )
        g2 = next(r for r in report.results if r.gate_id == "G2")
        assert g2.status == "fail"
        assert report.verdict == "FAIL"

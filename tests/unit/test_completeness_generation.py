"""Unit tests for src.dataset.completeness — artifact builder and validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset.completeness import (
    COMPLETENESS_SCHEMA_VERSION,
    build_completeness,
    find_unused_policies,
    load_completeness,
    save_completeness,
    summarize_completeness,
    taxonomy_fingerprint,
    validate_completeness,
)
from src.dataset.completeness_policies import CompletenessError
from src.dataset.manifest import CaptureSessionManifest, MergedManifest

_NC = 5
_NAMES = {0: "person", 1: "face", 2: "knife", 3: "door", 4: "stove"}

_DATA_YAML = """\
path: ./data/processed
train: images/train
val: images/val
test: images/test
nc: 5
names:
  0: person
  1: face
  2: knife
  3: door
  4: stove
"""

_SOURCES_YAML = """\
mode: smoke
completeness:
  policies:
    coco: trusted_list
    negatives: verified_absence_all
    custom_captures: per_session
sources:
  coco:
    trusted_classes: [person, knife]
  negatives:
    trusted_classes: []
  custom_captures:
    trusted_classes: []
"""


class Env:
    """Synthetic pipeline environment under one tmp_path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_yaml = root / "data.yaml"
        self.sources_yaml = root / "dataset_sources.yaml"
        self.merged_manifest = root / "merged" / "merged_manifest.json"
        self.images_root = root / "processed" / "images"
        self.split_summary = root / "processed" / "split_report" / "split_summary.json"
        self.capture_manifests = root / "captures" / "manifests"

    def build(self) -> dict[str, Any]:
        """Run build_completeness against this environment."""
        return build_completeness(
            merged_manifest_path=self.merged_manifest,
            processed_images_root=self.images_root,
            split_summary_path=self.split_summary,
            data_yaml_path=self.data_yaml,
            sources_yaml_path=self.sources_yaml,
            capture_manifests_dir=self.capture_manifests,
        )


def make_env(
    tmp_path: Path,
    images: dict[str, tuple[str, str]] | None = None,
    label_completeness: dict[str, list[str]] | None = None,
    sources_yaml: str = _SOURCES_YAML,
    sessions: dict[str, list[str]] | None = None,
) -> Env:
    """Create a synthetic environment.

    Args:
        tmp_path:           pytest tmp dir.
        images:             image name → (split, source). Defaults to a
                            coco+negatives+captures trio.
        label_completeness: merged-manifest label_completeness override.
        sources_yaml:       dataset_sources.yaml content.
        sessions:           session_id → trusted class names (finalized).

    Returns:
        A ready Env.
    """
    env = Env(tmp_path)
    env.data_yaml.write_text(_DATA_YAML, encoding="utf-8")
    env.sources_yaml.write_text(sources_yaml, encoding="utf-8")

    if images is None:
        images = {
            "coco_0001.jpg": ("train", "coco"),
            "coco_0002.jpg": ("val", "coco"),
            "negatives_0001.jpg": ("train", "negatives"),
            "custom_captures_h01_kitchen_s001_0001.jpg": ("train", "custom_captures"),
        }
    if label_completeness is None:
        label_completeness = {"coco": ["person", "knife"], "negatives": [], "custom_captures": []}
    if sessions is None:
        sessions = {"h01_kitchen_s001": ["knife", "stove"]}

    for name, (split, _source) in images.items():
        img = env.images_root / split / name
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8fake-jpeg")

    manifest = MergedManifest(
        image_provenance={name: source for name, (_split, source) in images.items()},
        label_completeness=label_completeness,
    )
    manifest.save(env.merged_manifest)

    env.split_summary.parent.mkdir(parents=True, exist_ok=True)
    env.split_summary.write_text(json.dumps({"seed": 42, "strategy": "group_aware"}), "utf-8")

    for session_id, trusted in sessions.items():
        env.capture_manifests.mkdir(parents=True, exist_ok=True)
        CaptureSessionManifest(
            source="custom_captures",
            session_id=session_id,
            annotation_status="finalized",
            trusted_classes=trusted,
        ).save(env.capture_manifests / f"{session_id}.json")

    return env


@pytest.mark.unit
class TestBuildCompleteness:
    """Builder happy path and hard-fail semantics."""

    def test_happy_path_policies_and_images(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()

        assert artifact["schema_version"] == COMPLETENESS_SCHEMA_VERSION
        assert artifact["policies"] == {
            "coco": {"mode": "trusted_list", "trusted_class_ids": [0, 2]},
            "custom_captures/h01_kitchen_s001": {
                "mode": "per_session",
                "trusted_class_ids": [2, 4],
            },
            "negatives": {"mode": "verified_absence_all", "trusted_class_ids": [0, 1, 2, 3, 4]},
        }
        assert artifact["images"]["coco_0001.jpg"] == {"policy": "coco", "split": "train"}
        assert artifact["images"]["custom_captures_h01_kitchen_s001_0001.jpg"] == {
            "policy": "custom_captures/h01_kitchen_s001",
            "split": "train",
        }
        assert artifact["stats"]["images_total"] == 4
        assert artifact["stats"]["by_split"] == {"train": 3, "val": 1}

    def test_negatives_get_all_ones_mask(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        assert artifact["policies"]["negatives"]["trusted_class_ids"] == list(range(_NC))

    def test_taxonomy_block_matches_data_yaml(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        assert artifact["taxonomy"]["nc"] == _NC
        assert artifact["taxonomy"]["names"]["4"] == "stove"
        assert artifact["taxonomy"]["fingerprint"] == taxonomy_fingerprint(_NC, _NAMES)

    def test_inputs_carry_hashes_and_split_seed(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        assert artifact["inputs"]["merged_manifest"]["sha256"]
        assert artifact["inputs"]["split_summary"]["seed"] == 42
        assert artifact["inputs"]["dataset_sources_mode"] == "smoke"

    def test_unknown_image_in_processed_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        stray = env.images_root / "train" / "mystery_0001.jpg"
        stray.write_bytes(b"\xff\xd8fake")
        with pytest.raises(CompletenessError, match="image_provenance"):
            env.build()

    def test_source_without_policy_is_rejected(self, tmp_path: Path) -> None:
        yaml_without_negatives = _SOURCES_YAML.replace("    negatives: verified_absence_all\n", "")
        env = make_env(tmp_path, sources_yaml=yaml_without_negatives)
        with pytest.raises(CompletenessError, match="unsupported datasets are rejected"):
            env.build()

    def test_missing_completeness_section_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path, sources_yaml="mode: smoke\nsources: {}\n")
        with pytest.raises(CompletenessError, match="completeness.policies"):
            env.build()

    def test_config_manifest_drift_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path, label_completeness={"coco": ["person"], "negatives": []})
        with pytest.raises(CompletenessError, match="drift"):
            env.build()

    def test_unknown_class_name_is_error(self, tmp_path: Path) -> None:
        env = make_env(
            tmp_path,
            label_completeness={"coco": ["person", "dragon"], "negatives": []},
            sources_yaml=_SOURCES_YAML.replace("[person, knife]", "[person, dragon]"),
        )
        with pytest.raises(CompletenessError, match="dragon"):
            env.build()

    def test_duplicate_filename_across_splits_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        dupe = env.images_root / "val" / "coco_0001.jpg"
        dupe.write_bytes(b"\xff\xd8fake")
        with pytest.raises(CompletenessError, match="Duplicate image filename"):
            env.build()

    def test_no_processed_images_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path, images={})
        # make_env writes no files for empty images; manifest is empty too.
        with pytest.raises(CompletenessError, match="No processed images"):
            env.build()

    def test_unfinalized_session_fails_build(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        CaptureSessionManifest(
            source="custom_captures",
            session_id="h02_hall_s001",
            annotation_status="staged",
            trusted_classes=["door"],
        ).save(env.capture_manifests / "h02_hall_s001.json")
        with pytest.raises(CompletenessError, match="finalized"):
            env.build()

    def test_capture_image_without_session_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path, sessions={})
        with pytest.raises(CompletenessError, match="no finalized capture session"):
            env.build()

    def test_fingerprint_stability_and_sensitivity(self) -> None:
        fp1 = taxonomy_fingerprint(_NC, dict(_NAMES))
        fp2 = taxonomy_fingerprint(_NC, dict(reversed(list(_NAMES.items()))))
        assert fp1 == fp2  # order-independent
        renamed = dict(_NAMES)
        renamed[4] = "gas_stove"
        assert taxonomy_fingerprint(_NC, renamed) != fp1


@pytest.mark.unit
class TestRoundTripAndValidation:
    """save/load round trip plus validator error surfaces."""

    def test_round_trip_validates_clean(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        artifact = env.build()
        out = tmp_path / "completeness.json"
        save_completeness(artifact, out)
        loaded = load_completeness(out)
        assert loaded == artifact
        assert validate_completeness(loaded, data_yaml_path=env.data_yaml) == []

    def test_unknown_keys_are_tolerated(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["future_extension"] = {"anything": 1}
        assert validate_completeness(artifact) == []

    def test_duplicate_json_keys_rejected_at_load(self, tmp_path: Path) -> None:
        out = tmp_path / "completeness.json"
        out.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
        with pytest.raises(CompletenessError, match="Duplicate JSON key"):
            load_completeness(out)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_completeness(tmp_path / "absent.json")

    def test_orphan_policy_reference_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["images"]["coco_0001.jpg"]["policy"] = "ghost_policy"
        errors = validate_completeness(artifact)
        assert any("orphan" in e for e in errors)

    def test_unsupported_schema_version_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["schema_version"] = 99
        assert any("schema_version" in e for e in validate_completeness(artifact))

    def test_taxonomy_drift_vs_live_config_is_error(self, tmp_path: Path) -> None:
        env = make_env(tmp_path)
        artifact = env.build()
        env.data_yaml.write_text(_DATA_YAML.replace("4: stove", "4: gas_stove"), encoding="utf-8")
        errors = validate_completeness(artifact, data_yaml_path=env.data_yaml)
        assert any("Taxonomy drift" in e for e in errors)

    def test_corrupt_fingerprint_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["taxonomy"]["fingerprint"] = "sha256:deadbeef"
        assert any("fingerprint" in e for e in validate_completeness(artifact))

    def test_verified_absence_all_must_cover_all_classes(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["policies"]["negatives"]["trusted_class_ids"] = [0, 1]
        errors = validate_completeness(artifact)
        assert any("verified_absence_all" in e for e in errors)

    def test_out_of_range_and_unsorted_ids_are_errors(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["policies"]["coco"]["trusted_class_ids"] = [2, 0, 99]
        errors = validate_completeness(artifact)
        assert any("outside" in e for e in errors)
        assert any("sorted and unique" in e for e in errors)

    def test_invalid_split_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["images"]["coco_0001.jpg"]["split"] = "holdout"
        assert any("invalid split" in e for e in validate_completeness(artifact))

    def test_stats_total_mismatch_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["stats"]["images_total"] = 999
        assert any("images_total" in e for e in validate_completeness(artifact))

    def test_unregistered_mode_is_error(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        artifact["policies"]["coco"]["mode"] = "vibes"
        assert any("unregistered mode" in e for e in validate_completeness(artifact))


@pytest.mark.unit
class TestSummaryAndWarnings:
    """Report summary and non-fatal warnings."""

    def test_unused_policy_is_warning_not_error(self, tmp_path: Path) -> None:
        env = make_env(
            tmp_path, sessions={"h01_kitchen_s001": ["knife", "stove"], "h09_unused_s001": ["door"]}
        )
        artifact = env.build()
        assert validate_completeness(artifact) == []
        assert find_unused_policies(artifact) == ["custom_captures/h09_unused_s001"]

    def test_summary_rows_are_report_ready(self, tmp_path: Path) -> None:
        artifact = make_env(tmp_path).build()
        summary = summarize_completeness(artifact)
        rows = {row["policy"]: row for row in summary["policy_rows"]}
        assert rows["coco"]["trusted_count"] == 2
        assert rows["coco"]["untrusted_count"] == _NC - 2
        assert rows["negatives"]["untrusted_count"] == 0
        assert rows["coco"]["trusted_classes"] == "person knife"
        assert summary["stats"]["images_total"] == 4

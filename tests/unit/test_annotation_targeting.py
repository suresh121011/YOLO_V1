"""Unit tests for untrusted-cell targeting (src/dataset/annotation/targeting.py)."""

from __future__ import annotations

import pytest

from src.dataset.annotation.base import AnnotationError, BackendConfig
from src.dataset.annotation.targeting import build_targets, promptable_class_ids
from src.dataset.manifest import MergedManifest

pytestmark = pytest.mark.unit

_IDS = {"person": 0, "face": 1, "medicine_bottle": 3, "charger": 10, "wire": 11, "cupboard": 14}

_POLICIES = {
    "coco": "trusted_list",
    "openimages": "trusted_list_with_ledger",
    "negatives": "verified_absence_all",
    "custom_captures": "per_session",
}


def _config(prompts: dict[str, list[str]] | None = None) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "fake",
        {
            "enabled": True,
            "prompts": (
                prompts
                if prompts is not None
                else {"charger": ["c"], "wire": ["w"], "cupboard": ["cb"], "face": []}
            ),
            "thresholds": {"default": 0.25},
        },
    )


def _manifest(
    provenance: dict[str, str] | None = None,
    completeness: dict[str, list[str]] | None = None,
) -> MergedManifest:
    return MergedManifest(
        image_provenance=(
            provenance
            if provenance is not None
            else {
                "coco_1.jpg": "coco",
                "openimages_1.jpg": "openimages",
                "negatives_1.jpg": "negatives",
            }
        ),
        label_completeness=(
            completeness
            if completeness is not None
            else {
                "coco": ["person"],
                "openimages": ["cupboard"],
                "negatives": [],
            }
        ),
    )


class TestPromptableClassIds:
    def test_only_nonempty_prompts_count(self) -> None:
        assert promptable_class_ids(_config(), _IDS) == (10, 11, 14)

    def test_unknown_prompted_class_raises(self) -> None:
        config = _config({"not_a_class": ["x"]})
        with pytest.raises(AnnotationError, match="not_a_class"):
            promptable_class_ids(config, _IDS)


class TestBuildTargets:
    def test_untrusted_promptable_classes_targeted(self) -> None:
        targets = build_targets(_manifest(), _POLICIES, (10, 11, 14), _IDS)
        # coco trusts person only → all three promptable classes targeted
        assert targets["coco_1.jpg"] == (10, 11, 14)

    def test_trusted_classes_excluded(self) -> None:
        targets = build_targets(_manifest(), _POLICIES, (10, 11, 14), _IDS)
        # openimages trusts cupboard → only charger/wire remain
        assert targets["openimages_1.jpg"] == (10, 11)

    def test_verified_absence_all_source_skipped(self) -> None:
        targets = build_targets(_manifest(), _POLICIES, (10, 11, 14), _IDS)
        assert "negatives_1.jpg" not in targets

    def test_per_session_source_skipped(self) -> None:
        manifest = _manifest(
            provenance={"custom_captures_h01_s001_0001.jpg": "custom_captures"},
            completeness={"custom_captures": []},
        )
        assert build_targets(manifest, _POLICIES, (10, 11), _IDS) == {}

    def test_verified_cells_excluded(self) -> None:
        targets = build_targets(
            _manifest(),
            _POLICIES,
            (10, 11, 14),
            _IDS,
            verified_cells={"coco_1.jpg": frozenset({10, 14})},
        )
        assert targets["coco_1.jpg"] == (11,)

    def test_fully_covered_image_omitted(self) -> None:
        targets = build_targets(
            _manifest(),
            _POLICIES,
            (10, 11, 14),
            _IDS,
            verified_cells={"coco_1.jpg": frozenset({10, 11, 14})},
        )
        assert "coco_1.jpg" not in targets

    def test_missing_policy_entry_raises(self) -> None:
        with pytest.raises(AnnotationError, match="completeness.policies"):
            build_targets(_manifest(), {"coco": "trusted_list"}, (10,), _IDS)

    def test_unknown_policy_mode_raises(self) -> None:
        policies = dict(_POLICIES, coco="mystery_mode")
        with pytest.raises(AnnotationError, match="mystery_mode"):
            build_targets(_manifest(), policies, (10,), _IDS)

    def test_provenance_without_completeness_raises(self) -> None:
        manifest = _manifest(
            provenance={"roboflow_1.jpg": "roboflow"},
            completeness={"coco": ["person"]},
        )
        policies = dict(_POLICIES, roboflow="trusted_list")
        with pytest.raises(AnnotationError, match="roboflow"):
            build_targets(manifest, policies, (10,), _IDS)

    def test_unknown_trusted_name_raises(self) -> None:
        manifest = _manifest(
            completeness={"coco": ["ghost_class"], "openimages": [], "negatives": []}
        )
        with pytest.raises(AnnotationError, match="ghost_class"):
            build_targets(manifest, _POLICIES, (10,), _IDS)

    def test_empty_promptable_set_yields_no_targets(self) -> None:
        assert build_targets(_manifest(), _POLICIES, (), _IDS) == {}

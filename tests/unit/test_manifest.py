"""Unit tests for src.dataset.manifest — provenance manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.manifest import (
    CaptureSessionManifest,
    MergedManifest,
    SourceManifest,
)


@pytest.mark.unit
class TestSourceManifest:
    """SourceManifest construction and JSON round-trip."""

    def test_defaults(self) -> None:
        manifest = SourceManifest(source="coco")
        assert manifest.source == "coco"
        assert manifest.image_count == 0
        assert manifest.trusted_classes == []
        assert manifest.schema_version == 1
        assert manifest.retrieved_at  # auto-populated timestamp

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        manifest = SourceManifest(
            source="coco",
            license="CC BY 4.0",
            url="http://example.test/coco",
            query={"limit": 60, "mode": "smoke"},
            image_count=2,
            class_counts={"person": 3, "chair": 1},
            trusted_classes=["person", "chair"],
            image_hashes={"a.jpg": "abc123"},
        )
        path = tmp_path / "sub" / "manifest.json"
        manifest.save(path)

        loaded = SourceManifest.load(path)
        assert loaded == manifest

    def test_load_ignores_unknown_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"source": "coco", "future_field": 42}), encoding="utf-8")
        loaded = SourceManifest.load(path)
        assert loaded.source == "coco"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SourceManifest.load(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            SourceManifest.load(path)

    def test_load_non_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError):
            SourceManifest.load(path)


@pytest.mark.unit
class TestCaptureSessionManifest:
    """Capture-session provenance (Phase-3 ingest schema)."""

    def test_round_trip_with_session_fields(self, tmp_path: Path) -> None:
        manifest = CaptureSessionManifest(
            source="custom_capture",
            house_id="H003",
            room="kitchen",
            capture_device="Pixel 6",
            lighting="low-light",
            captured_at="2026-07-01",
            consent_reference="CONSENT-2026-003",
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)

        loaded = CaptureSessionManifest.load(path)
        assert loaded.house_id == "H003"
        assert loaded.consent_reference == "CONSENT-2026-003"
        assert loaded == manifest

    def test_no_pii_fields_exist(self) -> None:
        # Governance: manifests reference consent records by ID only.
        field_names = set(CaptureSessionManifest(source="x").to_dict())
        assert "consent_reference" in field_names
        for banned in ("name", "address", "phone", "email"):
            assert banned not in field_names


@pytest.mark.unit
class TestMergedManifest:
    """Merged-dataset lineage record."""

    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = MergedManifest(
            sources=[{"source": "coco", "total": 60, "accepted": 58, "duplicates": 2}],
            image_provenance={"img1.jpg": "coco"},
            duplicates_removed=2,
            filtered_out=1,
            class_counts={"person": 40},
            label_completeness={"coco": ["person", "chair"]},
        )
        path = tmp_path / "merged_manifest.json"
        manifest.save(path)

        loaded = MergedManifest.load(path)
        assert loaded == manifest
        assert loaded.label_completeness["coco"] == ["person", "chair"]

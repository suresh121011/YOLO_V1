"""Unit tests for src.dataset.capture.progress — collection progress tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.capture.config import CollectionTargets
from src.dataset.capture.consent import ConsentRecord
from src.dataset.capture.ingest import lock_eval_set
from src.dataset.capture.progress import build_progress_report, write_progress_report
from src.dataset.manifest import CaptureSessionManifest

_TARGETS = CollectionTargets(
    total_images=100,
    min_instances_per_class=10,
    custom_classes=("stove", "gas_cylinder", "wet_floor"),
    min_houses=2,
)


def _session(
    root: Path,
    session_id: str,
    house_id: str,
    room: str,
    lighting: str = "daylight",
    image_count: int = 5,
    class_counts: dict[str, int] | None = None,
    annotation_status: str = "unannotated",
    consent_reference: str = "",
) -> None:
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    CaptureSessionManifest(
        source="custom_captures",
        session_id=session_id,
        house_id=house_id,
        room=room,
        lighting=lighting,
        image_count=image_count,
        class_counts=class_counts or {},
        annotation_status=annotation_status,
        consent_reference=consent_reference,
    ).save(manifests_dir / f"{session_id}.json")


@pytest.mark.unit
class TestBuildProgressReport:
    """Aggregation across sessions."""

    def test_empty_tree_reports_zero(self, tmp_path: Path) -> None:
        report = build_progress_report(tmp_path / "captures", tmp_path / "eval", _TARGETS, {})
        assert report.total_images == 0
        assert report.houses == set()
        assert report.targets_met is False
        assert set(report.class_counts) == {"stove", "gas_cylinder", "wet_floor"}
        assert all(v == 0 for v in report.class_counts.values())

    def test_aggregates_across_sessions_and_houses(self, tmp_path: Path) -> None:
        root = tmp_path / "captures"
        _session(
            root,
            "h01_kitchen_s001",
            "h01",
            "kitchen",
            image_count=10,
            class_counts={"stove": 6, "gas_cylinder": 2},
            annotation_status="finalized",
        )
        _session(
            root,
            "h01_hall_s001",
            "h01",
            "hall",
            image_count=5,
            class_counts={"wet_floor": 3},
            annotation_status="staged",
        )
        _session(
            root,
            "h02_kitchen_s001",
            "h02",
            "kitchen",
            image_count=8,
            class_counts={"stove": 4},
            lighting="dim",
        )

        report = build_progress_report(root, tmp_path / "eval", _TARGETS, {})

        assert report.total_images == 23
        assert report.houses == {"h01", "h02"}
        assert report.class_counts == {"stove": 10, "gas_cylinder": 2, "wet_floor": 3}
        assert report.rooms_by_house == {"h01": {"kitchen", "hall"}, "h02": {"kitchen"}}
        assert report.lighting_covered == {"daylight", "dim"}
        assert report.annotation_status_counts == {
            "finalized": 1,
            "staged": 1,
            "unannotated": 1,
        }

    def test_classes_met_and_pending(self, tmp_path: Path) -> None:
        root = tmp_path / "captures"
        _session(
            root,
            "h01_kitchen_s001",
            "h01",
            "kitchen",
            class_counts={"stove": 12, "gas_cylinder": 3},
        )
        report = build_progress_report(root, tmp_path / "eval", _TARGETS, {})
        assert report.classes_met == ["stove"]
        assert report.classes_pending == {"gas_cylinder": 7, "wet_floor": 10}

    def test_targets_met_requires_images_houses_and_classes(self, tmp_path: Path) -> None:
        root = tmp_path / "captures"
        _session(
            root,
            "h01_kitchen_s001",
            "h01",
            "kitchen",
            image_count=60,
            class_counts={"stove": 10, "gas_cylinder": 10, "wet_floor": 10},
        )
        report = build_progress_report(root, tmp_path / "eval", _TARGETS, {})
        assert report.classes_pending == {}
        assert report.targets_met is False  # only 1 house and 60 < 100 images

        _session(
            root,
            "h02_kitchen_s001",
            "h02",
            "kitchen",
            image_count=45,
        )
        report = build_progress_report(root, tmp_path / "eval", _TARGETS, {})
        assert report.total_images == 105
        assert len(report.houses) == 2
        assert report.targets_met is True

    def test_withdrawn_consent_blocks_targets_met(self, tmp_path: Path) -> None:
        root = tmp_path / "captures"
        _session(
            root,
            "h01_kitchen_s001",
            "h01",
            "kitchen",
            image_count=100,
            class_counts={"stove": 10, "gas_cylinder": 10, "wet_floor": 10},
            consent_reference="CONSENT-h01-2026-001",
        )
        _session(root, "h02_kitchen_s001", "h02", "kitchen", image_count=10)
        registry = {
            "CONSENT-h01-2026-001": ConsentRecord(
                consent_id="CONSENT-h01-2026-001", house_id="h01", withdrawn=True
            )
        }
        report = build_progress_report(root, tmp_path / "eval", _TARGETS, registry)
        assert report.withdrawn_sessions == {"h01_kitchen_s001": "CONSENT-h01-2026-001"}
        assert report.targets_met is False

    def test_eval_set_status(self, tmp_path: Path) -> None:
        eval_root = tmp_path / "eval"
        _session(eval_root, "h05_hall_s001", "h05", "hall", image_count=7)
        (eval_root / "images").mkdir(parents=True, exist_ok=True)
        for i in range(7):
            (eval_root / "images" / f"h05_hall_s001_{i:04d}.jpg").write_bytes(b"x")

        report_before = build_progress_report(tmp_path / "captures", eval_root, _TARGETS, {})
        assert report_before.eval_image_count == 7
        assert report_before.eval_locked is False

        lock_eval_set(eval_root)
        report_after = build_progress_report(tmp_path / "captures", eval_root, _TARGETS, {})
        assert report_after.eval_locked is True

    def test_absent_eval_root_reports_zero(self, tmp_path: Path) -> None:
        report = build_progress_report(tmp_path / "captures", tmp_path / "eval", _TARGETS, {})
        assert report.eval_image_count == 0
        assert report.eval_locked is False


@pytest.mark.unit
class TestWriteProgressReport:
    """Report file generation."""

    def test_writes_all_three_formats(self, tmp_path: Path) -> None:
        root = tmp_path / "captures"
        _session(root, "h01_kitchen_s001", "h01", "kitchen", class_counts={"stove": 5})
        report = build_progress_report(root, tmp_path / "eval", _TARGETS, {})

        paths = write_progress_report(report, tmp_path / "reports")

        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["markdown"].exists()

        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["total_images"] == 5
        assert data["class_counts"]["stove"] == 5
        assert data["targets_met"] is False

        md = paths["markdown"].read_text(encoding="utf-8")
        assert "Custom Capture Collection Progress" in md
        assert "stove" in md

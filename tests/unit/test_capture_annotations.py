"""Unit tests for src.dataset.capture.annotations — annotation import."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.dataset.capture.annotations import (
    finalize_annotations,
    read_yolo_export,
    stage_annotations,
    staged_annotators,
    update_annotation_status,
    validate_session_labels,
    verify_class_order,
)
from src.dataset.manifest import CaptureSessionManifest, SourceManifest

_CLASS_NAMES = {0: "person", 1: "stove", 2: "gas_cylinder"}
_NAMES_TEXT = "person\nstove\ngas_cylinder\n"


def _make_cvat_zip(
    path: Path,
    labels: dict[str, str],
    names_text: str = _NAMES_TEXT,
) -> Path:
    """Create a CVAT YOLO 1.1-style export zip."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("obj.names", names_text)
        zf.writestr("obj.data", "classes = 3\nnames = obj.names\n")
        zf.writestr("train.txt", "\n".join(f"obj_train_data/{stem}.jpg" for stem in labels) + "\n")
        for stem, text in labels.items():
            zf.writestr(f"obj_train_data/{stem}.txt", text)
    return path


def _make_export_dir(root: Path, labels: dict[str, str], names_text: str = _NAMES_TEXT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "obj.names").write_text(names_text, encoding="utf-8")
    (root / "obj.data").write_text("classes = 3\n", encoding="utf-8")
    data_dir = root / "obj_train_data"
    data_dir.mkdir(exist_ok=True)
    for stem, text in labels.items():
        (data_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    return root


def _capture_tree(tmp_path: Path, session_id: str = "h01_kitchen_s001") -> Path:
    """Minimal ingested-session tree (no PIL needed)."""
    root = tmp_path / "captures"
    for sub in ("images", "labels", "manifests"):
        (root / sub).mkdir(parents=True)
    hashes = {}
    for i in (1, 2):
        name = f"{session_id}_{i:04d}.jpg"
        (root / "images" / name).write_bytes(b"fake image bytes")
        hashes[name] = "0" * 64
    CaptureSessionManifest(
        source="custom_captures",
        session_id=session_id,
        house_id="h01",
        room="kitchen",
        image_count=2,
        image_hashes=hashes,
        trusted_classes=["stove"],
    ).save(root / "manifests" / f"{session_id}.json")
    return root


@pytest.mark.unit
class TestReadYoloExport:
    """Zip and directory export parsing."""

    def test_reads_cvat_zip(self, tmp_path: Path) -> None:
        archive = _make_cvat_zip(
            tmp_path / "export.zip",
            {"h01_kitchen_s001_0001": "1 0.5 0.5 0.2 0.2\n"},
        )
        export = read_yolo_export(archive)
        assert export.names == ["person", "stove", "gas_cylinder"]
        assert export.labels == {"h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2"]}

    def test_reads_extracted_dir(self, tmp_path: Path) -> None:
        root = _make_export_dir(
            tmp_path / "export",
            {"h01_kitchen_s001_0001": "0 0.1 0.1 0.05 0.05\n2 0.9 0.9 0.1 0.1\n"},
        )
        export = read_yolo_export(root)
        assert len(export.labels["h01_kitchen_s001_0001"]) == 2

    def test_missing_names_file_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "export"
        (root / "obj_train_data").mkdir(parents=True)
        (root / "obj_train_data" / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="names"):
            read_yolo_export(root)

    def test_no_labels_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "export"
        root.mkdir()
        (root / "obj.names").write_text(_NAMES_TEXT, encoding="utf-8")
        with pytest.raises(ValueError, match="label"):
            read_yolo_export(root)

    def test_missing_export_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_yolo_export(tmp_path / "missing.zip")


@pytest.mark.unit
class TestVerifyClassOrder:
    """The silent-killer check: export names must match taxonomy ID-for-ID."""

    def test_exact_match_passes(self) -> None:
        assert verify_class_order(["person", "stove", "gas_cylinder"], _CLASS_NAMES) == []

    def test_reordered_names_fail(self) -> None:
        problems = verify_class_order(["stove", "person", "gas_cylinder"], _CLASS_NAMES)
        assert len(problems) == 2
        assert "class ID 0" in problems[0]

    def test_subset_fails(self) -> None:
        problems = verify_class_order(["person", "stove"], _CLASS_NAMES)
        assert any("FULL ordered label list" in p for p in problems)

    def test_extra_class_fails(self) -> None:
        problems = verify_class_order(["person", "stove", "gas_cylinder", "extra"], _CLASS_NAMES)
        assert problems


@pytest.mark.unit
class TestValidateSessionLabels:
    """Session-scoped label validation."""

    _STEMS = {"h01_kitchen_s001_0001", "h01_kitchen_s001_0002"}

    def _export(self, labels: dict[str, list[str]]) -> object:
        from src.dataset.capture.annotations import YoloExport

        return YoloExport(names=list(_CLASS_NAMES.values()), labels=labels)

    def test_happy_path(self) -> None:
        export = self._export(
            {
                "h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2"],
                "h01_kitchen_s001_0002": ["2 0.3 0.3 0.1 0.1", "1 0.7 0.7 0.2 0.2"],
            }
        )
        result = validate_session_labels(
            export, self._STEMS, _CLASS_NAMES, 0.95, trusted_classes=("stove",)
        )
        assert result.problems == []
        assert result.warnings == []
        assert result.labeled_fraction == 1.0
        assert result.class_counts == {"gas_cylinder": 1, "stove": 2}

    def test_orphan_label_is_problem(self) -> None:
        export = self._export(
            {
                "h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2"],
                "h99_other_s001_0001": ["1 0.5 0.5 0.2 0.2"],
                "h01_kitchen_s001_0002": ["1 0.5 0.5 0.2 0.2"],
            }
        )
        result = validate_session_labels(export, self._STEMS, _CLASS_NAMES, 0.5)
        assert any("no ingested image" in p for p in result.problems)

    def test_under_coverage_is_problem(self) -> None:
        export = self._export({"h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2"]})
        result = validate_session_labels(export, self._STEMS, _CLASS_NAMES, 0.95)
        assert any("minimum is 95%" in p for p in result.problems)

    def test_bad_lines_are_problems(self) -> None:
        export = self._export(
            {
                "h01_kitchen_s001_0001": [
                    "not a yolo line",
                    "7 0.5 0.5 0.2 0.2",  # class id out of range
                    "1 1.5 0.5 0.2 0.2",  # cx out of bounds
                ],
                "h01_kitchen_s001_0002": ["1 0.5 0.5 0.2 0.2"],
            }
        )
        result = validate_session_labels(export, self._STEMS, _CLASS_NAMES, 0.5)
        assert any("malformed" in p for p in result.problems)
        assert sum("0001.txt" in p for p in result.problems) >= 3

    def test_duplicate_lines_are_problems(self) -> None:
        export = self._export(
            {
                "h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.2 0.2"],
                "h01_kitchen_s001_0002": ["1 0.5 0.5 0.2 0.2"],
            }
        )
        result = validate_session_labels(export, self._STEMS, _CLASS_NAMES, 0.5)
        assert any("duplicate lines" in p for p in result.problems)

    def test_missing_trusted_class_warns(self) -> None:
        export = self._export(
            {
                "h01_kitchen_s001_0001": ["0 0.5 0.5 0.2 0.2"],
                "h01_kitchen_s001_0002": ["0 0.4 0.4 0.2 0.2"],
            }
        )
        result = validate_session_labels(
            export, self._STEMS, _CLASS_NAMES, 0.95, trusted_classes=("gas_cylinder",)
        )
        assert result.problems == []
        assert any("gas_cylinder" in w for w in result.warnings)


@pytest.mark.unit
class TestStageAndFinalize:
    """Staging and finalize flows."""

    def test_stage_writes_per_annotator(self, tmp_path: Path) -> None:
        from src.dataset.capture.annotations import YoloExport

        export = YoloExport(
            names=list(_CLASS_NAMES.values()),
            labels={"h01_kitchen_s001_0001": ["1 0.5 0.5 0.2 0.2"]},
        )
        staging = tmp_path / "staging"
        dest = stage_annotations(export, "h01_kitchen_s001", "asha", staging)
        assert (dest / "h01_kitchen_s001_0001.txt").read_text(
            encoding="utf-8"
        ) == "1 0.5 0.5 0.2 0.2\n"
        assert staged_annotators(staging, "h01_kitchen_s001") == ["asha"]

    def test_update_annotation_status(self, tmp_path: Path) -> None:
        root = _capture_tree(tmp_path)
        manifest = update_annotation_status(root, "h01_kitchen_s001", "staged", annotator="asha")
        assert manifest.annotation_status == "staged"
        assert manifest.annotators == ["asha"]
        # Idempotent annotator append
        manifest = update_annotation_status(root, "h01_kitchen_s001", "staged", annotator="asha")
        assert manifest.annotators == ["asha"]

    def test_update_status_invalid_raises(self, tmp_path: Path) -> None:
        root = _capture_tree(tmp_path)
        with pytest.raises(ValueError, match="status"):
            update_annotation_status(root, "h01_kitchen_s001", "done")

    def test_update_status_missing_session_raises(self, tmp_path: Path) -> None:
        root = _capture_tree(tmp_path)
        with pytest.raises(FileNotFoundError, match="ingest"):
            update_annotation_status(root, "h09_hall_s001", "staged")

    def test_finalize_promotes_labels_and_updates_manifests(self, tmp_path: Path) -> None:
        root = _capture_tree(tmp_path)
        staging = tmp_path / "staging"
        annotator_dir = staging / "h01_kitchen_s001" / "asha"
        annotator_dir.mkdir(parents=True)
        (annotator_dir / "h01_kitchen_s001_0001.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n", encoding="utf-8"
        )
        (annotator_dir / "h01_kitchen_s001_0002.txt").write_text(
            "1 0.6 0.6 0.2 0.2\n", encoding="utf-8"
        )

        result = finalize_annotations(staging, "h01_kitchen_s001", "asha", root, _CLASS_NAMES)

        assert result.labels_written == 2
        assert result.class_counts == {"gas_cylinder": 1, "stove": 2}
        assert (root / "labels" / "h01_kitchen_s001_0001.txt").exists()

        manifest = CaptureSessionManifest.load(root / "manifests" / "h01_kitchen_s001.json")
        assert manifest.annotation_status == "finalized"
        assert manifest.annotators == ["asha"]
        assert manifest.class_counts == {"gas_cylinder": 1, "stove": 2}

        aggregate = SourceManifest.load(root / "manifest.json")
        assert aggregate.class_counts == {"gas_cylinder": 1, "stove": 2}

    def test_finalize_without_staged_raises(self, tmp_path: Path) -> None:
        root = _capture_tree(tmp_path)
        with pytest.raises(FileNotFoundError, match="staged"):
            finalize_annotations(
                tmp_path / "staging", "h01_kitchen_s001", "asha", root, _CLASS_NAMES
            )

"""
Unit tests for scripts.qa.check_annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.qa.check_annotations import (
    QAIssue,
    QAResults,
    build_qa_reports,
    check_annotation_format,
    check_file_pairs,
    check_split_leakage,
)

# ─── QAResults ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestQAResults:
    def test_critical_count(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("check_a", "CRITICAL", "train", "file.txt", 1, "bad"))
        r.add_issue(QAIssue("check_b", "WARNING", "val", "file2.txt", 2, "warn"))
        assert r.critical_count == 1
        assert r.warning_count == 1
        assert r.info_count == 0

    def test_finalize_check_pass(self) -> None:
        r = QAResults()
        r.finalize_check("my_check", 0)
        assert r.check_summaries["my_check"]["status"] == "PASS"
        assert r.check_summaries["my_check"]["count"] == 0

    def test_finalize_check_critical(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("my_check", "CRITICAL", "train", "", 0, "crit issue"))
        r.finalize_check("my_check", 1)
        assert r.check_summaries["my_check"]["status"] == "CRITICAL"

    def test_finalize_check_warning(self) -> None:
        r = QAResults()
        r.add_issue(QAIssue("my_check", "WARNING", "train", "", 0, "warn issue"))
        r.finalize_check("my_check", 1)
        assert r.check_summaries["my_check"]["status"] == "WARNING"


# ─── check_annotation_format ─────────────────────────────────────────────────


def _make_dataset(tmp_path: Path, split: str = "train") -> tuple[Path, Path]:
    """Create minimal dataset structure for QA tests."""
    img_dir = tmp_path / "images" / split
    lbl_dir = tmp_path / "labels" / split
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    return img_dir, lbl_dir


CLASS_NAMES = {
    i: name
    for i, name in enumerate(
        [
            "person",
            "face",
            "medicine_strip",
            "medicine_bottle",
            "water_bottle",
            "knife",
            "stove",
            "gas_cylinder",
            "passport",
            "book",
            "charger",
            "wire",
            "laptop",
            "monitor",
            "cupboard",
            "door",
            "chair",
            "bed",
            "toilet",
            "sink",
            "wet_floor",
            "walking_stick",
            "support_handle",
        ]
    )
}
NUM_CLASSES = 23


@pytest.mark.unit
class TestCheckAnnotationFormat:
    def test_valid_annotation_no_issues(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert results.critical_count == 0

    def test_detects_invalid_class_id(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("99 0.5 0.5 0.2 0.3\n")  # class 99 is invalid

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "invalid_class_ids" for i in results.issues)

    def test_detects_bbox_out_of_bounds(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("0 1.5 0.5 0.2 0.3\n")  # cx=1.5 out of range

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "bbox_out_of_bounds" for i in results.issues)

    def test_detects_zero_area_box(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.0 0.3\n")  # w=0

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "zero_area_boxes" for i in results.issues)

    def test_detects_invalid_format(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5\n")  # only 3 fields

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "invalid_yolo_format" for i in results.issues)

    def test_detects_duplicate_annotations(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        # Same annotation twice
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert any(i.check == "duplicate_annotations" for i in results.issues)

    def test_missing_label_dir_skipped(self, tmp_path: Path) -> None:
        """No crash when labels/ directory doesn't exist."""
        results = QAResults()
        check_annotation_format(tmp_path, CLASS_NAMES, NUM_CLASSES, results)
        assert results.critical_count == 0


# ─── check_file_pairs ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckFilePairs:
    def test_empty_label_file_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("")  # empty label

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "empty_label_files" for i in results.issues)

    def test_missing_label_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        # No label file created

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "missing_label_files" for i in results.issues)

    def test_missing_image_detected(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        # Only label, no image
        (lbl_dir / "orphan.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert any(i.check == "missing_image_files" for i in results.issues)

    def test_valid_pairs_no_issues(self, tmp_path: Path) -> None:
        img_dir, lbl_dir = _make_dataset(tmp_path)
        (img_dir / "img.jpg").write_bytes(b"x")
        (lbl_dir / "img.txt").write_text("5 0.5 0.5 0.2 0.3\n")

        results = QAResults()
        check_file_pairs(tmp_path, results)
        assert results.warning_count == 0


# ─── check_split_leakage ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestCheckSplitLeakage:
    def test_no_leakage_no_issues(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            d = tmp_path / "images" / split
            d.mkdir(parents=True)
            (d / f"{split}_only.jpg").write_bytes(f"unique_content_{split}".encode())

        results = QAResults()
        check_split_leakage(tmp_path, results)
        assert results.critical_count == 0

    def test_train_val_leakage_detected(self, tmp_path: Path) -> None:
        for split in ["train", "val", "test"]:
            (tmp_path / "images" / split).mkdir(parents=True)

        shared_content = b"identical_image_content"
        (tmp_path / "images" / "train" / "shared.jpg").write_bytes(shared_content)
        (tmp_path / "images" / "val" / "shared_copy.jpg").write_bytes(shared_content)
        (tmp_path / "images" / "test" / "unique.jpg").write_bytes(b"different")

        results = QAResults()
        check_split_leakage(tmp_path, results)
        assert any(i.check == "train_val_leakage" for i in results.issues)
        assert any(i.severity == "CRITICAL" for i in results.issues)


# ─── build_qa_reports ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildQaReports:
    def test_reports_built_without_error(self, tmp_path: Path) -> None:
        results = QAResults()
        results.total_images = 100
        results.total_labels = 100
        results.total_boxes = 500
        results.add_issue(QAIssue("test_check", "WARNING", "train", "file.txt", 1, "msg"))
        results.finalize_check("test_check", 1)

        json_report, csv_rows, md_sections = build_qa_reports(results, tmp_path, 23)

        assert "summary" in json_report
        assert json_report["summary"]["total_images"] == 100
        assert len(csv_rows) == 1
        assert len(md_sections) > 0

    def test_json_includes_all_issues(self) -> None:
        results = QAResults()
        for i in range(5):
            results.add_issue(QAIssue(f"check_{i}", "WARNING", "train", f"f{i}.txt", i, f"msg{i}"))

        json_report, _, _ = build_qa_reports(results, Path("."), 23)
        assert len(json_report["issues"]) == 5


# ─── Phase-3 eval-set guards (scripts.qa.run_full_qa) ─────────────────────────

PIL = pytest.importorskip("PIL", reason="Pillow required for eval-overlap tests")

from PIL import Image  # noqa: E402

from scripts.qa.run_full_qa import check_eval_overlap, check_house_exclusivity  # noqa: E402
from src.dataset.manifest import CaptureSessionManifest  # noqa: E402


def _block_image(path: Path, seed: int, size: tuple[int, int] = (64, 64)) -> None:
    """Deterministic 8x8 block-pattern image (distinct aHash per seed)."""
    import random

    from PIL import ImageDraw

    rng = random.Random(seed)  # noqa: S311
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    bw, bh = size[0] // 8, size[1] // 8
    for by in range(8):
        for bx in range(8):
            val = 255 if rng.random() > 0.5 else 0
            draw.rectangle(
                (bx * bw, by * bh, (bx + 1) * bw - 1, (by + 1) * bh - 1), fill=(val, val, val)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.mark.unit
class TestCheckEvalOverlap:
    """Eval-set leakage guard (exact + flip-robust perceptual)."""

    def test_absent_eval_dir_is_not_available(self, tmp_path: Path) -> None:
        report, critical = check_eval_overlap(
            tmp_path / "eval", tmp_path / "processed", tmp_path / "merged"
        )
        assert report == {"available": False}
        assert critical is False

    def test_no_overlap_passes(self, tmp_path: Path) -> None:
        _block_image(tmp_path / "merged" / "images" / "train1.jpg", seed=1)
        _block_image(tmp_path / "processed" / "images" / "train" / "train2.jpg", seed=2)
        _block_image(tmp_path / "eval" / "images" / "eval1.jpg", seed=99)

        report, critical = check_eval_overlap(
            tmp_path / "eval", tmp_path / "processed", tmp_path / "merged"
        )
        assert critical is False
        assert report["exact_overlap_count"] == 0
        assert report["near_overlap_count"] == 0
        assert report["eval_image_count"] == 1

    def test_exact_duplicate_is_critical(self, tmp_path: Path) -> None:
        _block_image(tmp_path / "merged" / "images" / "train1.jpg", seed=5)
        eval_path = tmp_path / "eval" / "images" / "eval1.jpg"
        eval_path.parent.mkdir(parents=True)
        eval_path.write_bytes((tmp_path / "merged" / "images" / "train1.jpg").read_bytes())

        report, critical = check_eval_overlap(
            tmp_path / "eval", tmp_path / "processed", tmp_path / "merged"
        )
        assert critical is True
        assert report["exact_overlap_count"] == 1

    def test_flipped_duplicate_is_near_overlap(self, tmp_path: Path) -> None:
        # Same seed but mirrored → different bytes, but flip-robust hash matches.
        train_path = tmp_path / "merged" / "images" / "train1.jpg"
        _block_image(train_path, seed=7)
        with Image.open(train_path) as img:
            flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            eval_path = tmp_path / "eval" / "images" / "eval1.jpg"
            eval_path.parent.mkdir(parents=True)
            flipped.save(eval_path)

        report, critical = check_eval_overlap(
            tmp_path / "eval", tmp_path / "processed", tmp_path / "merged"
        )
        assert critical is True
        assert report["near_overlap_count"] == 1
        assert report["exact_overlap_count"] == 0

    def test_eval_internal_duplicates_are_not_flagged(self, tmp_path: Path) -> None:
        # Two eval images identical to EACH OTHER but not to any train image
        # must not be reported as leakage.
        _block_image(tmp_path / "merged" / "images" / "train1.jpg", seed=1)
        _block_image(tmp_path / "eval" / "images" / "eval1.jpg", seed=50)
        eval2 = tmp_path / "eval" / "images" / "eval2.jpg"
        eval2.write_bytes((tmp_path / "eval" / "images" / "eval1.jpg").read_bytes())

        report, critical = check_eval_overlap(
            tmp_path / "eval", tmp_path / "processed", tmp_path / "merged"
        )
        assert critical is False
        assert report["exact_overlap_count"] == 0


@pytest.mark.unit
class TestCheckHouseExclusivity:
    """House-sharing WARNING between training captures and the eval set."""

    @staticmethod
    def _house_session(root: Path, session_id: str, house_id: str) -> None:
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        CaptureSessionManifest(
            source="custom_captures", session_id=session_id, house_id=house_id, room="kitchen"
        ).save(manifests_dir / f"{session_id}.json")

    def test_no_manifests_is_not_available(self, tmp_path: Path) -> None:
        report = check_house_exclusivity(tmp_path / "captures", tmp_path / "eval")
        assert report == {"available": False}

    def test_disjoint_houses_no_warning(self, tmp_path: Path) -> None:
        self._house_session(tmp_path / "captures", "h01_kitchen_s001", "h01")
        self._house_session(tmp_path / "eval", "h02_kitchen_s001", "h02")
        report = check_house_exclusivity(tmp_path / "captures", tmp_path / "eval")
        assert report["available"] is True
        assert report["shared_houses"] == []

    def test_shared_house_is_flagged(self, tmp_path: Path) -> None:
        self._house_session(tmp_path / "captures", "h01_kitchen_s001", "h01")
        self._house_session(tmp_path / "eval", "h01_hall_s001", "h01")
        report = check_house_exclusivity(tmp_path / "captures", tmp_path / "eval")
        assert report["shared_houses"] == ["h01"]
        assert report["train_houses"] == ["h01"]
        assert report["eval_houses"] == ["h01"]


# ─── M3 annotation artifact sweeps (scripts.qa.run_full_qa) ───────────────────

from scripts.qa.run_full_qa import sweep_annotation_artifacts  # noqa: E402
from src.dataset.annotation.base import ModelFingerprint  # noqa: E402
from src.dataset.annotation.batches import (  # noqa: E402
    BATCH_MANIFEST_FILENAME,
    VerificationBatchManifest,
)
from src.dataset.annotation.candidates import (  # noqa: E402
    CANDIDATES_FILENAME,
    build_candidates_artifact,
    save_candidates,
)
from src.dataset.annotation.ledger import new_ledger, record_verdict, save_ledger  # noqa: E402


def _fingerprint() -> ModelFingerprint:
    return ModelFingerprint(
        backend="fake",
        weights_path="",
        weights_sha256="",
        library_versions={},
        device="cpu",
        prompt_fingerprint="sha256:x",
    )


@pytest.mark.unit
class TestSweepAnnotationArtifacts:
    """M3: orphan candidates, duplicate ledger claims, unused batches, orphan deltas."""

    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "candidates_root": tmp_path / "candidates",
            "batches_root": tmp_path / "batches",
            "ledger_path": tmp_path / "ledger.json",
            "verified_labels_dir": tmp_path / "verified_labels",
            "merged_manifest_path": tmp_path / "merged_manifest.json",
        }

    def test_nothing_exists_is_not_available(self, tmp_path: Path) -> None:
        report = sweep_annotation_artifacts(**self._paths(tmp_path))
        assert report == {"available": False}

    def test_clean_state_no_findings(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        paths["ledger_path"].parent.mkdir(parents=True, exist_ok=True)
        save_ledger(new_ledger(), paths["ledger_path"])
        report = sweep_annotation_artifacts(**paths)
        assert report["available"] is True
        assert report["orphan_candidates_count"] == 0
        assert report["duplicate_ledger_claims_count"] == 0
        assert report["unused_batches_count"] == 0
        assert report["verified_labels_orphans_count"] == 0

    def test_orphan_candidate_detected(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        paths["merged_manifest_path"].write_text(
            '{"image_provenance": {"a.jpg": "coco"}}', encoding="utf-8"
        )
        artifact = build_candidates_artifact(
            backend="fake",
            model=_fingerprint(),
            taxonomy_fp="sha256:x",
            inputs={},
            determinism={},
            images={},
            runtime_s=0.0,
            class_names_by_id={0: "person"},
        )
        artifact["images"] = {"ghost.jpg": {"targeted_class_ids": [], "detections": []}}
        candidates_path = paths["candidates_root"] / "fake" / CANDIDATES_FILENAME
        save_candidates(artifact, candidates_path)
        report = sweep_annotation_artifacts(**paths)
        assert report["orphan_candidates_count"] == 1
        assert "fake/ghost.jpg" in report["orphan_candidates"]

    def _write_batch(
        self, batches_root: Path, batch_id: str, images: list[str], status: str
    ) -> None:
        VerificationBatchManifest(batch_id=batch_id, images=images, status=status).save(
            batches_root / batch_id / BATCH_MANIFEST_FILENAME
        )

    def test_duplicate_ledger_claim_detected(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        self._write_batch(paths["batches_root"], "vb001_x", ["a.jpg"], "imported")
        self._write_batch(paths["batches_root"], "vb002_x", ["a.jpg"], "imported")
        ledger = new_ledger()
        record_verdict(
            ledger, "a.jpg", "coco", "person", "verified_absent", [], "vb001_x", "v", "cvat", ""
        )
        paths["ledger_path"].parent.mkdir(parents=True, exist_ok=True)
        save_ledger(ledger, paths["ledger_path"])
        report = sweep_annotation_artifacts(**paths)
        assert report["duplicate_ledger_claims_count"] == 1

    def test_unused_batch_detected(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        self._write_batch(paths["batches_root"], "vb001_x", ["a.jpg"], "imported")
        paths["ledger_path"].parent.mkdir(parents=True, exist_ok=True)
        save_ledger(new_ledger(), paths["ledger_path"])  # empty — nothing was actually recorded
        report = sweep_annotation_artifacts(**paths)
        assert report["unused_batches_count"] == 1
        assert "vb001_x" in report["unused_batches"]

    def test_verified_labels_orphan_detected(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        paths["verified_labels_dir"].mkdir(parents=True, exist_ok=True)
        (paths["verified_labels_dir"] / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        paths["ledger_path"].parent.mkdir(parents=True, exist_ok=True)
        save_ledger(new_ledger(), paths["ledger_path"])
        report = sweep_annotation_artifacts(**paths)
        assert report["verified_labels_orphans_count"] == 1
        assert "orphan.txt" in report["verified_labels_orphans"]


# ─── M4 L4/L5 report sweep (scripts.qa.run_full_qa) ───────────────────────────

import json  # noqa: E402

from scripts.qa.run_full_qa import sweep_l4_l5_reports  # noqa: E402

_NAMES = {0: "person", 1: "charger"}
_NC = 2


def _write_data_yaml(path: Path) -> Path:
    path.write_text(
        json.dumps({"nc": _NC, "names": {str(k): v for k, v in _NAMES.items()}}),
        encoding="utf-8",
    )
    return path


def _live_fp() -> str:
    from src.dataset.completeness import taxonomy_fingerprint

    return taxonomy_fingerprint(_NC, _NAMES)


@pytest.mark.unit
class TestSweepL4L5Reports:
    """M4: coverage_report.json / dataset_quality_report.json schema + staleness."""

    def test_neither_report_exists_is_not_available(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        report = sweep_l4_l5_reports(
            tmp_path / "coverage_report.json", tmp_path / "dataset_quality_report.json", data_yaml
        )
        assert report == {"available": False}

    def test_valid_fresh_reports_no_problems(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        fp = _live_fp()
        coverage_path = tmp_path / "coverage_report.json"
        coverage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "taxonomy_fingerprint": fp,
                    "per_class": {
                        "person": {"coverage_score": 1.0, "residual_missing_estimate": 0.0}
                    },
                    "per_image": {},
                    "per_image_summary": {},
                    "dataset": {"residual_missing_total": 0.0, "unknown_objects_total": 0},
                }
            ),
            encoding="utf-8",
        )
        quality_path = tmp_path / "dataset_quality_report.json"
        quality_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "taxonomy_fingerprint": fp,
                    "dataset_scale": {"images_total": 1},
                    "completeness_summary": {"masked_cell_fraction": 0.5},
                    "coverage_summary": {},
                    "per_class_risk": {},
                    "verification_progress": {},
                }
            ),
            encoding="utf-8",
        )

        report = sweep_l4_l5_reports(coverage_path, quality_path, data_yaml)
        assert report["available"] is True
        assert report["coverage_report_present"] is True
        assert report["quality_report_present"] is True
        assert report["problems_count"] == 0

    def test_stale_taxonomy_fingerprint_flagged(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        coverage_path = tmp_path / "coverage_report.json"
        coverage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "taxonomy_fingerprint": "sha256:stale",
                    "per_class": {},
                    "per_image": {},
                    "per_image_summary": {},
                    "dataset": {"residual_missing_total": 0.0, "unknown_objects_total": 0},
                }
            ),
            encoding="utf-8",
        )

        report = sweep_l4_l5_reports(coverage_path, tmp_path / "missing.json", data_yaml)
        assert report["available"] is True
        assert report["quality_report_present"] is False
        assert report["problems_count"] == 1
        assert "stale" in report["problems"][0]

    def test_invalid_schema_flagged(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        quality_path = tmp_path / "dataset_quality_report.json"
        quality_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

        report = sweep_l4_l5_reports(tmp_path / "missing.json", quality_path, data_yaml)
        assert report["available"] is True
        assert report["problems_count"] >= 1
        assert any("missing required dimension" in p for p in report["problems"])

    def _write_fresh_pair(
        self, tmp_path: Path, cov_images: int, q_images: int
    ) -> tuple[Path, Path]:
        fp = _live_fp()
        coverage_path = tmp_path / "coverage_report.json"
        coverage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "taxonomy_fingerprint": fp,
                    "per_class": {},
                    "per_image": {f"img_{i}.jpg": {"completeness": 1.0} for i in range(cov_images)},
                    "per_image_summary": {},
                    "dataset": {"residual_missing_total": 0.0, "unknown_objects_total": 0},
                }
            ),
            encoding="utf-8",
        )
        quality_path = tmp_path / "dataset_quality_report.json"
        quality_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "taxonomy_fingerprint": fp,
                    "dataset_scale": {"images_total": q_images},
                    "completeness_summary": {"masked_cell_fraction": 0.5},
                    "coverage_summary": {},
                    "per_class_risk": {},
                    "verification_progress": {},
                }
            ),
            encoding="utf-8",
        )
        return coverage_path, quality_path

    def _write_completeness(self, tmp_path: Path, images_total: int) -> Path:
        path = tmp_path / "completeness.json"
        path.write_text(json.dumps({"stats": {"images_total": images_total}}), encoding="utf-8")
        return path

    def test_image_count_drift_flagged(self, tmp_path: Path) -> None:
        """A 188-image report over a 14k-image dataset must be flagged stale."""
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        coverage_path, quality_path = self._write_fresh_pair(tmp_path, cov_images=188, q_images=188)
        completeness_path = self._write_completeness(tmp_path, images_total=14005)

        report = sweep_l4_l5_reports(coverage_path, quality_path, data_yaml, completeness_path)
        assert report["available"] is True
        assert report["live_images_total"] == 14005
        assert report["problems_count"] == 2
        assert any(
            "coverage_report: image count 188 != live dataset 14005" in p
            for p in report["problems"]
        )
        assert any(
            "dataset_quality_report: images_total 188 != live dataset 14005" in p
            for p in report["problems"]
        )

    def test_image_count_match_no_drift(self, tmp_path: Path) -> None:
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        coverage_path, quality_path = self._write_fresh_pair(
            tmp_path, cov_images=14005, q_images=14005
        )
        completeness_path = self._write_completeness(tmp_path, images_total=14005)

        report = sweep_l4_l5_reports(coverage_path, quality_path, data_yaml, completeness_path)
        assert report["problems_count"] == 0

    def test_image_count_guard_skipped_without_completeness(self, tmp_path: Path) -> None:
        """No completeness artifact → drift guard is a no-op (backward compatible)."""
        data_yaml = _write_data_yaml(tmp_path / "data.yaml")
        coverage_path, quality_path = self._write_fresh_pair(tmp_path, cov_images=188, q_images=188)

        report = sweep_l4_l5_reports(coverage_path, quality_path, data_yaml)
        assert report["problems_count"] == 0
        assert report["live_images_total"] is None

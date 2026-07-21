"""
System test: Phase-5 M6 correctness-validation gate — full-loop smoke pipeline.

Chains every M1-M5 building block end-to-end on synthetic, offline data
(FakeAnnotator — no GPU/network, no real repo files touched): targeting ->
candidates -> verification batch -> simulated CVAT export -> import ->
ledger -> verified-labels overlay -> completeness (ledger-expanded policy)
-> coverage (L4) -> quality (L5) -> release-gate evaluation. Asserts
cross-artifact consistency (taxonomy fingerprint, ledger cell counts,
coverage/quality agreement) — the plan's M6 acceptance gate that must PASS
before any M7+ work starts.

Fast and fully offline (no GPU, no real dataset files) — unlike
test_training_smoke.py this runs by default, not env-gated.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from unit.annotation_fakes import FakeAnnotator  # noqa: F401 — registers "fake"

from src.dataset.annotation.apply import build_verified_labels_overlay
from src.dataset.annotation.base import BackendConfig
from src.dataset.annotation.batches import build_batch_manifests
from src.dataset.annotation.candidates import (
    CANDIDATES_FILENAME,
    ImageCandidates,
    build_candidates_artifact,
    save_candidates,
    validate_candidates,
)
from src.dataset.annotation.coverage import build_coverage_report, validate_coverage_report
from src.dataset.annotation.cvat_package import build_preannotation_zip
from src.dataset.annotation.ledger import (
    new_ledger,
    recompute_stats,
    save_ledger,
    validate_ledger,
)
from src.dataset.annotation.quality import build_quality_report, validate_quality_report
from src.dataset.annotation.registry import get_annotator
from src.dataset.annotation.targeting import build_targets, promptable_class_ids
from src.dataset.annotation.verified_import import import_verified_batch
from src.dataset.capture.annotations import read_yolo_export
from src.dataset.completeness import (
    build_completeness,
    taxonomy_fingerprint,
    validate_completeness,
)
from src.dataset.manifest import MergedManifest
from src.dataset.release.gates import evaluate_release

pytestmark = pytest.mark.system

_NAMES = {0: "person", 1: "charger", 2: "wire"}
_IDS_BY_NAME = {v: k for k, v in _NAMES.items()}
_NC = 3
_TAXONOMY_FP = taxonomy_fingerprint(_NC, _NAMES)
_IMAGES = ["img1.jpg", "img2.jpg"]


def _backend_config() -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "fake",
        {
            "enabled": True,
            "weights": "",
            "weights_sha256": "",
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": {"charger": ["phone charger"], "wire": ["cable"]},
            "thresholds": {"default": 0.25},
        },
    )


def _write_data_yaml(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump({"nc": _NC, "names": {int(k): v for k, v in _NAMES.items()}}),
        encoding="utf-8",
    )
    return path


def _generate_candidates() -> dict:
    """Run FakeAnnotator over untrusted cells and assemble the artifact."""
    manifest_obj = MergedManifest(
        image_provenance={name: "coco" for name in _IMAGES},
        label_completeness={"coco": ["person"]},
    )
    policies = {"coco": "trusted_list"}
    backend_cfg = _backend_config()
    promptable = promptable_class_ids(backend_cfg, _IDS_BY_NAME)
    targets = build_targets(manifest_obj, policies, promptable, _IDS_BY_NAME, verified_cells={})

    annotator = get_annotator("fake")
    annotator.load(backend_cfg, "cpu", _IDS_BY_NAME)
    images_out = {
        name: ImageCandidates(
            targeted_class_ids=targets[name],
            detections=tuple(annotator.annotate(Path(name), targets[name])),
        )
        for name in sorted(targets)
    }
    return build_candidates_artifact(
        backend="fake",
        model=annotator.fingerprint(),
        taxonomy_fp=_TAXONOMY_FP,
        inputs={"images_root": "images", "merged_manifest_sha256": "x", "ledger_sha256": "absent"},
        determinism={"seed": 0, "deterministic_algorithms": True, "image_order": "sorted"},
        images=images_out,
        runtime_s=0.01,
        class_names_by_id=_NAMES,
        git_commit="testcommit",
    )


class TestPhase5SmokePipeline:
    def test_full_loop_cross_artifact_consistency(self, tmp_path: Path) -> None:
        # ── merged (base) tree ──────────────────────────────────────────
        merged_labels_dir = tmp_path / "merged" / "labels"
        merged_labels_dir.mkdir(parents=True)
        (merged_labels_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
        (merged_labels_dir / "img2.txt").write_text("", encoding="utf-8")

        data_yaml = _write_data_yaml(tmp_path / "data.yaml")

        merged_manifest_path = tmp_path / "merged_manifest.json"
        MergedManifest(
            image_provenance={name: "coco" for name in _IMAGES},
            label_completeness={"coco": ["person"]},
            sources=[{"source": "coco", "total": 2, "accepted": 2}],
        ).save(merged_manifest_path)

        # ── L2: candidates (Fake backend) ───────────────────────────────
        candidates = _generate_candidates()
        assert validate_candidates(candidates, nc=_NC, expected_taxonomy_fp=_TAXONOMY_FP) == []
        candidates_root = tmp_path / "candidates"
        save_candidates(candidates, candidates_root / "fake" / CANDIDATES_FILENAME)

        # ── L1/L2: verification batch + simulated CVAT round-trip ──────
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="fake",
            candidates_sha256="deadbeef",
            batches_root=tmp_path / "batches",
            class_names_by_id=_NAMES,
            priority_classes=frozenset({"charger"}),
            batch_size=200,
        )
        batch = manifests[0]
        zip_path = tmp_path / "batches" / batch.batch_id / "preannotations.zip"
        build_preannotation_zip(
            batch_images=batch.images,
            candidate_images=candidates["images"],
            merged_labels_dir=merged_labels_dir,
            class_names_by_id=_NAMES,
            out_zip=zip_path,
        )

        export_labels: dict[str, list[str]] = {}
        for name in batch.images:
            stem = Path(name).stem
            dets = candidates["images"][name]["detections"]
            base_path = merged_labels_dir / f"{stem}.txt"
            base_lines = (
                [
                    line
                    for line in base_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if base_path.exists()
                else []
            )
            # Reviewer accepts the candidates as-is — export = untouched
            # trusted base lines UNION the target-class candidate boxes
            # (mirrors what the pre-annotation zip itself contained).
            export_labels[stem] = base_lines + [
                f"{d['class_id']} {d['bbox_xywhn'][0]} {d['bbox_xywhn'][1]} "
                f"{d['bbox_xywhn'][2]} {d['bbox_xywhn'][3]}"
                for d in dets
            ]
        export_zip = tmp_path / "export.zip"
        with zipfile.ZipFile(export_zip, "w") as zf:
            zf.writestr("obj.names", "\n".join(_NAMES[i] for i in range(_NC)) + "\n")
            for stem, lines in export_labels.items():
                zf.writestr(f"obj_train_data/{stem}.txt", "\n".join(lines) + "\n")
        export = read_yolo_export(export_zip)

        ledger = new_ledger()
        result = import_verified_batch(
            batch=batch,
            export=export,
            class_names_by_id=_NAMES,
            ids_by_name=_IDS_BY_NAME,
            merged_labels_dir=merged_labels_dir,
            verified_labels_dir=tmp_path / "verified_labels",
            ledger=ledger,
            source_by_image={name: "coco" for name in _IMAGES},
            verifier="anno_1",
        )
        assert result.images_imported == len(batch.images)
        recompute_stats(ledger, _TAXONOMY_FP)
        assert (
            validate_ledger(ledger, class_names=_IDS_BY_NAME, expected_taxonomy_fp=_TAXONOMY_FP)
            == []
        )
        ledger_path = tmp_path / "ledger.json"
        save_ledger(ledger, ledger_path)

        # ── ADR-P5-05: verified-labels overlay ──────────────────────────
        overlay_dir = tmp_path / "merged_verified" / "labels"
        build_verified_labels_overlay(merged_labels_dir, tmp_path / "verified_labels", overlay_dir)

        # ── minimal "processed" split (the split algorithm has its own
        # dedicated tests; this system test only proves downstream
        # ledger-expansion wiring) ───────────────────────────────────────
        processed_images_root = tmp_path / "processed" / "images"
        processed_labels_root = tmp_path / "processed" / "labels"
        (processed_images_root / "train").mkdir(parents=True, exist_ok=True)
        (processed_labels_root / "train").mkdir(parents=True, exist_ok=True)
        for name in _IMAGES:
            (processed_images_root / "train" / name).write_bytes(b"x")
            stem = Path(name).stem
            (processed_labels_root / "train" / f"{stem}.txt").write_text(
                (overlay_dir / f"{stem}.txt").read_text(encoding="utf-8"), encoding="utf-8"
            )
        split_summary_path = tmp_path / "split_summary.json"
        split_summary_path.write_text(
            json.dumps({"seed": 42, "strategy": "group_aware"}), encoding="utf-8"
        )

        sources_yaml = tmp_path / "sources.yaml"
        sources_yaml.write_text(
            yaml.safe_dump(
                {
                    "mode": "smoke",
                    "sources": {"coco": {"trusted_classes": ["person"]}},
                    "completeness": {
                        "policies": {"coco": "trusted_list_with_ledger"},
                        "ledger_path": str(ledger_path),
                    },
                }
            ),
            encoding="utf-8",
        )

        completeness = build_completeness(
            merged_manifest_path=merged_manifest_path,
            processed_images_root=processed_images_root,
            split_summary_path=split_summary_path,
            data_yaml_path=data_yaml,
            sources_yaml_path=sources_yaml,
            capture_manifests_dir=None,
        )
        assert validate_completeness(completeness, data_yaml_path=data_yaml) == []
        completeness_path = tmp_path / "completeness.json"
        completeness_path.write_text(json.dumps(completeness), encoding="utf-8")

        # Cross-artifact check #1: taxonomy fingerprint threads through.
        assert completeness["taxonomy"]["fingerprint"] == _TAXONOMY_FP

        # ── L4: coverage report ─────────────────────────────────────────
        coverage = build_coverage_report(
            candidates_root=candidates_root,
            ledger_path=ledger_path,
            completeness_path=completeness_path,
            processed_labels_root=processed_labels_root,
            data_yaml_path=data_yaml,
            iou_match_threshold=0.5,
            estimation_conf={"default": 0.35},
        )
        assert validate_coverage_report(coverage) == []
        assert coverage["taxonomy_fingerprint"] == _TAXONOMY_FP
        # Both target classes were settled by the ledger -> nothing "unknown".
        assert coverage["dataset"]["unknown_objects_total"] == 0
        assert coverage["per_class"]["charger"]["verified_present"] == len(batch.images)
        coverage_report_path = tmp_path / "coverage_report.json"
        coverage_report_path.write_text(json.dumps(coverage), encoding="utf-8")

        # ── L5: quality report ──────────────────────────────────────────
        quality = build_quality_report(
            completeness_path=completeness_path,
            coverage_report_path=coverage_report_path,
            merged_manifest_path=merged_manifest_path,
            ledger_path=ledger_path,
            batches_root=tmp_path / "batches",
            data_yaml_path=data_yaml,
        )
        assert validate_quality_report(quality) == []
        assert quality["taxonomy_fingerprint"] == _TAXONOMY_FP
        # Cross-artifact check #2: ledger cell count agrees between the
        # ledger itself and the quality report's aggregation of it.
        assert (
            quality["verification_progress"]["ledger_stats"]["cells_verified"]
            == ledger["stats"]["cells_verified"]
            == len(batch.images) * 2  # charger + wire per image
        )
        quality_report_path = tmp_path / "quality_report.json"
        quality_report_path.write_text(json.dumps(quality), encoding="utf-8")

        # ── release-check (RG1/RG2/RG3/RG4 over the artifacts just built) ─
        release_yaml = tmp_path / "release.yaml"
        release_yaml.write_text(
            yaml.safe_dump(
                {
                    "releases": {
                        "dataset-system-test": {
                            "mode": "smoke",
                            "gates": ["RG1", "RG2", "RG3", "RG4"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        qa_report_path = tmp_path / "qa_report.json"
        qa_report_path.write_text(
            json.dumps(
                {
                    "summary": {"critical_issues": 0},
                    "orchestrator": {
                        "license_critical": False,
                        "eval_overlap_critical": False,
                        "annotation_sweep_warnings": 0,
                        "l4_l5_report_warnings": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        changelog_path = tmp_path / "changelog.md"
        changelog_path.write_text("## dataset-system-test\n", encoding="utf-8")

        report = evaluate_release(
            "dataset-system-test",
            release_yaml_path=release_yaml,
            sources_yaml_path=sources_yaml,
            data_yaml_path=data_yaml,
            completeness_path=completeness_path,
            qa_report_path=qa_report_path,
            coverage_report_path=coverage_report_path,
            quality_report_path=quality_report_path,
            changelog_path=changelog_path,
            raw_root=tmp_path / "raw_empty",
        )
        assert report.verdict == "PASS", [r.format_line() for r in report.results]

    def test_candidates_generation_is_deterministic_across_runs(self) -> None:
        """Same (backend, config, data) triple -> byte-identical detections
        (ADR-P5-02) — the M6 "candidates double-run diff" determinism drill."""
        first = _generate_candidates()
        second = _generate_candidates()
        assert first["images"] == second["images"]
        assert first["stats"] == second["stats"]
        assert first["run_id"] == second["run_id"]

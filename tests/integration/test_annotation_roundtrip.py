"""
Integration test: Phase-5 M2 auto-annotation -> CVAT verification round-trip.

End-to-end on synthetic data, offline (FakeAnnotator — no GPU/network):
targeting (untrusted-cell computation) -> candidate generation -> batch
planning -> CVAT pre-annotation packaging -> simulated CVAT export ->
verified import -> ledger entries + delta label files.

This is the M2 acceptance drill (plan §M2 acceptance): ledger entry + deltas
recorded on a clean import; class-order tamper hard-fails; a non-target
label edit hard-fails; re-importing the same export twice is idempotent.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from unit.annotation_fakes import FakeAnnotator  # noqa: F401 — registers "fake"

from src.dataset.annotation.base import AnnotationError, BackendConfig
from src.dataset.annotation.batches import build_batch_manifests
from src.dataset.annotation.candidates import (
    ImageCandidates,
    build_candidates_artifact,
    validate_candidates,
)
from src.dataset.annotation.cvat_package import build_cvat_labels_spec, build_preannotation_zip
from src.dataset.annotation.ledger import LedgerView, new_ledger, recompute_stats, validate_ledger
from src.dataset.annotation.registry import get_annotator
from src.dataset.annotation.targeting import build_targets, promptable_class_ids
from src.dataset.annotation.verified_import import import_verified_batch
from src.dataset.capture.annotations import read_yolo_export
from src.dataset.manifest import MergedManifest

pytestmark = pytest.mark.integration

_CLASS_NAMES_BY_ID = {0: "person", 1: "charger", 2: "wire"}
_IDS_BY_NAME = {v: k for k, v in _CLASS_NAMES_BY_ID.items()}
_TAXONOMY_FP = "sha256:test-fixture"


def _backend_config(prompts: dict[str, list[str]]) -> BackendConfig:
    return BackendConfig.from_annotation_config(
        "fake",
        {
            "enabled": True,
            "weights": "",
            "weights_sha256": "",
            "imgsz": 640,
            "conf_floor": 0.05,
            "max_det": 100,
            "prompts": prompts,
            "thresholds": {"default": 0.25},
        },
    )


def _build_candidates(images: list[str]) -> dict:
    """Run FakeAnnotator over untrusted cells and assemble the artifact."""
    manifest = MergedManifest(
        image_provenance={name: "coco" for name in images},
        label_completeness={"coco": ["person"]},  # only 'person' trusted; charger/wire are not
    )
    policies = {"coco": "trusted_list"}
    backend_cfg = _backend_config({"charger": ["phone charger"], "wire": ["cable"]})
    promptable = promptable_class_ids(backend_cfg, _IDS_BY_NAME)
    targets = build_targets(manifest, policies, promptable, _IDS_BY_NAME, verified_cells={})

    annotator = get_annotator("fake")
    annotator.load(backend_cfg, "cpu", _IDS_BY_NAME)

    images_out = {}
    for name in sorted(targets):
        detections = annotator.annotate(Path(name), targets[name])
        images_out[name] = ImageCandidates(
            targeted_class_ids=targets[name], detections=tuple(detections)
        )

    artifact = build_candidates_artifact(
        backend="fake",
        model=annotator.fingerprint(),
        taxonomy_fp=_TAXONOMY_FP,
        inputs={"images_root": "images", "merged_manifest_sha256": "x", "ledger_sha256": "absent"},
        determinism={"seed": 0, "deterministic_algorithms": True, "image_order": "sorted"},
        images=images_out,
        runtime_s=0.01,
        class_names_by_id=_CLASS_NAMES_BY_ID,
        git_commit="testcommit",
    )
    problems = validate_candidates(artifact, nc=3, expected_taxonomy_fp=_TAXONOMY_FP)
    assert problems == []
    return artifact


def _make_export(tmp_path: Path, name: str, labels: dict[str, list[str]]) -> Path:
    """Build a "YOLO 1.1" export zip — obj.names in taxonomy order."""
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("obj.names", "\n".join(_CLASS_NAMES_BY_ID[i] for i in range(3)) + "\n")
        for stem, lines in labels.items():
            zf.writestr(f"obj_train_data/{stem}.txt", "\n".join(lines) + "\n")
    return path


class TestAnnotationRoundtrip:
    def test_full_roundtrip_records_ledger_and_deltas(self, tmp_path: Path) -> None:
        images = ["img1.jpg", "img2.jpg"]
        candidates = _build_candidates(images)
        # FakeAnnotator emits one centered box per targeted class — both
        # charger and wire for both images (neither is trusted for 'coco').
        assert candidates["images"]["img1.jpg"]["detections"]

        merged_labels_dir = tmp_path / "merged_labels"
        merged_labels_dir.mkdir()  # no base labels — nothing pre-trusted on disk

        manifests = build_batch_manifests(
            candidates=candidates,
            backend="fake",
            candidates_sha256="deadbeef",
            batches_root=tmp_path / "batches",
            class_names_by_id=_CLASS_NAMES_BY_ID,
            priority_classes=frozenset({"charger"}),
            batch_size=200,
        )
        assert len(manifests) == 1
        batch = manifests[0]
        assert set(batch.target_classes) == {"charger", "wire"}

        spec = build_cvat_labels_spec(_CLASS_NAMES_BY_ID)
        assert [s["name"] for s in spec] == ["person", "charger", "wire"]

        zip_path = tmp_path / "batches" / batch.batch_id / "preannotations.zip"
        sha = build_preannotation_zip(
            batch_images=batch.images,
            candidate_images=candidates["images"],
            merged_labels_dir=merged_labels_dir,
            class_names_by_id=_CLASS_NAMES_BY_ID,
            out_zip=zip_path,
        )
        assert sha

        # Simulate: reviewer opens the pre-annotation zip in CVAT, accepts the
        # candidates as-is (a real reviewer would tighten boxes; acceptance
        # here just proves the pipe carries data end-to-end), exports YOLO 1.1.
        export_labels = {}
        for name in batch.images:
            stem = Path(name).stem
            dets = candidates["images"][name]["detections"]
            export_labels[stem] = [
                f"{d['class_id']} {d['bbox_xywhn'][0]} {d['bbox_xywhn'][1]} "
                f"{d['bbox_xywhn'][2]} {d['bbox_xywhn'][3]}"
                for d in dets
            ]
        export_zip = _make_export(tmp_path, "export.zip", export_labels)
        export = read_yolo_export(export_zip)

        ledger = new_ledger()
        result = import_verified_batch(
            batch=batch,
            export=export,
            class_names_by_id=_CLASS_NAMES_BY_ID,
            ids_by_name=_IDS_BY_NAME,
            merged_labels_dir=merged_labels_dir,
            verified_labels_dir=tmp_path / "verified_labels",
            ledger=ledger,
            source_by_image={name: "coco" for name in images},
            verifier="anno_1",
        )
        assert result.images_imported == len(batch.images)
        assert result.verdicts_recorded == len(batch.images) * 2  # charger + wire per image

        recompute_stats(ledger, _TAXONOMY_FP)
        problems = validate_ledger(
            ledger, class_names=_IDS_BY_NAME, expected_taxonomy_fp=_TAXONOMY_FP
        )
        assert problems == []

        view = LedgerView(raw=ledger)
        for name in batch.images:
            assert view.verified_class_names(name) == {"charger", "wire"}
            delta_path = tmp_path / "verified_labels" / f"{Path(name).stem}.txt"
            assert delta_path.exists()
            assert len(delta_path.read_text(encoding="utf-8").splitlines()) == 2

    def test_class_order_tamper_hard_fails(self, tmp_path: Path) -> None:
        candidates = _build_candidates(["img1.jpg"])
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="fake",
            candidates_sha256="x",
            batches_root=tmp_path / "batches",
            class_names_by_id=_CLASS_NAMES_BY_ID,
            priority_classes=frozenset(),
            batch_size=200,
        )
        batch = manifests[0]

        tampered = tmp_path / "tampered.zip"
        with zipfile.ZipFile(tampered, "w") as zf:
            zf.writestr("obj.names", "wire\ncharger\nperson\n")  # WRONG order
            zf.writestr("obj_train_data/img1.txt", "1 0.5 0.5 0.2 0.2\n")
        export = read_yolo_export(tampered)

        with pytest.raises(AnnotationError, match="does not match the taxonomy"):
            import_verified_batch(
                batch=batch,
                export=export,
                class_names_by_id=_CLASS_NAMES_BY_ID,
                ids_by_name=_IDS_BY_NAME,
                merged_labels_dir=tmp_path / "merged_labels",
                verified_labels_dir=tmp_path / "verified_labels",
                ledger=new_ledger(),
                source_by_image={"img1.jpg": "coco"},
                verifier="anno_1",
            )

    def test_non_target_edit_hard_fails(self, tmp_path: Path) -> None:
        merged_labels_dir = tmp_path / "merged_labels"
        merged_labels_dir.mkdir()
        (merged_labels_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

        candidates = _build_candidates(["img1.jpg"])
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="fake",
            candidates_sha256="x",
            batches_root=tmp_path / "batches",
            class_names_by_id=_CLASS_NAMES_BY_ID,
            priority_classes=frozenset(),
            batch_size=200,
        )
        batch = manifests[0]

        # Reviewer accidentally nudged the trusted 'person' (class 0) box.
        export = read_yolo_export(
            _make_export(
                tmp_path,
                "export.zip",
                {"img1": ["0 0.51 0.5 0.3 0.3", "1 0.5 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2"]},
            )
        )

        with pytest.raises(AnnotationError, match="edited a trusted box"):
            import_verified_batch(
                batch=batch,
                export=export,
                class_names_by_id=_CLASS_NAMES_BY_ID,
                ids_by_name=_IDS_BY_NAME,
                merged_labels_dir=merged_labels_dir,
                verified_labels_dir=tmp_path / "verified_labels",
                ledger=new_ledger(),
                source_by_image={"img1.jpg": "coco"},
                verifier="anno_1",
            )

    def test_reimport_is_idempotent(self, tmp_path: Path) -> None:
        candidates = _build_candidates(["img1.jpg"])
        manifests = build_batch_manifests(
            candidates=candidates,
            backend="fake",
            candidates_sha256="x",
            batches_root=tmp_path / "batches",
            class_names_by_id=_CLASS_NAMES_BY_ID,
            priority_classes=frozenset(),
            batch_size=200,
        )
        batch = manifests[0]
        export_labels = {"img1": ["1 0.5 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2"]}
        export = read_yolo_export(_make_export(tmp_path, "export.zip", export_labels))

        ledger = new_ledger()
        kwargs = dict(
            batch=batch,
            export=export,
            class_names_by_id=_CLASS_NAMES_BY_ID,
            ids_by_name=_IDS_BY_NAME,
            merged_labels_dir=tmp_path / "merged_labels",
            verified_labels_dir=tmp_path / "verified_labels",
            source_by_image={"img1.jpg": "coco"},
            verifier="anno_1",
        )
        import_verified_batch(ledger=ledger, **kwargs)
        first = {k: v for k, v in ledger["entries"].items()}
        import_verified_batch(ledger=ledger, **kwargs)
        assert ledger["entries"] == first

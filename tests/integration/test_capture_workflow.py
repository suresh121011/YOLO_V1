"""
Integration test: Phase-3 custom capture workflow end-to-end.

Simulates the full human workflow on synthetic data, offline: inbox →
ingest (EXIF-stripped, consent-checked, manifested) → simulated
dual-annotator CVAT export → stage/compare (IAA) → finalize → merge
(custom capture priority, alongside a synthetic public source) →
group-aware + leave-one-house-out splits (session/house integrity) →
structural QA (the real CLI) → eval-set ingest with a deliberate
train-duplicate → overlap detected → fixed → locked → re-ingest refused.

Mirrors tests/integration/test_dataset_pipeline.py's shape but for the
Phase-3 capture/annotation layer built on top of the Phase-2 chain.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from src.dataset.capture.agreement import agreement_verdict, compare_annotators
from src.dataset.capture.annotations import (
    finalize_annotations,
    read_yolo_export,
    stage_annotations,
    validate_session_labels,
    verify_class_order,
)
from src.dataset.capture.config import (
    AnnotationSettings,
    CaptureConfig,
    CollectionTargets,
    ConsentSettings,
    IaaSettings,
    ImageRequirements,
)
from src.dataset.capture.exif import inspect_metadata
from src.dataset.capture.ingest import (
    SessionMeta,
    ingest_session,
    is_eval_locked,
    load_session_manifests,
    lock_eval_set,
)
from src.dataset.capture.progress import build_progress_report
from src.dataset.merge import MergeSource, merge_sources
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings
from src.dataset.splitting import SplitContext, get_strategy
from src.utils.annotation_utils import parse_yolo_line
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config
from src.utils.dataset_utils import find_image_files, group_files_by_key

PIL = pytest.importorskip("PIL", reason="Pillow required for image fixtures")
from PIL import Image  # noqa: E402

from scripts.dataset.split_dataset import copy_split_files, verify_no_leakage  # noqa: E402
from scripts.qa.run_full_qa import check_eval_overlap, check_house_exclusivity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _block_image(path: Path, seed: int, size: tuple[int, int] = (128, 128)) -> None:
    """Deterministic 8x8 block-pattern image (distinct aHash per seed)."""
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


def _public_source(root: Path, n_groups: int, seed_base: int) -> Path:
    """A tiny already-taxonomy-labeled public source (person=0), like COCO post-remap."""
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for g in range(n_groups):
        stem = f"coco_img{g:03d}"
        _block_image(root / "images" / f"{stem}.jpg", seed=seed_base + g)
        (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root


def _make_cvat_export(
    path: Path,
    names_in_order: list[str],
    labels: dict[str, str],
) -> Path:
    """A CVAT 'YOLO 1.1' style zip export."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("obj.names", "\n".join(names_in_order) + "\n")
        zf.writestr("obj.data", f"classes = {len(names_in_order)}\n")
        for stem, text in labels.items():
            zf.writestr(f"obj_train_data/{stem}.txt", text)
    return path


@pytest.mark.integration
def test_capture_workflow_end_to_end(tmp_path: Path) -> None:
    class_names = get_class_names_from_data_yaml(
        load_data_config(REPO_ROOT / "configs" / "data.yaml")
    )
    names_in_order = [class_names[i] for i in range(len(class_names))]
    stove_id = names_in_order.index("stove")
    gas_id = names_in_order.index("gas_cylinder")

    config = CaptureConfig(
        inbox_dir=tmp_path / "inbox",
        captures_root=tmp_path / "raw" / "custom_captures",
        eval_root=tmp_path / "eval" / "indian_home_v0",
        image=ImageRequirements(min_dim=100),
        consent=ConsentSettings(
            registry_path=tmp_path / "consent" / "consent_registry.yaml", required=True
        ),
        annotation=AnnotationSettings(
            staging_dir=tmp_path / "inbox" / "annotations",
            min_labeled_fraction=1.0,
            iaa=IaaSettings(iou_threshold=0.4, min_agreement=0.5),
        ),
    )

    # ── 1. Consent registry (local-only, PII-free) ──────────────────────────
    config.consent.registry_path.parent.mkdir(parents=True)
    config.consent.registry_path.write_text(
        yaml.dump(
            {
                "CONSENT-h01-2026-001": {
                    "house_id": "h01",
                    "granted_on": "2026-07-20",
                    "scope": "dataset-training",
                    "withdrawn": False,
                }
            }
        ),
        encoding="utf-8",
    )

    # ── 2. Inbox: 3 good photos (one carries GPS EXIF), 1 duplicate, 1 undersized
    config.inbox_dir.mkdir(parents=True)
    _block_image(config.inbox_dir / "photo_a.jpg", seed=1)
    _block_image(config.inbox_dir / "photo_b.jpg", seed=2)

    gps_photo = config.inbox_dir / "photo_c_gps.jpg"
    img = Image.new("RGB", (128, 128), (90, 60, 30))
    exif = Image.Exif()
    exif[0x0110] = "PhoneCam"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    img.save(gps_photo, exif=exif)

    (config.inbox_dir / "photo_a_dup.jpg").write_bytes(
        (config.inbox_dir / "photo_a.jpg").read_bytes()
    )
    _block_image(config.inbox_dir / "photo_tiny.jpg", seed=9, size=(50, 50))

    # ── 3. Ingest the session ────────────────────────────────────────────────
    meta = SessionMeta(
        session_id="h01_kitchen_s001",
        house_id="h01",
        room="kitchen",
        lighting="daylight",
        capture_device="TestCam",
        captured_at="2026-07-20",
        consent_reference="CONSENT-h01-2026-001",
        trusted_classes=("stove", "gas_cylinder"),
    )
    result = ingest_session(config.inbox_dir, meta, config, config.captures_root)

    assert result.accepted == 3  # photo_a, photo_b, photo_c_gps
    reasons = dict(result.rejected)
    assert "duplicate_of" in reasons["photo_a_dup.jpg"]
    assert "too_small" in reasons["photo_tiny.jpg"]

    ingested = sorted((config.captures_root / "images").iterdir())
    assert len(ingested) == 3
    for image in ingested:
        assert inspect_metadata(image)["clean"] is True  # GPS EXIF was stripped

    stems = [p.stem for p in ingested]

    # ── 4. Simulated dual-annotator CVAT export → stage → compare → finalize ──
    labels_a = {
        stems[0]: f"{stove_id} 0.50 0.50 0.20 0.20\n",
        stems[1]: f"{gas_id} 0.30 0.30 0.10 0.10\n",
        stems[2]: f"{stove_id} 0.40 0.40 0.15 0.15\n{gas_id} 0.70 0.70 0.10 0.10\n",
    }
    # Annotator B: same boxes, slightly shifted (still high-IoU matches).
    labels_b = {
        stems[0]: f"{stove_id} 0.51 0.50 0.20 0.20\n",
        stems[1]: f"{gas_id} 0.31 0.30 0.10 0.10\n",
        stems[2]: f"{stove_id} 0.41 0.40 0.15 0.15\n{gas_id} 0.71 0.70 0.10 0.10\n",
    }

    for annotator, labels in (("asha", labels_a), ("ravi", labels_b)):
        export_path = _make_cvat_export(
            tmp_path / f"export_{annotator}.zip", names_in_order, labels
        )
        export = read_yolo_export(export_path)
        assert verify_class_order(export.names, class_names) == []
        validation = validate_session_labels(
            export,
            set(stems),
            class_names,
            config.annotation.min_labeled_fraction,
            trusted_classes=meta.trusted_classes,
        )
        assert validation.problems == []
        assert validation.warnings == []
        stage_annotations(export, meta.session_id, annotator, config.annotation.staging_dir)

    labels_staged_a = read_yolo_export(tmp_path / "export_asha.zip").labels
    labels_staged_b = read_yolo_export(tmp_path / "export_ravi.zip").labels
    parsed_a = {k: [parse_yolo_line(line) for line in v] for k, v in labels_staged_a.items()}
    parsed_b = {k: [parse_yolo_line(line) for line in v] for k, v in labels_staged_b.items()}
    report = compare_annotators(
        parsed_a,
        parsed_b,
        config.annotation.iaa.iou_threshold,
        class_names,
        annotator_a="asha",
        annotator_b="ravi",
    )
    verdict, failures = agreement_verdict(report, config.annotation.iaa)
    assert verdict == "pass", failures
    assert report.overall_agreement == pytest.approx(1.0)

    finalize_result = finalize_annotations(
        config.annotation.staging_dir, meta.session_id, "asha", config.captures_root, class_names
    )
    assert finalize_result.labels_written == 3
    assert finalize_result.class_counts == {"gas_cylinder": 2, "stove": 2}

    session_manifest = load_session_manifests(config.captures_root)[0]
    assert session_manifest.annotation_status == "finalized"
    assert session_manifest.annotators == ["asha"]

    # ── 5. Merge: custom capture + a synthetic public source ────────────────
    public_root = _public_source(tmp_path / "raw" / "public", n_groups=5, seed_base=500)
    merged = tmp_path / "merged"
    manifest = merge_sources(
        sources=[
            MergeSource(
                name="custom_captures",
                root=config.captures_root,
                trusted_classes=list(meta.trusted_classes),
                labels_dir=None,  # already taxonomy-id — no remap stage needed
            ),
            MergeSource(
                name="public",
                root=public_root,
                trusted_classes=["person"],
                labels_dir=None,
            ),
        ],
        output_dir=merged,
        dedup_settings=DedupSettings(),
        indoor_settings=IndoorFilterSettings(enabled=False),
        class_names=class_names,
    )
    assert manifest.class_counts["stove"] == 2
    assert manifest.class_counts["gas_cylinder"] == 2
    assert manifest.class_counts["person"] == 5
    assert sorted(manifest.label_completeness["custom_captures"]) == ["gas_cylinder", "stove"]
    assert len(manifest.image_provenance) == 8  # 3 custom + 5 public

    # ── 6. Group split — session AND house integrity ────────────────────────
    images = find_image_files(merged / "images")
    groups = group_files_by_key(images)
    custom_group_keys = [k for k in groups if k.startswith("custom_captures_")]
    assert custom_group_keys == ["custom_captures_h01_kitchen_s001"]
    assert len(groups[custom_group_keys[0]]) == 3  # all 3 session images share the group

    for strategy_name in ("group_aware", "leave_one_house_out"):
        assignments = get_strategy(strategy_name).assign(SplitContext(groups=groups, seed=42))
        key_to_split = {k: s for s, keys in assignments.items() for k in keys}
        assert len({key_to_split[k] for k in custom_group_keys}) == 1  # one house, one split

    processed = tmp_path / "processed"
    assignments = get_strategy("group_aware").assign(SplitContext(groups=groups, seed=42))
    copy_split_files(
        groups=groups,
        assignments=assignments,
        images_source_dir=merged / "images",
        labels_source_dir=merged / "labels",
        output_dir=processed,
    )
    assert verify_no_leakage(processed) == []

    # ── 7. Structural QA (the real CLI, as the qa_check DVC stage runs it) ──
    reports = tmp_path / "qa_reports"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "qa" / "check_annotations.py"),
            "--data-dir",
            str(processed),
            "--config",
            str(REPO_ROOT / "configs" / "data.yaml"),
            "--output",
            str(reports),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode in (0, 2), completed.stdout[-2000:]
    qa_report = json.loads((reports / "annotation_qa_report.json").read_text(encoding="utf-8"))
    assert qa_report["summary"]["critical_issues"] == 0

    # ── 8. Progress tracking sees the finalized session ─────────────────────
    progress = build_progress_report(
        config.captures_root,
        config.eval_root,
        CollectionTargets(
            total_images=1,
            min_instances_per_class=1,
            custom_classes=("stove", "gas_cylinder"),
            min_houses=1,
        ),
        {},
    )
    assert progress.class_counts == {"stove": 2, "gas_cylinder": 2}
    assert progress.houses == {"h01"}
    assert progress.annotation_status_counts == {"finalized": 1}

    # ── 9. Eval set: a deliberate train-duplicate is caught, then fixed ─────
    eval_inbox = tmp_path / "eval_inbox"
    eval_inbox.mkdir()
    (eval_inbox / "dup_of_train.jpg").write_bytes(
        (config.captures_root / "images" / f"{stems[0]}.jpg").read_bytes()
    )
    eval_meta = SessionMeta(
        session_id="h02_hall_s001",
        house_id="h02",
        room="hall",
        lighting="daylight",
        capture_device="TestCam",
        captured_at="2026-07-21",
        consent_reference="",  # ingest_session itself doesn't gate on consent;
    )  # the CLI (08_...) does, before calling it
    ingest_session(eval_inbox, eval_meta, config, config.eval_root)

    overlap_report, overlap_critical = check_eval_overlap(config.eval_root, processed, merged)
    assert overlap_critical is True
    assert overlap_report["exact_overlap_count"] == 1

    house_report = check_house_exclusivity(config.captures_root, config.eval_root)
    assert house_report["shared_houses"] == []  # h01 (train) vs h02 (eval)

    # Fix: remove the offending eval image and its manifest entry, re-derive.
    shutil.rmtree(config.eval_root)
    eval_inbox_clean = tmp_path / "eval_inbox_clean"
    eval_inbox_clean.mkdir()
    _block_image(eval_inbox_clean / "clean.jpg", seed=777)
    ingest_session(eval_inbox_clean, eval_meta, config, config.eval_root)

    overlap_report, overlap_critical = check_eval_overlap(config.eval_root, processed, merged)
    assert overlap_critical is False
    assert overlap_report["exact_overlap_count"] == 0
    assert overlap_report["near_overlap_count"] == 0

    # ── 10. Lock the eval set; further ingest is refused ────────────────────
    lock_eval_set(config.eval_root)
    assert is_eval_locked(config.eval_root) is True

    another_inbox = tmp_path / "another_inbox"
    another_inbox.mkdir()
    _block_image(another_inbox / "late.jpg", seed=888)
    with pytest.raises(ValueError, match="LOCKED"):
        ingest_session(another_inbox, eval_meta, config, config.eval_root)

"""
Integration test: dataset pipeline end-to-end on offline fixtures.

Exercises the real Phase-2 chain — remap (copy mode) → merge (dedup +
provenance) → group-aware split → structural QA — over synthetic sources,
with no network access. Mirrors what the DVC stages do, minus the
downloads.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.dataset.manifest import MERGED_MANIFEST_FILENAME, MergedManifest
from src.dataset.merge import MergeSource, merge_sources
from src.dataset.remap import remap_label_dir
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings
from src.dataset.splitting import SplitContext, get_strategy
from src.utils.dataset_utils import find_image_files, group_files_by_key

PIL = pytest.importorskip("PIL", reason="Pillow required for image fixtures")
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Distinct source class vocabularies remapped into the taxonomy
ALPHA_TABLE = {"person": 0, "knife": 5}
BETA_TABLE = {"face": 1}
CLASS_NAMES = {0: "person", 1: "face", 5: "knife"}


def _make_image(path: Path, seed: int) -> None:
    img = Image.new("L", (400, 400))
    img.putdata(
        [
            ((x // 10) * (seed * 17 % 89) + (y // 10) * 11) % 256
            for y in range(400)
            for x in range(400)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _make_source(
    root: Path,
    classes: dict[str, str],
    n_groups: int,
    label_line: str,
    seed_base: int,
) -> Path:
    """Fake raw source: grouped frames, local-id labels, sidecar."""
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "source_classes.json").write_text(json.dumps(classes), encoding="utf-8")
    for g in range(n_groups):
        for f in range(3):
            stem = f"clip{g:03d}_frame_{f}"
            _make_image(root / "images" / f"{stem}.jpg", seed=seed_base + g * 3 + f)
            (root / "labels" / f"{stem}.txt").write_text(label_line, encoding="utf-8")
    return root


@pytest.mark.integration
def test_dataset_pipeline_end_to_end(tmp_path: Path) -> None:
    # ── Fixture sources ─────────────────────────────────────────────────
    alpha = _make_source(
        tmp_path / "raw" / "alpha",
        classes={"0": "person", "1": "knife", "2": "pizza"},  # pizza → dropped
        n_groups=10,
        label_line="0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n2 0.8 0.8 0.1 0.1\n",
        seed_base=1,
    )
    beta = _make_source(
        tmp_path / "raw" / "beta",
        classes={"0": "face"},
        n_groups=5,
        label_line="0 0.5 0.4 0.1 0.1\n",
        seed_base=500,
    )

    # ── 1. Remap (copy mode, like the remap_classes DVC stage) ─────────
    interim = tmp_path / "interim"
    result_a = remap_label_dir(alpha, ALPHA_TABLE, output_labels_dir=interim / "alpha" / "labels")
    result_b = remap_label_dir(beta, BETA_TABLE, output_labels_dir=interim / "beta" / "labels")
    assert result_a.annotations_dropped == 30  # one pizza per image
    assert result_a.annotations_remapped == 60
    assert result_b.annotations_remapped == 15
    # Raw labels untouched by copy mode
    assert "2 0.8" in (alpha / "labels" / "clip000_frame_0.txt").read_text(encoding="utf-8")

    # ── 2. Merge (dedup + provenance + completeness) ────────────────────
    merged = tmp_path / "merged"
    manifest = merge_sources(
        sources=[
            MergeSource(
                name="alpha",
                root=alpha,
                trusted_classes=["person", "knife"],
                labels_dir=interim / "alpha" / "labels",
            ),
            MergeSource(
                name="beta",
                root=beta,
                trusted_classes=["face"],
                labels_dir=interim / "beta" / "labels",
            ),
        ],
        output_dir=merged,
        dedup_settings=DedupSettings(),
        indoor_settings=IndoorFilterSettings(enabled=False),
        class_names=CLASS_NAMES,
    )
    assert len(manifest.image_provenance) == 45
    assert manifest.class_counts == {"person": 30, "knife": 30, "face": 15}
    assert manifest.label_completeness["alpha"] == ["person", "knife"]
    loaded = MergedManifest.load(merged / MERGED_MANIFEST_FILENAME)
    assert loaded.image_provenance == manifest.image_provenance

    # ── 3. Group-aware split (like split_train_val_test) ────────────────
    processed = tmp_path / "processed"
    images = find_image_files(merged / "images")
    groups = group_files_by_key(images)
    # Source-prefixed names keep groups intact: 15 groups total
    assert len(groups) == 15

    assignments = get_strategy("group_aware").assign(SplitContext(groups=groups, seed=42))

    from scripts.dataset.split_dataset import copy_split_files, verify_no_leakage

    copy_split_files(
        groups=groups,
        assignments=assignments,
        images_source_dir=merged / "images",
        labels_source_dir=merged / "labels",
        output_dir=processed,
    )
    assert verify_no_leakage(processed) == []

    # Group integrity: all frames of each clip in exactly one split
    for split_name in ("train", "val", "test"):
        for img in (processed / "images" / split_name).iterdir():
            group_key = img.name.rsplit("_frame_", 1)[0]
            for other in ("train", "val", "test"):
                if other == split_name:
                    continue
                overlap = list((processed / "images" / other).glob(f"{group_key}_frame_*"))
                assert not overlap, f"group {group_key} straddles {split_name}/{other}"

    # ── 4. Structural QA (the real CLI, as the qa_check stage runs it) ──
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

    report = json.loads((reports / "annotation_qa_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["critical_issues"] == 0
    assert report["summary"]["total_images"] == 45
    for leak_check in ("train_val_leakage", "train_test_leakage"):
        assert report["checks"][leak_check]["status"] == "PASS"

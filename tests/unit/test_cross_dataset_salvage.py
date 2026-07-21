"""Unit tests for L3 cross-dataset label salvage (ADR-P5-08, D7).

Pure-function tests for src.dataset.cross_dataset_salvage, plus real
merge_sources() integration for both the exact-sha256 transplant path and
the near-dup cross_dataset_links path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.cross_dataset_salvage import (
    build_cross_dataset_link,
    render_transplanted_lines,
    transplant_trusted_boxes,
)
from src.dataset.merge import CROSS_DATASET_LINKS_FILENAME, MergeSource, merge_sources
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings
from src.utils.annotation_utils import Annotation

PIL = pytest.importorskip("PIL", reason="Pillow required for image fixtures")
from PIL import Image  # noqa: E402

pytestmark = pytest.mark.unit

_CLASS_NAMES = {0: "person", 1: "charger", 2: "wire"}


def _ann(
    class_id: int, cx: float = 0.5, cy: float = 0.5, w: float = 0.2, h: float = 0.2
) -> Annotation:
    return Annotation(class_id=class_id, cx=cx, cy=cy, w=w, h=h, line_num=1, raw="")


class TestTransplantTrustedBoxes:
    def test_untrusted_class_never_transplants(self) -> None:
        result = transplant_trusted_boxes(
            dropped_annotations=[_ann(1)],  # charger
            dropped_trusted_classes=frozenset({"person"}),  # charger not trusted
            class_names_by_id=_CLASS_NAMES,
            kept_annotations=[],
        )
        assert result.transplanted == ()

    def test_trusted_class_with_no_overlap_transplants(self) -> None:
        result = transplant_trusted_boxes(
            dropped_annotations=[_ann(1)],
            dropped_trusted_classes=frozenset({"charger"}),
            class_names_by_id=_CLASS_NAMES,
            kept_annotations=[],
        )
        assert len(result.transplanted) == 1
        assert result.transplanted[0].class_id == 1

    def test_high_iou_same_class_box_is_suppressed(self) -> None:
        """D7: both sources labeling the same object must not double-box."""
        result = transplant_trusted_boxes(
            dropped_annotations=[_ann(1, cx=0.5, cy=0.5, w=0.2, h=0.2)],
            dropped_trusted_classes=frozenset({"charger"}),
            class_names_by_id=_CLASS_NAMES,
            kept_annotations=[_ann(1, cx=0.51, cy=0.5, w=0.2, h=0.2)],  # near-identical box
        )
        assert result.transplanted == ()
        assert len(result.suppressed) == 1

    def test_low_iou_same_class_box_still_transplants(self) -> None:
        """A different object of the same class elsewhere in the image."""
        result = transplant_trusted_boxes(
            dropped_annotations=[_ann(1, cx=0.1, cy=0.1, w=0.1, h=0.1)],
            dropped_trusted_classes=frozenset({"charger"}),
            class_names_by_id=_CLASS_NAMES,
            kept_annotations=[_ann(1, cx=0.9, cy=0.9, w=0.1, h=0.1)],  # far away
        )
        assert len(result.transplanted) == 1

    def test_unknown_class_id_is_skipped_not_crashed(self) -> None:
        result = transplant_trusted_boxes(
            dropped_annotations=[_ann(99)],
            dropped_trusted_classes=frozenset({"person"}),
            class_names_by_id=_CLASS_NAMES,
            kept_annotations=[],
        )
        assert result.transplanted == ()


class TestRenderTransplantedLines:
    def test_renders_yolo_format(self) -> None:
        lines = render_transplanted_lines((_ann(1, 0.5, 0.5, 0.2, 0.2),))
        assert lines == ["1 0.5 0.5 0.2 0.2"]


class TestBuildCrossDatasetLink:
    def test_no_eligible_box_returns_none(self) -> None:
        link = build_cross_dataset_link(
            dropped_annotations=[_ann(1)],
            dropped_trusted_classes=frozenset({"person"}),
            class_names_by_id=_CLASS_NAMES,
            dropped_source="roboflow",
        )
        assert link is None

    def test_eligible_box_included_with_provenance(self) -> None:
        link = build_cross_dataset_link(
            dropped_annotations=[_ann(1, 0.5, 0.5, 0.2, 0.2)],
            dropped_trusted_classes=frozenset({"charger"}),
            class_names_by_id=_CLASS_NAMES,
            dropped_source="roboflow",
        )
        assert link == {"source": "roboflow", "boxes": [[1, 0.5, 0.5, 0.2, 0.2]]}


# ─── merge_sources() integration ────────────────────────────────────────────


def _make_image(path: Path, seed: int, size: tuple[int, int] = (400, 400)) -> Path:
    width, height = size
    img = Image.new("L", size)
    img.putdata(
        [
            ((x // 8) * (seed * 13 % 97) + (y // 8) * 7) % 256
            for y in range(height)
            for x in range(width)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _perturb_corner(path: Path) -> None:
    """Mutate a small corner region — changes file bytes but stays within
    the default Hamming threshold (aHash averages 8x8 cells; one 20x20
    corner out of 400x400 barely moves one cell's average)."""
    img = Image.open(path)
    img = img.convert("L")
    pixels = img.load()
    for x in range(20):
        for y in range(20):
            pixels[x, y] = 255 - pixels[x, y]
    img.save(path)


def _make_source_dir(
    tmp_path: Path, name: str, images: dict[str, int], labels: dict[str, str]
) -> Path:
    root = tmp_path / name
    for img_name, seed in images.items():
        _make_image(root / "images" / img_name, seed=seed)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for lbl_name, content in labels.items():
        (root / "labels" / lbl_name).write_text(content, encoding="utf-8")
    return root


class TestExactDuplicateSalvageIntegration:
    def test_exact_duplicate_transplants_trusted_class_not_kept_by_first_source(
        self, tmp_path: Path
    ) -> None:
        # alpha only trusts 'person'; beta (byte-identical dup) trusts 'charger'.
        src_a = _make_source_dir(
            tmp_path, "alpha", images={"a1.png": 7}, labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n"}
        )
        src_b = _make_source_dir(
            tmp_path, "beta", images={"b1.png": 7}, labels={"b1.txt": "1 0.1 0.1 0.1 0.1\n"}
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[
                MergeSource(name="alpha", root=src_a, trusted_classes=["person"]),
                MergeSource(name="beta", root=src_b, trusted_classes=["charger"]),
            ],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=_CLASS_NAMES,
        )

        kept_label = (out / "labels" / "alpha_a1.txt").read_text(encoding="utf-8")
        assert "0 0.5 0.5 0.2 0.2" in kept_label
        assert "1 0.1 0.1 0.1 0.1" in kept_label  # salvaged from beta
        assert manifest.labels_salvaged == 1
        assert manifest.class_counts["charger"] == 1
        assert not (out / "images" / "beta_b1.png").exists()

    def test_exact_duplicate_does_not_double_box_same_class(self, tmp_path: Path) -> None:
        src_a = _make_source_dir(
            tmp_path, "alpha", images={"a1.png": 7}, labels={"a1.txt": "1 0.5 0.5 0.2 0.2\n"}
        )
        src_b = _make_source_dir(
            tmp_path,
            "beta",
            images={"b1.png": 7},
            labels={"b1.txt": "1 0.51 0.5 0.2 0.2\n"},  # near-identical box, same class
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[
                MergeSource(name="alpha", root=src_a, trusted_classes=["charger"]),
                MergeSource(name="beta", root=src_b, trusted_classes=["charger"]),
            ],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=_CLASS_NAMES,
        )

        kept_label = (out / "labels" / "alpha_a1.txt").read_text(encoding="utf-8")
        assert kept_label.count("1 0.5") == 1  # only alpha's original box — beta's suppressed
        assert manifest.labels_salvaged == 0

    def test_no_trusted_classes_transplants_nothing(self, tmp_path: Path) -> None:
        """Backward-compat: sources with no trusted_classes declared (the
        pre-L3 default) behave exactly as before — nothing salvaged."""
        src_a = _make_source_dir(
            tmp_path, "alpha", images={"a1.png": 7}, labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n"}
        )
        src_b = _make_source_dir(
            tmp_path, "beta", images={"b1.png": 7}, labels={"b1.txt": "1 0.1 0.1 0.1 0.1\n"}
        )
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[MergeSource(name="alpha", root=src_a), MergeSource(name="beta", root=src_b)],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=_CLASS_NAMES,
        )

        kept_label = (out / "labels" / "alpha_a1.txt").read_text(encoding="utf-8")
        assert "1 0.1 0.1 0.1 0.1" not in kept_label
        assert manifest.labels_salvaged == 0


class TestNearDuplicateLinkIntegration:
    def test_near_duplicate_writes_cross_dataset_link_not_a_transplant(
        self, tmp_path: Path
    ) -> None:
        src_a = _make_source_dir(
            tmp_path, "alpha", images={"a1.png": 7}, labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n"}
        )
        src_b = _make_source_dir(
            tmp_path, "beta", images={"b1.png": 7}, labels={"b1.txt": "1 0.1 0.1 0.1 0.1\n"}
        )
        _perturb_corner(src_b / "images" / "b1.png")  # near-dup, NOT byte-identical
        out = tmp_path / "merged"

        manifest = merge_sources(
            sources=[
                MergeSource(name="alpha", root=src_a, trusted_classes=["person"]),
                MergeSource(name="beta", root=src_b, trusted_classes=["charger"]),
            ],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=_CLASS_NAMES,
        )

        # Never transplanted directly — geometry is unverified for a near-dup.
        kept_label = (out / "labels" / "alpha_a1.txt").read_text(encoding="utf-8")
        assert "1 0.1 0.1 0.1 0.1" not in kept_label
        assert manifest.labels_salvaged == 0
        assert manifest.cross_dataset_candidates_linked == 1

        links = json.loads((out / CROSS_DATASET_LINKS_FILENAME).read_text(encoding="utf-8"))
        assert links["alpha_a1.png"] == [{"source": "beta", "boxes": [[1, 0.1, 0.1, 0.1, 0.1]]}]

    def test_no_duplicates_writes_empty_links_file(self, tmp_path: Path) -> None:
        src_a = _make_source_dir(
            tmp_path, "alpha", images={"a1.png": 1}, labels={"a1.txt": "0 0.5 0.5 0.2 0.2\n"}
        )
        out = tmp_path / "merged"

        merge_sources(
            sources=[MergeSource(name="alpha", root=src_a, trusted_classes=["person"])],
            output_dir=out,
            dedup_settings=DedupSettings(),
            indoor_settings=IndoorFilterSettings(enabled=False),
            class_names=_CLASS_NAMES,
        )

        links = json.loads((out / CROSS_DATASET_LINKS_FILENAME).read_text(encoding="utf-8"))
        assert links == {}

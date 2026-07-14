"""Unit tests for the source-specific downloader parsers (offline).

COCO / Open Images / WIDER FACE / negatives are exercised against
fabricated annotation indexes and archives pre-placed in the download
cache — which also exercises the resume-by-cache path (fetch_url skips
existing files). Per-image network fetches are stubbed at the instance
level; no test touches the network.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.dataset.downloaders.base import DownloadSkippedError
from src.dataset.downloaders.coco import CocoDownloader
from src.dataset.downloaders.negatives_dl import NegativesDownloader
from src.dataset.downloaders.openimages import OpenImagesDownloader
from src.dataset.downloaders.wider_face import WiderFaceDownloader
from src.dataset.sources_config import SourceConfig, SourcesConfig

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

# ─── Shared helpers ───────────────────────────────────────────────────────────


def _stub_image_fetch(downloader: Any, monkeypatch: Any, fail_urls: set[str] | None = None) -> None:
    """Replace per-URL fetches with an offline stub writing fake bytes."""

    def _fake(url: str, dest: Path, **kwargs: Any) -> bool:
        if fail_urls and url in fail_urls:
            return False
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake-image-bytes")
        return True

    monkeypatch.setattr(downloader, "fetch_url", _fake)


def _prepare_dirs(downloader: Any) -> None:
    for directory in (downloader.images_dir, downloader.labels_dir, downloader.downloads_dir):
        directory.mkdir(parents=True, exist_ok=True)


# ─── COCO ─────────────────────────────────────────────────────────────────────

# Alphabetical local ids over REMAP_TABLES["coco"]:
# bed=0 book=1 bottle=2 chair=3 knife=4 laptop=5 person=6 sink=7 toilet=8 tv=9
COCO_PERSON_LOCAL_ID = 6
COCO_CHAIR_LOCAL_ID = 3


def make_coco_downloader(
    tmp_path: Path,
    annotations: dict[str, Any],
    class_caps: dict[str, int] | None = None,
    extra_sources: dict[str, SourceConfig] | None = None,
) -> CocoDownloader:
    """Build a CocoDownloader with a pre-cached instances JSON."""
    source = SourceConfig(
        name="coco",
        output_dir=tmp_path / "raw" / "coco",
        license="CC BY 4.0",
        options={
            "annotations_url": "http://coco/annotations_trainval2017.zip",
            "image_url_template": "http://coco/{split}/{file_name}",
            "smoke_split": "val2017",
            "class_caps": class_caps or {},
        },
    )
    sources = {"coco": source, **(extra_sources or {})}
    config = SourcesConfig(sources=sources, downloads_cache=tmp_path / "downloads_cache")
    downloader = CocoDownloader(source, config)
    _prepare_dirs(downloader)
    (downloader.downloads_dir / "instances_val2017.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    return downloader


def coco_annotations() -> dict[str, Any]:
    """Three images: person+zebra, crowd-person+chair, degenerate-person."""
    return {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "chair"},
            {"id": 3, "name": "zebra"},
        ],
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 11, "file_name": "b.jpg", "width": 100, "height": 100},
            {"id": 12, "file_name": "c.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 1, "bbox": [10, 10, 50, 50], "iscrowd": 0},
            {"image_id": 10, "category_id": 3, "bbox": [0, 0, 20, 20], "iscrowd": 0},
            {"image_id": 11, "category_id": 1, "bbox": [5, 5, 40, 40], "iscrowd": 1},
            {"image_id": 11, "category_id": 2, "bbox": [0, 0, 30, 30], "iscrowd": 0},
            {"image_id": 12, "category_id": 1, "bbox": [5, 5, 0.5, 20], "iscrowd": 0},
        ],
    }


@pytest.mark.unit
class TestCocoDownloader:
    """Annotation parsing, class filtering, caps, and determinism."""

    def test_source_classes_are_alphabetical(self, tmp_path: Path) -> None:
        downloader = make_coco_downloader(tmp_path, coco_annotations())
        classes = downloader.source_classes()
        assert classes["6"] == "person"
        assert classes["3"] == "chair"
        assert len(classes) == 10

    def test_fetch_writes_wanted_labels_only(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = make_coco_downloader(tmp_path, coco_annotations())
        _stub_image_fetch(downloader, monkeypatch)

        counts = downloader.fetch(limit=None)

        # a.jpg: person kept, zebra (unwanted) dropped.
        a_label = (downloader.labels_dir / "a.txt").read_text(encoding="utf-8").splitlines()
        assert len(a_label) == 1
        assert a_label[0].split()[0] == str(COCO_PERSON_LOCAL_ID)
        # b.jpg: crowd person dropped, chair kept.
        b_label = (downloader.labels_dir / "b.txt").read_text(encoding="utf-8").splitlines()
        assert [line.split()[0] for line in b_label] == [str(COCO_CHAIR_LOCAL_ID)]
        # c.jpg: only a degenerate box → image not acquired at all.
        assert not (downloader.images_dir / "c.jpg").exists()
        assert counts == {"person": 1, "chair": 1}

    def test_class_caps_skip_saturated_images(self, tmp_path: Path, monkeypatch: Any) -> None:
        annotations = {
            "categories": [{"id": 1, "name": "person"}],
            "images": [
                {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},
                {"id": 11, "file_name": "b.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"image_id": 10, "category_id": 1, "bbox": [10, 10, 50, 50], "iscrowd": 0},
                {"image_id": 11, "category_id": 1, "bbox": [10, 10, 50, 50], "iscrowd": 0},
            ],
        }
        downloader = make_coco_downloader(tmp_path, annotations, class_caps={"person": 1})
        _stub_image_fetch(downloader, monkeypatch)

        counts = downloader.fetch(limit=None)

        assert counts == {"person": 1}
        assert len(list(downloader.images_dir.glob("*.jpg"))) == 1

    def test_limit_caps_selection_deterministically(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = make_coco_downloader(tmp_path, coco_annotations())
        _stub_image_fetch(downloader, monkeypatch)

        downloader.fetch(limit=1)

        # Lowest image id first (deterministic ordering).
        assert (downloader.images_dir / "a.jpg").exists()
        assert not (downloader.images_dir / "b.jpg").exists()

    def test_failed_image_fetch_is_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = make_coco_downloader(tmp_path, coco_annotations())
        _stub_image_fetch(downloader, monkeypatch, fail_urls={"http://coco/val2017/a.jpg"})

        counts = downloader.fetch(limit=None)

        assert not (downloader.labels_dir / "a.txt").exists()
        assert counts == {"chair": 1}

    def test_build_image_class_index(self, tmp_path: Path) -> None:
        downloader = make_coco_downloader(tmp_path, coco_annotations())
        index = downloader.build_image_class_index()
        assert index["a.jpg"] == {"person", "zebra"}
        assert index["c.jpg"] == {"person"}


# ─── Open Images ──────────────────────────────────────────────────────────────


def make_openimages_downloader(
    tmp_path: Path,
    descriptions_csv: str,
    bbox_csv: str,
    classes: list[str] | None = None,
) -> OpenImagesDownloader:
    """Build an OpenImagesDownloader with pre-cached CSV indexes."""
    source = SourceConfig(
        name="openimages",
        output_dir=tmp_path / "raw" / "openimages",
        options={
            "class_descriptions_url": "http://oi/class-descriptions.csv",
            "smoke_bbox_url": "http://oi/validation-annotations-bbox.csv",
            "smoke_split": "validation",
            "image_url_template": "http://oi/{split}/{image_id}.jpg",
            "classes": classes or ["Cupboard", "Door"],
        },
    )
    config = SourcesConfig(
        sources={"openimages": source}, downloads_cache=tmp_path / "downloads_cache"
    )
    downloader = OpenImagesDownloader(source, config)
    _prepare_dirs(downloader)
    (downloader.downloads_dir / "class-descriptions.csv").write_text(
        descriptions_csv, encoding="utf-8"
    )
    (downloader.downloads_dir / "validation-annotations-bbox.csv").write_text(
        bbox_csv, encoding="utf-8"
    )
    return downloader


OI_DESCRIPTIONS = "/m/1,Door\n/m/2,Cupboard\n/m/3,Zebra\n"
OI_BBOXES = (
    "ImageID,LabelName,XMin,XMax,YMin,YMax\n"
    "img1,/m/1,0.1,0.5,0.2,0.6\n"  # Door
    "img1,/m/3,0.0,0.4,0.0,0.4\n"  # Zebra — unwanted
    "img2,/m/2,0.3,0.3,0.1,0.2\n"  # Cupboard — zero width, dropped
    "img2,/m/2,0.2,0.8,0.2,0.8\n"  # Cupboard — valid
)


@pytest.mark.unit
class TestOpenImagesDownloader:
    """CSV streaming, normalized-bbox conversion, and validation."""

    def test_fetch_converts_normalized_boxes(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = make_openimages_downloader(tmp_path, OI_DESCRIPTIONS, OI_BBOXES)
        _stub_image_fetch(downloader, monkeypatch)

        counts = downloader.fetch(limit=None)

        # sorted(classes): Cupboard=0, Door=1
        img1 = (downloader.labels_dir / "img1.txt").read_text(encoding="utf-8").splitlines()
        assert len(img1) == 1  # zebra row ignored
        parts = img1[0].split()
        assert parts[0] == "1"
        assert [float(v) for v in parts[1:]] == pytest.approx([0.3, 0.4, 0.4, 0.4])
        # img2: degenerate row dropped, valid row kept.
        img2 = (downloader.labels_dir / "img2.txt").read_text(encoding="utf-8").splitlines()
        assert [line.split()[0] for line in img2] == ["0"]
        assert counts == {"Door": 1, "Cupboard": 1}

    def test_limit_caps_images(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = make_openimages_downloader(tmp_path, OI_DESCRIPTIONS, OI_BBOXES)
        _stub_image_fetch(downloader, monkeypatch)

        downloader.fetch(limit=1)

        assert (downloader.images_dir / "img1.jpg").exists()
        assert not (downloader.images_dir / "img2.jpg").exists()

    def test_unknown_class_raises(self, tmp_path: Path) -> None:
        downloader = make_openimages_downloader(
            tmp_path, OI_DESCRIPTIONS, OI_BBOXES, classes=["Door", "Hoverboard"]
        )
        with pytest.raises(RuntimeError, match="Hoverboard"):
            downloader.fetch(limit=None)


# ─── WIDER FACE ───────────────────────────────────────────────────────────────

WIDER_GT = (
    "0--Parade/img1.jpg\n"
    "3\n"
    "10 10 40 40 0 0 0 0 0 0\n"  # valid face
    "5 5 4 4 0 0 0 0 0 0\n"  # tiny (<8px) — dropped
    "20 20 30 30 0 0 0 1 0 0\n"  # invalid flag — dropped
    "1--Fest/img2.jpg\n"
    "0\n"
    "0 0 0 0 0 0 0 0 0 0\n"  # count==0 placeholder line
)


def make_wider_downloader(
    tmp_path: Path,
    noncommercial: bool = True,
    allow_noncommercial: bool = True,
) -> WiderFaceDownloader:
    """Build a WiderFaceDownloader with pre-cached annotation/image zips."""
    source = SourceConfig(
        name="wider_face",
        output_dir=tmp_path / "raw" / "wider_face",
        noncommercial=noncommercial,
        license="research-only",
        options={
            "smoke_images_url": "http://wf/WIDER_val.zip",
            "annotations_url": "http://wf/wider_face_split.zip",
        },
    )
    config = SourcesConfig(
        sources={"wider_face": source},
        downloads_cache=tmp_path / "downloads_cache",
        allow_noncommercial=allow_noncommercial,
    )
    downloader = WiderFaceDownloader(source, config)
    _prepare_dirs(downloader)

    with zipfile.ZipFile(downloader.downloads_dir / "wider_face_split.zip", "w") as zf:
        zf.writestr("wider_face_split/wider_face_val_bbx_gt.txt", WIDER_GT)

    jpeg = io.BytesIO()
    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(jpeg, format="JPEG")
    with zipfile.ZipFile(downloader.downloads_dir / "WIDER_val.zip", "w") as zf:
        zf.writestr("WIDER_val/images/0--Parade/img1.jpg", jpeg.getvalue())

    return downloader


@pytest.mark.unit
class TestWiderFaceDownloader:
    """License gate, quirky bbx_gt parsing, and zip extraction."""

    def test_license_gate_raises_skip(self, tmp_path: Path) -> None:
        downloader = make_wider_downloader(tmp_path, allow_noncommercial=False)
        with pytest.raises(DownloadSkippedError, match="non-commercial"):
            downloader.fetch(limit=None)

    def test_parse_ground_truth_filters_boxes(self, tmp_path: Path) -> None:
        downloader = make_wider_downloader(tmp_path)
        ground_truth = downloader._parse_ground_truth(
            downloader.downloads_dir / "wider_face_split.zip", "val"
        )
        # Only the one valid box survives; the zero-count image is excluded.
        assert ground_truth == {"0--Parade/img1.jpg": [[10.0, 10.0, 40.0, 40.0]]}

    def test_fetch_extracts_and_flattens(self, tmp_path: Path) -> None:
        downloader = make_wider_downloader(tmp_path)

        counts = downloader.fetch(limit=None)

        flat = downloader.images_dir / "0--Parade_img1.jpg"
        assert flat.exists()
        label = downloader.labels_dir / "0--Parade_img1.txt"
        lines = label.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert lines[0].split()[0] == "0"
        assert counts == {"face": 1}


# ─── Negatives ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNegativesDownloader:
    """Background selection from the cached COCO index."""

    def _make(self, tmp_path: Path) -> NegativesDownloader:
        annotations = {
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 4, "name": "couch"},  # confusable — excluded
            ],
            "images": [
                {"id": 20, "file_name": "clean.jpg", "width": 100, "height": 100},
                {"id": 21, "file_name": "has_couch.jpg", "width": 100, "height": 100},
                {"id": 22, "file_name": "has_person.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"image_id": 21, "category_id": 4, "bbox": [0, 0, 30, 30], "iscrowd": 0},
                {"image_id": 22, "category_id": 1, "bbox": [0, 0, 30, 30], "iscrowd": 0},
            ],
        }
        neg_source = SourceConfig(
            name="negatives",
            output_dir=tmp_path / "raw" / "negatives",
            options={"smoke_count": 5},
        )
        coco = make_coco_downloader(tmp_path, annotations, extra_sources={"negatives": neg_source})
        downloader = NegativesDownloader(neg_source, coco.config)
        _prepare_dirs(downloader)
        return downloader

    def test_selects_only_clean_images(self, tmp_path: Path, monkeypatch: Any) -> None:
        downloader = self._make(tmp_path)
        _stub_image_fetch(downloader, monkeypatch)

        counts = downloader.fetch(limit=None)

        assert counts == {}
        images = [p.name for p in downloader.images_dir.glob("*.jpg")]
        assert images == ["clean.jpg"]
        # Negatives carry an EMPTY label file by definition.
        assert (downloader.labels_dir / "clean.txt").read_text(encoding="utf-8") == ""

    def test_requires_coco_source(self, tmp_path: Path) -> None:
        neg_source = SourceConfig(name="negatives", output_dir=tmp_path / "raw" / "negatives")
        config = SourcesConfig(
            sources={"negatives": neg_source}, downloads_cache=tmp_path / "downloads_cache"
        )
        downloader = NegativesDownloader(neg_source, config)
        _prepare_dirs(downloader)
        with pytest.raises(RuntimeError, match="coco"):
            downloader.fetch(limit=None)

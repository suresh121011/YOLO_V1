"""Unit tests for the ``mode="full"`` selection branches of every downloader
(final-audit Fix-7: all downloader tests constructed smoke configs, so the
full-mode split/url/count selection lines were never executed).

No real network or download happens — the pure selection methods are asserted
directly, and where the full-mode branch lives inside ``fetch()`` the network
seam is monkeypatched so only the selection line runs.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import src.dataset.downloaders.negatives_dl as neg_mod
from src.dataset.downloaders.coco import CocoDownloader
from src.dataset.downloaders.negatives_dl import NegativesDownloader
from src.dataset.downloaders.openimages import OpenImagesDownloader
from src.dataset.downloaders.wider_face import WiderFaceDownloader
from src.dataset.sources_config import SourceConfig, SourcesConfig

pytestmark = pytest.mark.unit


def _full_config(
    tmp_path: Path, sources: dict[str, SourceConfig], **kwargs: object
) -> SourcesConfig:
    return SourcesConfig(
        sources=sources,
        mode="full",
        downloads_cache=tmp_path / "cache",
        **kwargs,  # type: ignore[arg-type]
    )


def test_coco_full_mode_selects_full_split(tmp_path: Path) -> None:
    source = SourceConfig(
        name="coco",
        output_dir=tmp_path / "raw" / "coco",
        options={
            "smoke_split": "val2017",
            "full_split": "train2017",
            "annotations_url": "http://x/annotations_trainval2017.zip",
            "image_url_template": "http://x/{split}/{file_name}",
        },
    )
    downloader = CocoDownloader(source, _full_config(tmp_path, {"coco": source}))
    assert downloader._split() == "train2017"
    assert downloader._query_extras()["split"] == "train2017"


def test_openimages_full_mode_selects_full_split_and_bbox_url(tmp_path: Path) -> None:
    source = SourceConfig(
        name="openimages",
        output_dir=tmp_path / "raw" / "openimages",
        options={
            "smoke_split": "validation",
            "full_split": "train",
            "smoke_bbox_url": "http://x/val-annotations-bbox.csv",
            "full_bbox_url": "http://x/train-annotations-bbox.csv",
            "class_descriptions_url": "http://x/class-descriptions.csv",
        },
    )
    downloader = OpenImagesDownloader(source, _full_config(tmp_path, {"openimages": source}))
    assert downloader._split() == "train"
    assert downloader._bbox_url() == "http://x/train-annotations-bbox.csv"


def test_wider_face_full_mode_selects_train_split(tmp_path: Path) -> None:
    source = SourceConfig(name="wider_face", output_dir=tmp_path / "raw" / "wf")
    downloader = WiderFaceDownloader(source, _full_config(tmp_path, {"wider_face": source}))
    assert downloader._split() == "train"


def test_wider_face_full_mode_fetch_uses_full_images_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``full_images_url`` selection lives inside ``fetch()`` (line ~58);
    monkeypatch the network + ground-truth parse so only that line runs."""
    source = SourceConfig(
        name="wider_face",
        output_dir=tmp_path / "raw" / "wf",
        noncommercial=True,
        options={
            "smoke_images_url": "http://x/WIDER_val.zip",
            "full_images_url": "http://x/WIDER_train.zip",
            "annotations_url": "http://x/wider_face_split.zip",
        },
    )
    config = _full_config(tmp_path, {"wider_face": source}, allow_noncommercial=True)
    downloader = WiderFaceDownloader(source, config)
    for directory in (downloader.images_dir, downloader.labels_dir, downloader.downloads_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []

    def _fake_fetch(url: str, dest: Path, **kwargs: object) -> bool:
        fetched.append(str(url))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w"):  # empty archive — extraction loop no-ops
            pass
        return True

    monkeypatch.setattr(downloader, "fetch_url", _fake_fetch)
    monkeypatch.setattr(downloader, "_parse_ground_truth", lambda zip_path, split: {})

    downloader.fetch(limit=None)

    assert "http://x/WIDER_train.zip" in fetched
    assert "http://x/WIDER_val.zip" not in fetched


def test_negatives_full_mode_uses_full_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``full_count`` is read inside ``fetch()`` (line ~48); capture the count
    handed to the (monkeypatched) selector to prove the full branch ran."""
    coco_source = SourceConfig(
        name="coco",
        output_dir=tmp_path / "raw" / "coco",
        options={
            "smoke_split": "val2017",
            "full_split": "train2017",
            "image_url_template": "http://x/{split}/{file_name}",
        },
    )
    neg_source = SourceConfig(
        name="negatives",
        output_dir=tmp_path / "raw" / "negatives",
        options={"smoke_count": 20, "full_count": 500},
    )
    config = _full_config(tmp_path, {"coco": coco_source, "negatives": neg_source})
    downloader = NegativesDownloader(neg_source, config)
    for directory in (downloader.images_dir, downloader.labels_dir, downloader.downloads_dir):
        directory.mkdir(parents=True, exist_ok=True)

    captured: dict[str, int] = {}

    def _capture(index: object, excluded_classes: object, count: int) -> list[str]:
        captured["count"] = count
        return []

    monkeypatch.setattr(neg_mod, "select_negative_candidates", _capture)
    monkeypatch.setattr(neg_mod.CocoDownloader, "build_image_class_index", lambda self: {})

    downloader.fetch(limit=None)

    assert captured["count"] == 500

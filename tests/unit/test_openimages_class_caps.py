"""OpenImages ``class_caps`` support.

The downloader had no cap mechanism at all — only smoke mode's ``limit``,
which is ``None`` in full mode. A full-mode build therefore fetched every
train image containing Door, Cupboard or Gas stove, unbounded and with no
configuration knob to bound it.

Caps here follow COCO's design exactly (``base.is_capped_out``): an image is
fetched only when every class in it still has budget. The property these
tests care about beyond the count is *where* the check happens — before the
network call, because not downloading the image is the entire point.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.dataset.downloaders.openimages import OpenImagesDownloader
from src.dataset.sources_config import SourceConfig, SourcesConfig

pytestmark = pytest.mark.unit

_MIDS = {"/m/door": "Door", "/m/cup": "Cupboard", "/m/gas": "Gas stove"}
_BBOX_URL = "http://x/oidv6-train-annotations-bbox.csv"


def _write_bbox_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows: (image_id, mid) — one row per box, full-frame coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])
        for image_id, mid in rows:
            writer.writerow([image_id, mid, "0.1", "0.5", "0.1", "0.5"])


def _downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[str, str]],
    caps: dict[str, int],
) -> tuple[OpenImagesDownloader, list[str]]:
    source = SourceConfig(
        name="openimages",
        output_dir=tmp_path / "raw" / "openimages",
        options={
            "smoke_split": "validation",
            "full_split": "train",
            "smoke_bbox_url": _BBOX_URL,
            "full_bbox_url": _BBOX_URL,
            "class_descriptions_url": "http://x/desc.csv",
            "image_url_template": "http://x/{split}/{image_id}.jpg",
            "classes": ["Door", "Cupboard", "Gas stove"],
            "class_caps": caps,
        },
    )
    config = SourcesConfig(
        sources={"openimages": source}, mode="full", downloads_cache=tmp_path / "cache"
    )
    downloader = OpenImagesDownloader(source, config)
    downloader.images_dir.mkdir(parents=True, exist_ok=True)
    downloader.labels_dir.mkdir(parents=True, exist_ok=True)
    _write_bbox_csv(downloader.downloads_dir / Path(_BBOX_URL).name, rows)

    monkeypatch.setattr(OpenImagesDownloader, "_load_mid_map", lambda self: dict(_MIDS))

    fetched: list[str] = []

    def fake_fetch_url(self: OpenImagesDownloader, url: str, dest: Path, **kw: object) -> bool:
        if dest.suffix == ".csv":  # the pre-written index, never re-fetched
            return dest.exists()
        fetched.append(dest.stem)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(OpenImagesDownloader, "fetch_url", fake_fetch_url)
    return downloader, fetched


class TestOpenImagesCaps:
    def test_cap_blocks_further_images_of_that_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [(f"img{i}", "/m/door") for i in range(5)]
        downloader, fetched = _downloader(tmp_path, monkeypatch, rows, {"Door": 2})
        counts = downloader.fetch(limit=None)
        assert counts["Door"] == 2
        assert len(fetched) == 2

    def test_capped_image_is_never_downloaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check must precede fetch_url, not filter after it.

        A cap that discards an image after paying for the download saves
        nothing — bandwidth is the resource being budgeted.
        """
        rows = [(f"img{i}", "/m/door") for i in range(5)]
        downloader, fetched = _downloader(tmp_path, monkeypatch, rows, {"Door": 1})
        downloader.fetch(limit=None)
        assert fetched == ["img0"], "images past the cap must not be requested at all"

    def test_saturated_class_does_not_ride_along(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [("img0", "/m/door"), ("img1", "/m/door"), ("img1", "/m/gas"), ("img2", "/m/gas")]
        downloader, fetched = _downloader(tmp_path, monkeypatch, rows, {"Door": 1, "Gas stove": 10})
        counts = downloader.fetch(limit=None)
        assert counts["Door"] == 1
        assert "img1" not in fetched, "a saturated Door must block the mixed image"
        assert counts["Gas stove"] == 1, "the Door-free image is still recruited"

    def test_no_caps_configured_is_unbounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compatibility — the pre-cap behaviour, exactly."""
        rows = [(f"img{i}", "/m/door") for i in range(7)]
        downloader, fetched = _downloader(tmp_path, monkeypatch, rows, {})
        counts = downloader.fetch(limit=None)
        assert counts["Door"] == 7
        assert len(fetched) == 7

    def test_caps_are_per_class_not_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [("a", "/m/door"), ("b", "/m/door"), ("c", "/m/cup"), ("d", "/m/cup")]
        downloader, _ = _downloader(tmp_path, monkeypatch, rows, {"Door": 1, "Cupboard": 2})
        counts = downloader.fetch(limit=None)
        assert counts["Door"] == 1
        assert counts["Cupboard"] == 2

    def test_limit_and_caps_compose(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Smoke mode's limit is orthogonal to caps and still binds first."""
        rows = [(f"img{i}", "/m/door") for i in range(10)]
        downloader, fetched = _downloader(tmp_path, monkeypatch, rows, {"Door": 8})
        downloader.fetch(limit=3)
        assert len(fetched) == 3

    def test_caps_recorded_in_query_extras(self, tmp_path: Path) -> None:
        """The manifest records the budget the build actually used."""
        source = SourceConfig(
            name="openimages",
            output_dir=tmp_path / "raw" / "openimages",
            options={
                "full_split": "train",
                "classes": ["Door"],
                "class_caps": {"Door": 1500},
            },
        )
        config = SourcesConfig(
            sources={"openimages": source}, mode="full", downloads_cache=tmp_path / "cache"
        )
        extras = OpenImagesDownloader(source, config)._query_extras()
        assert extras["class_caps"] == {"Door": 1500}

"""COCO ``class_caps`` selection semantics.

The original rule fetched an image when ANY class in it was under cap, then
credited every box in that image. A saturated class therefore rode along on
images recruited by other classes and kept growing: on train2017 an 800-box
``person`` cap produced 36,469 person boxes (45x) and ``chair``'s 500 produced
18,009 — the exact opposite of the "prevent imbalance" the config claims.
Smoke mode capped the whole source at 60 images, so it never showed there.

These tests pin the corrected rule: an image is fetched only when EVERY class
in it still has budget. The important case is not that the cap holds — it is
that a saturated common class does not block a rare co-occurring one from
reaching its own cap via images that do not contain the saturated class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.dataset.downloaders.coco import CocoDownloader
from src.dataset.sources_config import SourceConfig, SourcesConfig

pytestmark = pytest.mark.unit

# Real COCO category ids for the names under test.
_CATS = {1: "person", 49: "knife", 84: "book"}


def _instances(images: list[tuple[int, list[str]]]) -> dict[str, Any]:
    """Build a minimal COCO instances dict.

    Args:
        images: (image_id, [class name per box]) — one annotation per name,
            so a repeated name means several boxes of that class.
    """
    name_to_id = {v: k for k, v in _CATS.items()}
    ann_id = 0
    annotations = []
    for image_id, names in images:
        for name in names:
            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": name_to_id[name],
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "iscrowd": 0,
                }
            )
    return {
        "categories": [{"id": cid, "name": name} for cid, name in _CATS.items()],
        "images": [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 640, "height": 480}
            for image_id, _ in images
        ],
        "annotations": annotations,
    }


def _downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instances: dict[str, Any],
    caps: dict[str, int],
) -> tuple[CocoDownloader, list[str]]:
    """A downloader whose network seam records fetches instead of performing them."""
    source = SourceConfig(
        name="coco",
        output_dir=tmp_path / "raw" / "coco",
        options={
            "smoke_split": "val2017",
            "full_split": "train2017",
            "annotations_url": "http://x/annotations_trainval2017.zip",
            "image_url_template": "http://x/{split}/{file_name}",
            "class_caps": caps,
        },
    )
    config = SourcesConfig(
        sources={"coco": source}, mode="full", downloads_cache=tmp_path / "cache"
    )
    downloader = CocoDownloader(source, config)
    # run() normally creates these; these tests drive fetch() directly.
    downloader.images_dir.mkdir(parents=True, exist_ok=True)
    downloader.labels_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CocoDownloader, "load_instances", lambda self: instances)

    fetched: list[str] = []

    def fake_fetch_url(self: CocoDownloader, url: str, dest: Path, **kwargs: object) -> bool:
        fetched.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(CocoDownloader, "fetch_url", fake_fetch_url)
    return downloader, fetched


class TestSaturatedClassBlocks:
    def test_saturated_class_blocks_images_containing_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression: image 2 must NOT be fetched once person is at cap."""
        instances = _instances([(1, ["person"]), (2, ["person"]), (3, ["person"])])
        downloader, fetched = _downloader(tmp_path, monkeypatch, instances, {"person": 1})
        counts = downloader.fetch(limit=None)
        assert len(fetched) == 1
        assert counts["person"] == 1

    def test_saturated_class_does_not_ride_along_on_other_recruits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 36,469-box bug in miniature.

        Under the old rule image 2 was fetched because `knife` was under cap,
        and its person box was then credited, pushing person past its cap.
        """
        instances = _instances([(1, ["person"]), (2, ["person", "knife"]), (3, ["knife"])])
        downloader, fetched = _downloader(
            tmp_path, monkeypatch, instances, {"person": 1, "knife": 10}
        )
        counts = downloader.fetch(limit=None)
        assert counts["person"] == 1, "a capped class must not accumulate via co-occurrence"
        assert "000000000002.jpg" not in fetched

    def test_rare_class_still_reaches_its_cap_without_the_saturated_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strictness must not starve co-occurring rare classes.

        Measured on the real train2017 at cap 1200, `knife` reaches its full
        cap alongside a saturated `person`; this is that property in the small.
        """
        instances = _instances(
            [(1, ["person"]), (2, ["person", "knife"]), (3, ["knife"]), (4, ["knife"])]
        )
        downloader, _ = _downloader(tmp_path, monkeypatch, instances, {"person": 1, "knife": 10})
        counts = downloader.fetch(limit=None)
        assert counts["knife"] == 2, "knife-only images must still be recruited"


class TestCapAccounting:
    def test_uncapped_class_is_unbounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compatibility: a class with no cap entry is not gated."""
        instances = _instances([(i, ["book"]) for i in range(1, 6)])
        downloader, fetched = _downloader(tmp_path, monkeypatch, instances, {})
        counts = downloader.fetch(limit=None)
        assert len(fetched) == 5
        assert counts["book"] == 5

    def test_overshoot_bounded_by_boxes_in_the_final_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cap 2, and one image carrying 3 person boxes → 3, not unbounded.

        The cap gates entry to an image, so the last admitted image can carry
        the class past its cap by however many boxes it holds. That residue is
        inherent and bounded; the 45x growth it replaces was not.
        """
        instances = _instances([(1, ["person", "person", "person"]), (2, ["person"])])
        downloader, fetched = _downloader(tmp_path, monkeypatch, instances, {"person": 2})
        counts = downloader.fetch(limit=None)
        assert counts["person"] == 3
        assert len(fetched) == 1, "image 2 must be blocked once person is over cap"

    def test_limit_still_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke mode's limit is independent of caps and still applies."""
        instances = _instances([(i, ["book"]) for i in range(1, 11)])
        downloader, fetched = _downloader(tmp_path, monkeypatch, instances, {})
        downloader.fetch(limit=3)
        assert len(fetched) == 3

    def test_caps_are_recorded_in_query_extras(self, tmp_path: Path) -> None:
        """Caps land in the manifest, so a build records the budget it used."""
        source = SourceConfig(
            name="coco",
            output_dir=tmp_path / "raw" / "coco",
            options={"full_split": "train2017", "class_caps": {"person": 1200}},
        )
        config = SourcesConfig(
            sources={"coco": source}, mode="full", downloads_cache=tmp_path / "cache"
        )
        assert CocoDownloader(source, config)._query_extras()["class_caps"] == {"person": 1200}

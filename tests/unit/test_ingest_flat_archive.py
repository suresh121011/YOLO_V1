"""Flat-archive ingestion support (Phase B / medicine_bottle).

Most local capture archives wrap one or more ``*.yolov11.zip`` exports inside an
outer ZIP. ``medicine bottle.zip`` is flat instead — ``<name>/data.yaml`` plus
``<name>/images/`` and ``<name>/labels/`` directly in the outer archive — so
``ingest_one`` bailed out with "No inner ZIPs found ... skipping" and the whole
source was silently dropped. That is why ``medicine_bottle`` had 0 instances
while a perfectly good 501-image dataset sat in ``Dataset/``.

These tests pin three things: flat archives are detected and ingested, the
nested path is byte-for-byte unaffected, and a malformed archive still fails
loudly rather than ingesting as empty.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ingest = importlib.import_module("scripts.dataset.20_ingest_local_zips")

pytestmark = pytest.mark.unit


def _png_bytes(size: tuple[int, int] = (32, 24)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 150)).save(buf, format="PNG")
    return buf.getvalue()


_DATA_YAML = b"""path: .
train: images
val: images

names:
  0: 'vase'
  1: 'medicine bottle'
  2: 'bottle'
  3: 'person'
"""


def _flat_zip(path: Path, root: str = "medicine bottle", n_images: int = 3) -> Path:
    """Build a flat archive: <root>/data.yaml + <root>/images/ + <root>/labels/."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root}/data.yaml", _DATA_YAML)
        for i in range(n_images):
            zf.writestr(f"{root}/images/img_{i}.png", _png_bytes())
            # one in-taxonomy 'medicine bottle' (1) + one out-of-taxonomy 'vase' (0)
            zf.writestr(
                f"{root}/labels/img_{i}.txt",
                "1 0.500000 0.500000 0.200000 0.200000\n0 0.100000 0.100000 0.050000 0.050000\n",
            )
    return path


def _nested_zip(path: Path, n_images: int = 2) -> Path:
    """Build the conventional wrapper layout: outer ZIP containing an inner ZIP."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as izf:
        izf.writestr("data.yaml", _DATA_YAML)
        for i in range(n_images):
            izf.writestr(f"train/images/n_{i}.png", _png_bytes())
            izf.writestr(f"train/labels/n_{i}.txt", "1 0.400000 0.400000 0.300000 0.300000\n")
    with zipfile.ZipFile(path, "w") as ozf:
        ozf.writestr("dataset.yolov11.zip", inner.getvalue())
    return path


class TestIsFlatYoloArchive:
    def test_flat_layout_detected(self, tmp_path: Path) -> None:
        with zipfile.ZipFile(_flat_zip(tmp_path / "a.zip")) as zf:
            assert ingest._is_flat_yolo_archive(zf) is True

    def test_wrapper_archive_not_flat(self, tmp_path: Path) -> None:
        with zipfile.ZipFile(_nested_zip(tmp_path / "b.zip")) as zf:
            assert ingest._is_flat_yolo_archive(zf) is False

    def test_empty_archive_not_flat(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.zip"
        with zipfile.ZipFile(p, "w"):
            pass
        with zipfile.ZipFile(p) as zf:
            assert ingest._is_flat_yolo_archive(zf) is False

    def test_yaml_without_images_not_flat(self, tmp_path: Path) -> None:
        """A stray data.yaml must not be mistaken for a dataset."""
        p = tmp_path / "yaml_only.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("thing/data.yaml", _DATA_YAML)
        with zipfile.ZipFile(p) as zf:
            assert ingest._is_flat_yolo_archive(zf) is False

    def test_images_without_yaml_not_flat(self, tmp_path: Path) -> None:
        p = tmp_path / "img_only.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("thing/images/x.png", _png_bytes())
        with zipfile.ZipFile(p) as zf:
            assert ingest._is_flat_yolo_archive(zf) is False


class TestIngestOneFlat:
    @staticmethod
    def _rec(zip_path: Path) -> dict:
        return {
            "zip": zip_path,
            "key": "medicine bottle",
            "slug": "medicine_bottle",
            "primary_class": "medicine_bottle",
            "temporary": False,
        }

    def test_flat_archive_is_ingested(self, tmp_path: Path) -> None:
        z = _flat_zip(tmp_path / "medicine bottle.zip", n_images=3)
        out = tmp_path / "out"
        summary = ingest.ingest_one(self._rec(z), out)

        assert summary is not None, "flat archive was skipped"
        assert summary["image_count"] == 3
        images = sorted((out / "images").glob("medicine_bottle__*"))
        labels = sorted((out / "labels").glob("medicine_bottle__*.txt"))
        assert len(images) == 3
        assert len(labels) == 3

    def test_labels_remapped_and_out_of_taxonomy_dropped(self, tmp_path: Path) -> None:
        z = _flat_zip(tmp_path / "medicine bottle.zip", n_images=2)
        out = tmp_path / "out"
        ingest.ingest_one(self._rec(z), out)

        for label in (out / "labels").glob("*.txt"):
            rows = [r for r in label.read_text(encoding="utf-8").splitlines() if r.strip()]
            # source 'medicine bottle'(1) -> taxonomy 3; 'vase'(0) dropped
            assert len(rows) == 1
            assert rows[0].split()[0] == "3"

    def test_labels_written_with_lf(self, tmp_path: Path) -> None:
        """Labels land inside a DVC output — CRLF would make the hash OS-specific."""
        z = _flat_zip(tmp_path / "medicine bottle.zip", n_images=1)
        out = tmp_path / "out"
        ingest.ingest_one(self._rec(z), out)
        for label in (out / "labels").glob("*.txt"):
            assert b"\r\n" not in label.read_bytes()

    def test_provenance_written(self, tmp_path: Path) -> None:
        z = _flat_zip(tmp_path / "medicine bottle.zip")
        out = tmp_path / "out"
        ingest.ingest_one(self._rec(z), out)
        dest = out / "medicine_bottle"
        assert (dest / "manifest.json").exists()
        assert (dest / "source_classes.json").exists()
        assert (dest / ingest.SENTINEL_FILENAME).exists()
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source"] == "local_captures/medicine_bottle"

    def test_malformed_archive_still_skipped(self, tmp_path: Path) -> None:
        """No inner ZIPs AND not flat must fail loudly, not ingest as empty."""
        p = tmp_path / "junk.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("readme.txt", "nothing useful here")
        assert ingest.ingest_one(self._rec(p), tmp_path / "out") is None


class TestNestedPathUnaffected:
    def test_wrapper_archive_still_ingests(self, tmp_path: Path) -> None:
        """The 22 existing archives must keep the original code path."""
        z = _nested_zip(tmp_path / "Knife-20260722T103738Z-1-001.zip", n_images=2)
        rec = {
            "zip": z,
            "key": "knife",
            "slug": "knife",
            "primary_class": "knife",
            "temporary": False,
        }
        summary = ingest.ingest_one(rec, tmp_path / "out")
        assert summary is not None
        assert summary["image_count"] == 2


class TestMedicineBottleTables:
    def test_zip_meta_entry_present(self) -> None:
        assert ingest._ZIP_META["medicine bottle"] == (
            "medicine_bottle",
            "medicine_bottle",
            False,
        )

    def test_stem_key_resolves_archive_name(self, tmp_path: Path) -> None:
        assert ingest._stem_key(tmp_path / "medicine bottle.zip") == "medicine bottle"

    def test_bottle_maps_to_medicine_bottle_not_water_bottle(self) -> None:
        """All 17 'bottle' crops were visually confirmed to be medicine bottles.

        Guarding the value explicitly: mapping to 4 (water_bottle) would push
        medicine bottles into a different safety class.
        """
        remap = ingest._ARCHIVE_CLASS_REMAPS["medicine bottle"]
        assert remap["bottle"] == 3
        assert remap["medicine bottle"] == 3

    def test_secondary_in_taxonomy_classes_kept(self) -> None:
        remap = ingest._ARCHIVE_CLASS_REMAPS["medicine bottle"]
        assert remap["person"] == 0
        assert remap["book"] == 9
        assert remap["medicine strip"] == 2

    @pytest.mark.parametrize("ambiguous", ["cylinder", "gas", "dining table", "vase", "buttle"])
    def test_ambiguous_source_classes_dropped(self, ambiguous: str) -> None:
        assert ambiguous not in ingest._ARCHIVE_CLASS_REMAPS["medicine bottle"]

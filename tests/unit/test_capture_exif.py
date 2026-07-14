"""Unit tests for src.dataset.capture.exif — metadata inspection and stripping."""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL", reason="Pillow required for EXIF tests")

from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

from src.dataset.capture.exif import GPS_IFD_TAG, inspect_metadata, strip_metadata  # noqa: E402


def _jpeg_with_gps(path: Path, size: tuple[int, int] = (600, 500)) -> Path:
    img = Image.new("RGB", size, (120, 80, 40))
    exif = Image.Exif()
    exif[0x0110] = "TestCam"  # Model
    gps = exif.get_ifd(GPS_IFD_TAG)
    gps[1] = "N"
    gps[2] = (13.0, 4.0, 0.0)
    img.save(path, exif=exif)
    return path


def _clean_jpeg(path: Path, size: tuple[int, int] = (600, 500)) -> Path:
    Image.new("RGB", size, (10, 200, 90)).save(path)
    return path


@pytest.mark.unit
class TestInspectMetadata:
    """Metadata detection."""

    def test_detects_exif_and_gps(self, tmp_path: Path) -> None:
        report = inspect_metadata(_jpeg_with_gps(tmp_path / "gps.jpg"))
        assert report["has_exif"] is True
        assert report["has_gps"] is True
        assert report["clean"] is False

    def test_clean_image_reports_clean(self, tmp_path: Path) -> None:
        report = inspect_metadata(_clean_jpeg(tmp_path / "clean.jpg"))
        assert report["clean"] is True
        assert report["has_gps"] is False

    def test_detects_png_text_chunks(self, tmp_path: Path) -> None:
        info = PngInfo()
        info.add_text("Author", "someone")
        path = tmp_path / "meta.png"
        Image.new("RGB", (600, 500)).save(path, pnginfo=info)
        report = inspect_metadata(path)
        assert report["has_text"] is True
        assert report["clean"] is False


@pytest.mark.unit
class TestStripMetadata:
    """Privacy stripping."""

    def test_strips_gps_exif_from_jpeg(self, tmp_path: Path) -> None:
        src = _jpeg_with_gps(tmp_path / "gps.jpg")
        dst = tmp_path / "out" / "stripped.jpg"
        strip_metadata(src, dst)
        report = inspect_metadata(dst)
        assert report["clean"] is True
        with Image.open(dst) as img:
            assert img.size == (600, 500)

    def test_strips_png_text(self, tmp_path: Path) -> None:
        info = PngInfo()
        info.add_text("Author", "someone")
        src = tmp_path / "meta.png"
        Image.new("RGB", (600, 500)).save(src, pnginfo=info)
        dst = tmp_path / "stripped.png"
        strip_metadata(src, dst)
        assert inspect_metadata(dst)["clean"] is True

    def test_bakes_orientation_into_pixels(self, tmp_path: Path) -> None:
        # Orientation 6 = rotate 90° CW on display; after stripping, the
        # rotation must survive in the pixel data (600×500 → 500×600).
        src = tmp_path / "rotated.jpg"
        img = Image.new("RGB", (600, 500), (50, 50, 50))
        exif = Image.Exif()
        exif[0x0112] = 6
        img.save(src, exif=exif)

        dst = tmp_path / "stripped.jpg"
        strip_metadata(src, dst)
        report = inspect_metadata(dst)
        assert report["clean"] is True
        with Image.open(dst) as out:
            assert out.size == (500, 600)

    def test_source_left_untouched(self, tmp_path: Path) -> None:
        src = _jpeg_with_gps(tmp_path / "gps.jpg")
        before = src.read_bytes()
        strip_metadata(src, tmp_path / "out.jpg")
        assert src.read_bytes() == before

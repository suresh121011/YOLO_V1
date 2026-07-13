"""Unit tests for src.dataset.dedup and src.dataset.filters."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.dedup import DedupIndex, compute_image_hashes
from src.dataset.filters import check_image_filter, compute_mean_brightness
from src.dataset.sources_config import DedupSettings, IndoorFilterSettings

PIL = pytest.importorskip("PIL", reason="Pillow required for image fixtures")
from PIL import Image  # noqa: E402


def _make_image(
    path: Path,
    size: tuple[int, int] = (400, 400),
    brightness: int = 100,
    gradient: bool = True,
) -> Path:
    """Write a synthetic grayscale test image with structure (not flat)."""
    img = Image.new("L", size)
    width, height = size
    if gradient:
        # Diagonal gradient — gives the aHash real structure to fingerprint.
        img.putdata(
            [
                min(255, brightness + ((x * 7 + y * 3) % 120) - 60)
                for y in range(height)
                for x in range(width)
            ]
        )
    else:
        img.putdata([brightness] * (width * height))
    img.save(path)
    return path


@pytest.mark.unit
class TestComputeImageHashes:
    """Perceptual hashing incl. mirrored variant."""

    def test_identical_images_share_hash(self, tmp_path: Path) -> None:
        a = _make_image(tmp_path / "a.png")
        b = _make_image(tmp_path / "b.png")
        ha = compute_image_hashes(a)
        hb = compute_image_hashes(b)
        assert ha.ahash == hb.ahash
        assert ha.flip_ahash is not None

    def test_flip_hash_matches_mirrored_image(self, tmp_path: Path) -> None:
        original = _make_image(tmp_path / "orig.png")
        with Image.open(original) as img:
            img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(tmp_path / "mirror.png")

        orig_hashes = compute_image_hashes(original)
        mirror_hashes = compute_image_hashes(tmp_path / "mirror.png")
        # The mirrored image's plain hash equals the original's flip hash.
        assert mirror_hashes.ahash == orig_hashes.flip_ahash

    def test_flip_disabled(self, tmp_path: Path) -> None:
        path = _make_image(tmp_path / "a.png")
        hashes = compute_image_hashes(path, check_flip=False)
        assert hashes.ahash is not None
        assert hashes.flip_ahash is None


@pytest.mark.unit
class TestDedupIndex:
    """Cross-image duplicate detection."""

    def test_exact_duplicate_detected(self, tmp_path: Path) -> None:
        index = DedupIndex(settings=DedupSettings())
        first = _make_image(tmp_path / "first.png")
        copy = _make_image(tmp_path / "copy.png")

        assert index.check_and_add(first) is None
        assert index.check_and_add(copy) == first
        assert index.kept_count == 1

    def test_mirrored_duplicate_detected(self, tmp_path: Path) -> None:
        index = DedupIndex(settings=DedupSettings(check_flips=True))
        original = _make_image(tmp_path / "orig.png")
        with Image.open(original) as img:
            img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(tmp_path / "mirror.png")

        assert index.check_and_add(original) is None
        assert index.check_and_add(tmp_path / "mirror.png") == original

    def test_mirrored_duplicate_missed_without_flip_check(self, tmp_path: Path) -> None:
        index = DedupIndex(settings=DedupSettings(check_flips=False))
        original = _make_image(tmp_path / "orig.png")
        with Image.open(original) as img:
            img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(tmp_path / "mirror.png")

        assert index.check_and_add(original) is None
        # Asymmetric gradient → mirrored hash differs beyond the threshold.
        assert index.check_and_add(tmp_path / "mirror.png") is None

    def test_distinct_images_kept(self, tmp_path: Path) -> None:
        index = DedupIndex(settings=DedupSettings())
        # Coarse structural difference that survives 8×8 downscaling:
        # vertical split vs. horizontal split.
        a = Image.new("L", (400, 400))
        a.putdata([255 if x < 200 else 0 for y in range(400) for x in range(400)])
        a.save(tmp_path / "a.png")
        b = Image.new("L", (400, 400))
        b.putdata([255 if y < 200 else 0 for y in range(400) for x in range(400)])
        b.save(tmp_path / "b.png")

        assert index.check_and_add(tmp_path / "a.png") is None
        assert index.check_and_add(tmp_path / "b.png") is None
        assert index.kept_count == 2


@pytest.mark.unit
class TestFilters:
    """Quality/indoor heuristics."""

    def test_brightness_computation(self, tmp_path: Path) -> None:
        dark = _make_image(tmp_path / "dark.png", brightness=30, gradient=False)
        value = compute_mean_brightness(dark)
        assert value is not None
        assert value == pytest.approx(30, abs=2)

    def test_too_small_rejected(self, tmp_path: Path) -> None:
        small = _make_image(tmp_path / "small.png", size=(100, 100))
        keep, reason = check_image_filter(small, IndoorFilterSettings(min_image_dim=320))
        assert keep is False
        assert reason == "too_small"

    def test_bright_landscape_rejected_as_outdoor(self, tmp_path: Path) -> None:
        bright = _make_image(
            tmp_path / "bright.png", size=(640, 400), brightness=230, gradient=False
        )
        keep, reason = check_image_filter(bright, IndoorFilterSettings())
        assert keep is False
        assert reason == "likely_outdoor"

    def test_bright_portrait_kept(self, tmp_path: Path) -> None:
        portrait = _make_image(
            tmp_path / "portrait.png", size=(300, 600), brightness=230, gradient=False
        )
        keep, reason = check_image_filter(portrait, IndoorFilterSettings(min_image_dim=200))
        assert keep is True

    def test_normal_indoor_kept(self, tmp_path: Path) -> None:
        indoor = _make_image(tmp_path / "indoor.png", size=(640, 480), brightness=90)
        keep, reason = check_image_filter(indoor, IndoorFilterSettings())
        assert keep is True
        assert reason == ""

    def test_disabled_filter_keeps_everything(self, tmp_path: Path) -> None:
        tiny = _make_image(tmp_path / "tiny.png", size=(50, 50), brightness=250)
        keep, _ = check_image_filter(tiny, IndoorFilterSettings(enabled=False))
        assert keep is True

"""Unit tests for src.dataset.downloaders — acquisition framework.

Network access is always mocked; these tests exercise the shared
BaseDownloader plumbing (retry, resume-by-skip, atomic writes, manifest
sidecars), the Roboflow graceful-skip contract and its cross-dataset
image-budget accounting, and the shared CLI runner's exit-code contract.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from src.dataset.downloaders.base import (
    BaseDownloader,
    DownloadSkippedError,
    coco_bbox_to_yolo,
    write_yolo_label,
)
from src.dataset.downloaders.roboflow_dl import RoboflowDownloader
from src.dataset.downloaders.runner import run_acquisition_cli
from src.dataset.manifest import MANIFEST_FILENAME
from src.dataset.remap import SOURCE_CLASSES_FILENAME
from src.dataset.sources_config import SourceConfig, SourcesConfig

# ─── Fixtures & helpers ───────────────────────────────────────────────────────


def make_source(tmp_path: Path, name: str = "stub", **kwargs: Any) -> SourceConfig:
    """Build a SourceConfig rooted under tmp_path."""
    kwargs.setdefault("output_dir", tmp_path / "raw" / name)
    return SourceConfig(name=name, **kwargs)


def make_config(tmp_path: Path, source: SourceConfig, **kwargs: Any) -> SourcesConfig:
    """Build a SourcesConfig with caches rooted under tmp_path."""
    kwargs.setdefault("downloads_cache", tmp_path / "downloads_cache")
    return SourcesConfig(sources={source.name: source}, **kwargs)


class StubDownloader(BaseDownloader):
    """Minimal downloader that fabricates n_images files locally."""

    n_images = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.received_limit: int | None = None

    def fetch(self, limit: int | None) -> dict[str, int]:
        self.received_limit = limit
        for i in range(self.n_images):
            (self.images_dir / f"img_{i}.jpg").write_bytes(b"fake-image-bytes")
            write_yolo_label(self.labels_dir / f"img_{i}.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        return {"person": self.n_images}

    def source_classes(self) -> dict[str, str]:
        return {"0": "person"}


class _FakeResponse:
    """Context-manager stand-in for requests.Response (streaming)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return self._chunks


# ─── fetch_url: retry / resume / atomicity ────────────────────────────────────


@pytest.mark.unit
class TestFetchUrl:
    """Shared HTTP helper: skip-if-cached, retries, atomic writes."""

    def _downloader(self, tmp_path: Path) -> StubDownloader:
        source = make_source(tmp_path)
        return StubDownloader(source, make_config(tmp_path, source))

    def test_cached_file_skips_network(self, tmp_path: Path, monkeypatch: Any) -> None:
        import requests

        dest = tmp_path / "cached.zip"
        dest.write_bytes(b"already-here")

        def _fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("network must not be touched for cached files")

        monkeypatch.setattr(requests, "get", _fail)
        assert self._downloader(tmp_path).fetch_url("http://x/y.zip", dest) is True
        assert dest.read_bytes() == b"already-here"

    def test_downloads_atomically(self, tmp_path: Path, monkeypatch: Any) -> None:
        import requests

        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse([b"part1-", b"part2"]))
        dest = tmp_path / "sub" / "file.zip"
        assert self._downloader(tmp_path).fetch_url("http://x/y.zip", dest) is True
        assert dest.read_bytes() == b"part1-part2"
        assert not dest.with_suffix(dest.suffix + ".part").exists()

    def test_retries_then_succeeds(self, tmp_path: Path, monkeypatch: Any) -> None:
        import requests

        calls: list[int] = []

        def _flaky(*args: Any, **kwargs: Any) -> _FakeResponse:
            calls.append(1)
            if len(calls) < 3:
                raise OSError("transient network failure")
            return _FakeResponse([b"ok"])

        monkeypatch.setattr(requests, "get", _flaky)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        dest = tmp_path / "file.zip"
        assert self._downloader(tmp_path).fetch_url("http://x/y.zip", dest, retries=3) is True
        assert len(calls) == 3
        assert dest.read_bytes() == b"ok"

    def test_gives_up_after_retries(self, tmp_path: Path, monkeypatch: Any) -> None:
        import requests

        def _always_fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("permanent failure")

        monkeypatch.setattr(requests, "get", _always_fail)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        dest = tmp_path / "file.zip"
        assert self._downloader(tmp_path).fetch_url("http://x/y.zip", dest, retries=2) is False
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".part").exists()


# ─── download(): template method + manifest sidecars ─────────────────────────


@pytest.mark.unit
class TestDownloadTemplate:
    """Layout, sidecar, and manifest written by the template method."""

    def test_writes_layout_sidecar_and_manifest(self, tmp_path: Path) -> None:
        source = make_source(tmp_path, license="CC BY 4.0", trusted_classes=["person"])
        config = make_config(tmp_path, source, mode="smoke", smoke_limit=60)
        downloader = StubDownloader(source, config)

        manifest = downloader.download()

        assert downloader.images_dir.is_dir()
        assert downloader.labels_dir.is_dir()
        assert downloader.downloads_dir.is_dir()

        sidecar = json.loads(
            (source.output_dir / SOURCE_CLASSES_FILENAME).read_text(encoding="utf-8")
        )
        assert sidecar == {"0": "person"}

        assert (source.output_dir / MANIFEST_FILENAME).exists()
        assert manifest.source == "stub"
        assert manifest.license == "CC BY 4.0"
        assert manifest.image_count == 2
        assert manifest.class_counts == {"person": 2}
        assert manifest.trusted_classes == ["person"]
        assert set(manifest.image_hashes) == {"img_0.jpg", "img_1.jpg"}
        assert manifest.query["mode"] == "smoke"
        assert manifest.query["limit"] == 60

    def test_limit_defaults_to_mode_based_cap(self, tmp_path: Path) -> None:
        source = make_source(tmp_path)
        downloader = StubDownloader(source, make_config(tmp_path, source, smoke_limit=7))
        downloader.download()
        assert downloader.received_limit == 7

    def test_explicit_limit_overrides_config(self, tmp_path: Path) -> None:
        source = make_source(tmp_path)
        downloader = StubDownloader(source, make_config(tmp_path, source, smoke_limit=7))
        downloader.download(limit=3)
        assert downloader.received_limit == 3


# ─── YOLO conversion helpers ──────────────────────────────────────────────────


@pytest.mark.unit
class TestBboxHelpers:
    """coco_bbox_to_yolo clamping/rejection and label writing."""

    def test_converts_and_normalizes(self) -> None:
        result = coco_bbox_to_yolo([10.0, 20.0, 100.0, 50.0], img_w=200.0, img_h=100.0)
        assert result is not None
        cx, cy, w, h = result
        assert (cx, cy, w, h) == pytest.approx((0.3, 0.45, 0.5, 0.5))

    def test_clamps_out_of_bounds_box(self) -> None:
        result = coco_bbox_to_yolo([-10.0, -10.0, 120.0, 120.0], img_w=100.0, img_h=100.0)
        assert result is not None
        cx, cy, w, h = result
        assert w <= 1.0 and h <= 1.0
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0

    def test_rejects_subpixel_box(self) -> None:
        assert coco_bbox_to_yolo([5.0, 5.0, 0.5, 10.0], img_w=100.0, img_h=100.0) is None

    def test_rejects_degenerate_image_dims(self) -> None:
        assert coco_bbox_to_yolo([5.0, 5.0, 10.0, 10.0], img_w=0.0, img_h=100.0) is None

    def test_write_yolo_label_formats_lines(self, tmp_path: Path) -> None:
        dest = tmp_path / "label.txt"
        write_yolo_label(dest, [(3, 0.5, 0.5, 0.25, 0.125)])
        assert dest.read_text(encoding="utf-8") == "3 0.500000 0.500000 0.250000 0.125000\n"

    def test_write_yolo_label_empty_is_empty_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "label.txt"
        write_yolo_label(dest, [])
        assert dest.read_text(encoding="utf-8") == ""


# ─── Roboflow: graceful-skip contract ─────────────────────────────────────────


def _roboflow_downloader(
    tmp_path: Path, datasets: list[dict[str, Any]] | None
) -> RoboflowDownloader:
    source = make_source(
        tmp_path,
        name="roboflow",
        options={"datasets": datasets, "api_key_env": "ROBOFLOW_API_KEY"},
    )
    downloader = RoboflowDownloader(source, make_config(tmp_path, source))
    for directory in (downloader.images_dir, downloader.labels_dir, downloader.downloads_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return downloader


@pytest.mark.unit
class TestRoboflowSkips:
    """All three DownloadSkippedError paths (exit-0 contract in the CLI)."""

    def test_skips_when_no_datasets_configured(self, tmp_path: Path) -> None:
        downloader = _roboflow_downloader(tmp_path, datasets=[])
        with pytest.raises(DownloadSkippedError, match="no Roboflow Universe datasets"):
            downloader.fetch(limit=None)

    def test_skips_when_api_key_missing(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        downloader = _roboflow_downloader(tmp_path, datasets=[{"slug": "ws/proj", "version": 1}])
        with pytest.raises(DownloadSkippedError, match="ROBOFLOW_API_KEY"):
            downloader.fetch(limit=None)

    def test_skips_when_package_missing(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
        # None in sys.modules forces ImportError regardless of environment.
        monkeypatch.setitem(sys.modules, "roboflow", None)
        downloader = _roboflow_downloader(tmp_path, datasets=[{"slug": "ws/proj", "version": 1}])
        with pytest.raises(DownloadSkippedError, match="not installed"):
            downloader.fetch(limit=None)


@pytest.mark.unit
class TestRoboflowQueryExtras:
    """M7: per-dataset license strings recorded for the release-gate machinery."""

    def test_dataset_licenses_recorded_per_slug(self, tmp_path: Path) -> None:
        datasets = [
            {"slug": "ws/med-bottle", "version": 2, "license": "CC BY 4.0"},
            {"slug": "ws/gas-cylinder", "version": 1, "license": "MIT"},
        ]
        downloader = _roboflow_downloader(tmp_path, datasets=datasets)
        extras = downloader._query_extras()
        assert extras["dataset_licenses"] == {
            "ws/med-bottle": "CC BY 4.0",
            "ws/gas-cylinder": "MIT",
        }
        assert extras["datasets"] == ["ws/med-bottle:2", "ws/gas-cylinder:1"]

    def test_no_datasets_yields_empty_licenses(self, tmp_path: Path) -> None:
        downloader = _roboflow_downloader(tmp_path, datasets=[])
        assert downloader._query_extras()["dataset_licenses"] == {}


# ─── Roboflow: consolidation + cross-dataset image budget ─────────────────────


def _write_fake_export(
    export_dir: Path, class_names: list[str], n_images: int, prefix: str
) -> None:
    """Fabricate a YOLO-format Roboflow export on disk."""
    (export_dir / "train" / "images").mkdir(parents=True)
    (export_dir / "train" / "labels").mkdir(parents=True)
    (export_dir / "data.yaml").write_text(
        "names: [" + ", ".join(class_names) + "]\n", encoding="utf-8"
    )
    for i in range(n_images):
        stem = f"{prefix}_{i}"
        (export_dir / "train" / "images" / f"{stem}.jpg").write_bytes(b"fake")
        local_id = i % len(class_names)
        (export_dir / "train" / "labels" / f"{stem}.txt").write_text(
            f"{local_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )


def _install_fake_roboflow(monkeypatch: Any) -> None:
    """Inject a roboflow module whose API must never be reached."""

    class _Roboflow:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def workspace(self, name: str) -> Any:
            raise AssertionError("exports are cached — the Roboflow API must not be called")

    module = types.ModuleType("roboflow")
    module.Roboflow = _Roboflow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roboflow", module)


@pytest.mark.unit
class TestRoboflowConsolidation:
    """_consolidate_export copying, id-offsets, and budget accounting."""

    def test_consolidate_reports_images_copied(self, tmp_path: Path) -> None:
        downloader = _roboflow_downloader(tmp_path, datasets=[])
        export = tmp_path / "export_a"
        _write_fake_export(export, ["med_bottle", "charger"], n_images=3, prefix="a")

        counts, n_classes, copied = downloader._consolidate_export(export, 0, limit=None)

        assert copied == 3
        assert n_classes == 2
        assert sum(counts.values()) == 3
        assert len(list(downloader.images_dir.glob("*.jpg"))) == 3

    def test_consolidate_respects_limit(self, tmp_path: Path) -> None:
        downloader = _roboflow_downloader(tmp_path, datasets=[])
        export = tmp_path / "export_a"
        _write_fake_export(export, ["med_bottle"], n_images=5, prefix="a")

        _, _, copied = downloader._consolidate_export(export, 0, limit=2)

        assert copied == 2
        assert len(list(downloader.images_dir.glob("*.jpg"))) == 2

    def test_budget_decrements_by_images_not_classes(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Regression: the smoke budget must shrink by images copied.

        Dataset A has 2 classes / 3 images. With limit=4, dataset B must
        receive a budget of exactly 1 (4 - 3 images), not 2 (4 - 2 class
        names) — the historical accounting bug.
        """
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
        _install_fake_roboflow(monkeypatch)

        datasets = [
            {"slug": "ws/alpha", "version": 1},
            {"slug": "ws/beta", "version": 1},
        ]
        downloader = _roboflow_downloader(tmp_path, datasets=datasets)
        _write_fake_export(
            downloader.downloads_dir / "ws_alpha_v1", ["med_bottle", "charger"], 3, "a"
        )
        _write_fake_export(downloader.downloads_dir / "ws_beta_v1", ["wire"], 3, "b")

        downloader.fetch(limit=4)

        assert len(list(downloader.images_dir.glob("*.jpg"))) == 4

    def test_label_ids_are_offset_across_datasets(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
        _install_fake_roboflow(monkeypatch)

        datasets = [
            {"slug": "ws/alpha", "version": 1},
            {"slug": "ws/beta", "version": 1},
        ]
        downloader = _roboflow_downloader(tmp_path, datasets=datasets)
        _write_fake_export(
            downloader.downloads_dir / "ws_alpha_v1", ["med_bottle", "charger"], 1, "a"
        )
        _write_fake_export(downloader.downloads_dir / "ws_beta_v1", ["wire"], 1, "b")

        downloader.fetch(limit=None)

        beta_label = downloader.labels_dir / "ws_beta_v1_b_0.txt"
        assert beta_label.read_text(encoding="utf-8").split()[0] == "2"
        assert downloader.source_classes() == {
            "0": "med_bottle",
            "1": "charger",
            "2": "wire",
        }


# ─── Shared CLI runner: exit-code contract ────────────────────────────────────


RUNNER_CONFIG = """
mode: smoke
smoke:
  limit_per_source: 5
allow_noncommercial: false
paths:
  downloads_cache: {cache_dir}
sources:
  stub:
    enabled: {enabled}
    noncommercial: {noncommercial}
    output_dir: {output_dir}
    license: "test"
"""


@pytest.mark.unit
class TestRunAcquisitionCli:
    """Exit codes: 0 = success/skip (incl. license gate), 1 = failure."""

    def _write_config(
        self,
        tmp_path: Path,
        enabled: bool = True,
        noncommercial: bool = False,
    ) -> Path:
        path = tmp_path / "sources.yaml"
        path.write_text(
            RUNNER_CONFIG.format(
                cache_dir=(tmp_path / "cache").as_posix(),
                enabled=str(enabled).lower(),
                noncommercial=str(noncommercial).lower(),
                output_dir=(tmp_path / "raw" / "stub").as_posix(),
            ),
            encoding="utf-8",
        )
        return path

    def _run(self, tmp_path: Path, monkeypatch: Any, config_path: Path, factory: Any) -> int:
        monkeypatch.setattr(sys, "argv", ["prog", "--sources-config", str(config_path)])
        return run_acquisition_cli("stub", factory, "test CLI")

    def test_success_returns_zero(self, tmp_path: Path, monkeypatch: Any) -> None:
        config_path = self._write_config(tmp_path)
        assert self._run(tmp_path, monkeypatch, config_path, StubDownloader) == 0

    def test_disabled_source_skips_with_zero(self, tmp_path: Path, monkeypatch: Any) -> None:
        config_path = self._write_config(tmp_path, enabled=False)

        def _must_not_construct(*args: Any) -> BaseDownloader:
            raise AssertionError("factory must not run for a disabled source")

        assert self._run(tmp_path, monkeypatch, config_path, _must_not_construct) == 0

    def test_license_gate_skips_with_zero(self, tmp_path: Path, monkeypatch: Any) -> None:
        # noncommercial source + allow_noncommercial: false → governance gate.
        config_path = self._write_config(tmp_path, noncommercial=True)

        def _must_not_construct(*args: Any) -> BaseDownloader:
            raise AssertionError("factory must not run for a license-gated source")

        assert self._run(tmp_path, monkeypatch, config_path, _must_not_construct) == 0

    def test_download_skipped_error_returns_zero(self, tmp_path: Path, monkeypatch: Any) -> None:
        config_path = self._write_config(tmp_path)

        class _Skipping(StubDownloader):
            def fetch(self, limit: int | None) -> dict[str, int]:
                raise DownloadSkippedError("credentials missing")

        assert self._run(tmp_path, monkeypatch, config_path, _Skipping) == 0

    def test_real_failure_returns_one(self, tmp_path: Path, monkeypatch: Any) -> None:
        config_path = self._write_config(tmp_path)

        class _Failing(StubDownloader):
            def fetch(self, limit: int | None) -> dict[str, int]:
                raise ValueError("corrupt annotation index")

        assert self._run(tmp_path, monkeypatch, config_path, _Failing) == 1

    def test_missing_config_returns_one(self, tmp_path: Path, monkeypatch: Any) -> None:
        missing = tmp_path / "nope.yaml"
        monkeypatch.setattr(sys, "argv", ["prog", "--sources-config", str(missing)])
        assert run_acquisition_cli("stub", StubDownloader, "test CLI") == 1

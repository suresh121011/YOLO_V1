"""Unit tests for scripts/qa/full_build_preflight.py (gates FB1–FB6)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.qa.full_build_preflight as preflight
from scripts.qa.full_build_preflight import (
    gate_disk_space,
    gate_dvc_remote,
    gate_gpu,
    gate_mode,
    gate_onedrive,
    gate_roboflow,
    read_dvc_remote,
    run_full_build_preflight,
    write_report,
)

pytestmark = pytest.mark.unit


def _fake_disk_usage(free_gb: float) -> SimpleNamespace:
    gb = 1024**3
    return SimpleNamespace(total=500 * gb, used=100 * gb, free=int(free_gb * gb))


def _write_dvc_config(path: Path, url: str, default: str = "localstore") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[core]\n"
        "    analytics = false\n"
        f"    remote = {default}\n"
        f"['remote \"{default}\"']\n"
        f"    url = {url}\n",
        encoding="utf-8",
    )
    return path


def _write_sources_yaml(
    path: Path,
    mode: str = "smoke",
    roboflow_enabled: bool = True,
    datasets: list[dict[str, object]] | None = None,
) -> Path:
    lines = [
        f"mode: {mode}",
        "sources:",
        "  roboflow:",
        f"    enabled: {str(roboflow_enabled).lower()}",
        "    api_key_env: ROBOFLOW_API_KEY",
    ]
    if datasets:
        lines.append("    datasets:")
        for entry in datasets:
            lines.append(f"      - slug: {entry['slug']}")
    else:
        lines.append("    datasets: []")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ─── FB1 disk space ───────────────────────────────────────────────────────────


class TestGateDiskSpace:
    def test_pass_when_enough_free(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(200.0))
        result = gate_disk_space(tmp_path, min_free_gb=150.0)
        assert result.status == "pass"
        assert result.gate_id == "FB1"

    def test_fail_when_below_floor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(20.0))
        result = gate_disk_space(tmp_path, min_free_gb=150.0)
        assert result.status == "fail"
        assert "150" in result.details

    def test_walks_up_to_existing_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(200.0))
        result = gate_disk_space(tmp_path / "not" / "yet" / "created", min_free_gb=150.0)
        assert result.status == "pass"


# ─── FB2 dvc remote (+ config parsing) ────────────────────────────────────────


class TestReadDvcRemote:
    def test_parses_quoted_remote_section(self, tmp_path: Path) -> None:
        config = _write_dvc_config(tmp_path / ".dvc" / "config", url=str(tmp_path / "remote"))
        assert read_dvc_remote(config) == ("localstore", str(tmp_path / "remote"))

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_dvc_remote(tmp_path / "nope") is None

    def test_no_default_remote_returns_none(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        config.write_text("[core]\n    analytics = false\n", encoding="utf-8")
        assert read_dvc_remote(config) is None


class TestGateDvcRemote:
    def test_fail_without_remote(self, tmp_path: Path) -> None:
        result = gate_dvc_remote(tmp_path / "missing_config", min_free_gb=150.0)
        assert result.status == "fail"
        assert "C-1" in result.details

    def test_warn_for_nonlocal_remote(self, tmp_path: Path) -> None:
        config = _write_dvc_config(tmp_path / "config", url="s3://bucket/prefix")
        result = gate_dvc_remote(config, min_free_gb=150.0)
        assert result.status == "warn"
        assert "s3://bucket/prefix" in result.details

    def test_fail_for_onedrive_remote(self, tmp_path: Path) -> None:
        onedrive = tmp_path / "OneDrive" / "dvc_remote"
        onedrive.mkdir(parents=True)
        config = _write_dvc_config(tmp_path / "config", url=str(onedrive))
        result = gate_dvc_remote(config, min_free_gb=150.0)
        assert result.status == "fail"
        assert "R34" in result.details

    def test_fail_for_missing_local_path(self, tmp_path: Path) -> None:
        config = _write_dvc_config(tmp_path / "config", url=str(tmp_path / "unmounted"))
        result = gate_dvc_remote(config, min_free_gb=150.0)
        assert result.status == "fail"

    def test_pass_for_existing_local_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        remote = tmp_path / "remote"
        remote.mkdir()
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(200.0))
        config = _write_dvc_config(tmp_path / "config", url=str(remote))
        result = gate_dvc_remote(config, min_free_gb=150.0)
        assert result.status == "pass"

    def test_warn_when_remote_drive_low(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        remote = tmp_path / "remote"
        remote.mkdir()
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(10.0))
        config = _write_dvc_config(tmp_path / "config", url=str(remote))
        result = gate_dvc_remote(config, min_free_gb=150.0)
        assert result.status == "warn"


# ─── FB3 roboflow readiness ───────────────────────────────────────────────────


class TestGateRoboflow:
    def test_warn_when_disabled(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml", roboflow_enabled=False)
        result = gate_roboflow(yaml_path, env={})
        assert result.status == "warn"
        assert "medicine_bottle" in result.details

    def test_warn_when_no_slugs(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml")
        result = gate_roboflow(yaml_path, env={"ROBOFLOW_API_KEY": "k"})
        assert result.status == "warn"
        assert "H-B" in result.details

    def test_fail_when_slugs_but_no_key(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml", datasets=[{"slug": "ws/proj"}])
        result = gate_roboflow(yaml_path, env={})
        assert result.status == "fail"
        assert "ROBOFLOW_API_KEY" in result.details

    def test_pass_when_slugs_and_key(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml", datasets=[{"slug": "ws/proj"}])
        result = gate_roboflow(yaml_path, env={"ROBOFLOW_API_KEY": "k"})
        assert result.status == "pass"


# ─── FB4 gpu ──────────────────────────────────────────────────────────────────


class TestGateGpu:
    def test_pass_when_visible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (True, "CUDA device: X"))
        assert gate_gpu().status == "pass"

    def test_warn_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (False, "no GPU"))
        result = gate_gpu()
        assert result.status == "warn"
        assert "auto_annotate" in result.details

    def test_warn_when_undeterminable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (None, "unavailable"))
        assert gate_gpu().status == "warn"


# ─── FB5 onedrive hazard ──────────────────────────────────────────────────────


class TestGateOnedrive:
    def test_warn_inside_onedrive(self, tmp_path: Path) -> None:
        repo = tmp_path / "OneDrive" / "Desktop" / "repo"
        repo.mkdir(parents=True)
        result = gate_onedrive(repo)
        assert result.status == "warn"
        assert "cache" in result.details

    def test_pass_outside_onedrive(self, tmp_path: Path) -> None:
        assert gate_onedrive(tmp_path).status == "pass"


# ─── FB6 acquisition mode ─────────────────────────────────────────────────────


class TestGateMode:
    def test_warn_in_smoke_mode(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml", mode="smoke")
        result = gate_mode(yaml_path)
        assert result.status == "warn"
        assert "M7" in result.details

    def test_pass_in_full_mode(self, tmp_path: Path) -> None:
        yaml_path = _write_sources_yaml(tmp_path / "sources.yaml", mode="full")
        assert gate_mode(yaml_path).status == "pass"


# ─── Aggregation + report ─────────────────────────────────────────────────────


class TestRunAndReport:
    def test_aggregates_all_six_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(200.0))
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (True, "CUDA device: X"))
        remote = tmp_path / "remote"
        remote.mkdir()
        sources = _write_sources_yaml(tmp_path / "sources.yaml", mode="full")
        config = _write_dvc_config(tmp_path / "config", url=str(remote))

        report = run_full_build_preflight(
            repo_root=tmp_path,
            sources_yaml=sources,
            dvc_config_path=config,
            min_free_gb=150.0,
            env={"ROBOFLOW_API_KEY": "k"},
        )
        assert [r.gate_id for r in report.results] == [
            "FB1",
            "FB2",
            "FB3",
            "FB4",
            "FB5",
            "FB6",
        ]
        # roboflow slugs empty → warn; everything else passes
        assert report.verdict == "WARN"

    def test_fail_dominates_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(1.0))
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (True, "CUDA device: X"))
        sources = _write_sources_yaml(tmp_path / "sources.yaml")
        report = run_full_build_preflight(
            repo_root=tmp_path,
            sources_yaml=sources,
            dvc_config_path=tmp_path / "missing",
            min_free_gb=150.0,
            env={},
        )
        assert report.verdict == "FAIL"

    def test_write_report_emits_triplet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "disk_usage", lambda _p: _fake_disk_usage(200.0))
        monkeypatch.setattr(preflight, "detect_gpu", lambda: (True, "CUDA device: X"))
        remote = tmp_path / "remote"
        remote.mkdir()
        sources = _write_sources_yaml(tmp_path / "sources.yaml", mode="full")
        config = _write_dvc_config(tmp_path / "config", url=str(remote))
        report = run_full_build_preflight(
            repo_root=tmp_path,
            sources_yaml=sources,
            dvc_config_path=config,
            min_free_gb=150.0,
            env={"ROBOFLOW_API_KEY": "k"},
        )

        paths = write_report(report, tmp_path / "reports")
        for key in ("json", "csv", "markdown"):
            assert paths[key].exists(), key

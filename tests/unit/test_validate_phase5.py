"""Unit tests for scripts.qa.validate_phase5 — the M6 correctness-validation
gate (final-audit Fix-7: the script had zero coverage).

``subprocess`` is fully monkeypatched — no real ``pytest`` run and no real
``dvc repro`` are ever launched — so these tests exercise the gate's own
logic (the ``_NOOP_MARKERS`` pending-detection, verdict computation, report
writing) deterministically and offline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import types
from pathlib import Path

import pytest

validate_phase5 = importlib.import_module("scripts.qa.validate_phase5")

pytestmark = pytest.mark.unit


def _proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> types.SimpleNamespace:
    """Stand-in for subprocess.CompletedProcess."""
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class TestCheckDvcReproIdempotent:
    def test_all_stages_noop_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = (
            "Stage 'download_coco' didn't change, skipping\n"
            "Stage 'merge_datasets' is cached\n"
            "Stage 'record_release' is frozen.\n"
        )
        monkeypatch.setattr(validate_phase5.subprocess, "run", lambda *a, **k: _proc(stdout=out))
        result = validate_phase5.check_dvc_repro_idempotent()
        assert result["passed"] is True
        assert result["pending_stages"] == []
        assert result["stage_status_lines"] == 3

    def test_pending_stage_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = (
            "Stage 'download_coco' didn't change, skipping\n" "Stage 'auto_annotate' will be run\n"
        )
        monkeypatch.setattr(validate_phase5.subprocess, "run", lambda *a, **k: _proc(stdout=out))
        result = validate_phase5.check_dvc_repro_idempotent()
        assert result["passed"] is False
        assert any("auto_annotate" in line for line in result["pending_stages"])

    def test_nonzero_exit_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_phase5.subprocess,
            "run",
            lambda *a, **k: _proc(stderr="dvc exploded", returncode=1),
        )
        result = validate_phase5.check_dvc_repro_idempotent()
        assert result["passed"] is False
        assert result["exit_code"] == 1


class TestRunFullSuite:
    def test_returncode_zero_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_phase5.subprocess,
            "run",
            lambda *a, **k: _proc(stdout="...\n123 passed in 4.5s", returncode=0),
        )
        result = validate_phase5.run_full_suite()
        assert result["passed"] is True
        assert "123 passed" in result["summary"]

    def test_returncode_nonzero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_phase5.subprocess,
            "run",
            lambda *a, **k: _proc(stdout="1 failed, 2 passed", returncode=1),
        )
        assert validate_phase5.run_full_suite()["passed"] is False


class TestBuildMarkdownSections:
    def test_sections_render_pass_verdict(self) -> None:
        report = {
            "verdict": "PASS",
            "full_suite": {"passed": True, "summary": "10 passed", "paths": ["tests/unit"]},
            "dvc_repro_idempotency": {
                "passed": True,
                "stage_status_lines": 3,
                "pending_stages": [],
            },
        }
        sections = validate_phase5.build_markdown_sections(report)
        assert sections[0]["heading"] == "Verdict"
        assert "PASS" in sections[0]["content"]
        assert len(sections) == 3


class TestRun:
    def test_skip_dvc_check_pass_writes_reports_and_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            validate_phase5.subprocess,
            "run",
            lambda *a, **k: _proc(stdout="42 passed in 1.0s", returncode=0),
        )
        args = argparse.Namespace(report_dir=tmp_path, skip_dvc_check=True)
        assert validate_phase5.run(args) == 0

        report = json.loads(
            (tmp_path / "phase5_validation_report.json").read_text(encoding="utf-8")
        )
        assert report["verdict"] == "PASS"
        assert report["dvc_repro_idempotency"]["skipped"] is True
        assert (tmp_path / "phase5_validation_report.md").exists()

    def test_suite_failure_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            validate_phase5.subprocess,
            "run",
            lambda *a, **k: _proc(stdout="1 failed", returncode=1),
        )
        args = argparse.Namespace(report_dir=tmp_path, skip_dvc_check=True)
        assert validate_phase5.run(args) == 1

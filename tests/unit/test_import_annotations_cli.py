"""Unit tests for scripts.dataset.09_import_annotations ``--compare`` CLI
wiring (final-audit Fix-7: the IAA library was well covered, but the
``_compare`` subcommand — report writing to ``iaa_*.json`` and verdict-based
exit codes — was never exercised through the script).

Everything is synthetic and offline: staged labels are written straight into
the staging layout that ``load_staged_labels`` reads, and a minimal capture
config + session manifest are placed under ``tmp_path``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from src.dataset.manifest import CaptureSessionManifest

import_annotations = importlib.import_module("scripts.dataset.09_import_annotations")

pytestmark = pytest.mark.unit

_SESSION = "h01_kitchen_s001"


def _write_data_yaml(path: Path) -> Path:
    path.write_text(
        json.dumps({"nc": 3, "names": {"0": "person", "1": "charger", "2": "wire"}}),
        encoding="utf-8",
    )
    return path


def _write_capture_config(tmp_path: Path) -> Path:
    captures_root = (tmp_path / "captures").as_posix()
    eval_root = (tmp_path / "eval").as_posix()
    staging_dir = (tmp_path / "staging").as_posix()
    path = tmp_path / "capture_config.yaml"
    path.write_text(
        "capture:\n"
        f"  captures_root: {captures_root}\n"
        f"  eval_root: {eval_root}\n"
        "annotation:\n"
        f"  staging_dir: {staging_dir}\n"
        "  iaa:\n"
        "    iou_threshold: 0.5\n"
        "    min_agreement: 0.7\n",
        encoding="utf-8",
    )
    return path


def _stage_labels(tmp_path: Path, annotator: str, lines: list[str]) -> None:
    dest = tmp_path / "staging" / _SESSION / annotator
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "img1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_session_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "captures" / "manifests" / f"{_SESSION}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    CaptureSessionManifest(
        source="custom_captures",
        session_id=_SESSION,
        house_id="h01",
        annotation_status="staged",
        image_hashes={"img1.jpg": "deadbeef"},
    ).save(manifest_path)


def _run_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, annotators: str | None = None
) -> int:
    config = _write_capture_config(tmp_path)
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    output = tmp_path / "reports"
    argv = [
        "prog",
        "--session",
        _SESSION,
        "--compare",
        "--config",
        str(config),
        "--data-config",
        str(data_yaml),
        "--output",
        str(output),
    ]
    if annotators is not None:
        argv += ["--annotators", annotators]
    monkeypatch.setattr(sys, "argv", argv)
    return import_annotations.main()


class TestCompareCli:
    def test_agreeing_annotators_pass_and_write_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stage_labels(tmp_path, "asha", ["0 0.5 0.5 0.2 0.2"])
        _stage_labels(tmp_path, "ben", ["0 0.5 0.5 0.2 0.2"])
        _write_session_manifest(tmp_path)

        rc = _run_compare(tmp_path, monkeypatch)

        assert rc == 0
        report = json.loads(
            (tmp_path / "reports" / f"iaa_{_SESSION}.json").read_text(encoding="utf-8")
        )
        assert report["verdict"] == "pass"
        assert (tmp_path / "reports" / f"iaa_{_SESSION}.md").exists()
        # The manifest's IAA field is updated by the CLI.
        manifest = json.loads(
            (tmp_path / "captures" / "manifests" / f"{_SESSION}.json").read_text(encoding="utf-8")
        )
        assert manifest["iaa_agreement"] == pytest.approx(1.0)

    def test_disagreeing_annotators_return_warning_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stage_labels(tmp_path, "asha", ["0 0.5 0.5 0.2 0.2"])
        _stage_labels(tmp_path, "ben", ["1 0.1 0.1 0.2 0.2"])
        _write_session_manifest(tmp_path)

        rc = _run_compare(tmp_path, monkeypatch)

        # Exit 2 == below the agreement gate (not a crash).
        assert rc == 2
        report = json.loads(
            (tmp_path / "reports" / f"iaa_{_SESSION}.json").read_text(encoding="utf-8")
        )
        assert report["verdict"] == "fail"

    def test_fewer_than_two_annotators_is_a_hard_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stage_labels(tmp_path, "asha", ["0 0.5 0.5 0.2 0.2"])
        _write_session_manifest(tmp_path)

        assert _run_compare(tmp_path, monkeypatch) == 1

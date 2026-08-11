"""
Unit tests for scripts/scenarios/32_ingest_scenario_clips.py.

The ingest path itself needs ffmpeg and real footage, neither of which exists in
CI. What is testable — and what actually matters — is that the CLI **refuses**:
without a toolchain, without consent, with a malformed clip id. A privacy gate
that can only be exercised on the collection machine is a gate nobody has
checked, so each refusal is asserted here.

`--verify-all` is the cmd of the frozen ``ingest_scenario_clips`` DVC stage, so
its exit codes are a contract.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.scenario_engine.clips import ffmpeg_available

_CLI_PATH = Path("scripts/scenarios/32_ingest_scenario_clips.py")


def _load_cli() -> Any:
    """Import the numbered CLI module, whose name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location("ingest_scenario_clips_cli", _CLI_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> Path:
    payload = {
        "clips": {
            "inbox_dir": str(tmp_path / "inbox"),
            "clips_root": str(tmp_path / "clips"),
            "clip_id_pattern": r"^h\d{2}_[a-z_]+_s\d{3}_c\d{3}$",
            "consent_scope": "scenario-video",
            "min_negative_fraction": 0.30,
            "video": {"allowed_extensions": [".mp4"], "min_fps": 15},
        }
    }
    path = tmp_path / "capture_config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _artifact(tmp_path: Path, rule_hash: str = "sha256:deadbeef") -> Path:
    path = tmp_path / "scenarios.compiled.json"
    path.write_text(
        json.dumps({"scenarios": [{"scenario_id": "SC-KIT-001", "rule_hash": rule_hash}]}),
        encoding="utf-8",
    )
    return path


class TestInit:
    @pytest.mark.unit
    def test_init_creates_the_tree_and_is_idempotent(self, tmp_path: Path) -> None:
        cli = _load_cli()
        argv = ["--init", "--config", str(_config(tmp_path))]
        assert cli.main(argv) == 0
        assert cli.main(argv) == 0
        assert (tmp_path / "clips" / "video").is_dir()
        assert (tmp_path / "clips" / "manifests").is_dir()
        assert (tmp_path / "clips" / "README.md").exists()


class TestVerifyAll:
    """The cmd of the frozen DVC stage."""

    @pytest.mark.unit
    def test_empty_tree_is_success_not_failure(self, tmp_path: Path) -> None:
        """The expected state before the first capture session."""
        cli = _load_cli()
        assert cli.main(["--verify-all", "--config", str(_config(tmp_path))]) == 0

    @pytest.mark.unit
    def test_manifest_without_its_video_fails(self, tmp_path: Path) -> None:
        cli = _load_cli()
        config = _config(tmp_path)
        assert cli.main(["--init", "--config", str(config)]) == 0

        manifest = {
            "clip_id": "h01_kitchen_s001_c001",
            "session_id": "h01_kitchen_s001",
            "house_id": "h01",
            "room": "kitchen",
            "lighting": "evening",
            "polarity": "positive",
            "scenario_id": "SC-KIT-001",
            "consent_reference": "CONSENT-h01-2026-002",
            "sha256": "abc123",
            "duration_s": 45.0,
            "fps": 30.0,
            "metadata_stripped": True,
            "rule_hash_at_label_time": "sha256:deadbeef",
            "expected": {
                "fires": True,
                "scenario_id": "SC-KIT-001",
                "first_alert_within_s": 30.0,
            },
        }
        path = tmp_path / "clips" / "manifests" / "h01_kitchen_s001_c001.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")

        assert (
            cli.main(
                ["--verify-all", "--config", str(config), "--artifact", str(_artifact(tmp_path))]
            )
            == 1
        )


class TestIngestRefusals:
    @pytest.mark.unit
    def test_missing_arguments_are_named(self, tmp_path: Path) -> None:
        cli = _load_cli()
        assert cli.main(["--config", str(_config(tmp_path)), "--clip-id", "h01_x_s001_c001"]) == 1

    @pytest.mark.unit
    def test_reingesting_an_existing_clip_id_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A clip id is what a test result refers to; replacing the footage behind
        one silently changes what a green suite means."""
        cli = _load_cli()
        config = _config(tmp_path)
        assert cli.main(["--init", "--config", str(config)]) == 0
        (tmp_path / "clips" / "manifests" / "h01_kitchen_s001_c001.json").write_text(
            "{}", encoding="utf-8"
        )
        with caplog.at_level("ERROR"):
            exit_code = cli.main(
                [
                    "--config",
                    str(config),
                    "--clip-id",
                    "h01_kitchen_s001_c001",
                    "--source",
                    "whatever.mp4",
                    "--scenario-id",
                    "SC-KIT-001",
                    "--consent-ref",
                    "CONSENT-h01-2026-002",
                    "--expect",
                    "no-alert",
                    "--negative-kind",
                    "absence",
                ]
            )
        assert exit_code == 1
        assert "already ingested" in caplog.text

    @pytest.mark.unit
    @pytest.mark.skipif(ffmpeg_available(), reason="ffmpeg is installed on this machine")
    def test_ingest_refuses_without_ffmpeg(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The clip is never copied anywhere unsanitised."""
        cli = _load_cli()
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"not-a-real-mp4")
        with caplog.at_level("ERROR"):
            exit_code = cli.main(
                [
                    "--config",
                    str(_config(tmp_path)),
                    "--clip-id",
                    "h01_kitchen_s001_c001",
                    "--source",
                    str(source),
                    "--scenario-id",
                    "SC-KIT-001",
                    "--consent-ref",
                    "CONSENT-h01-2026-002",
                    "--expect",
                    "fires",
                    "--first-alert-within",
                    "30",
                ]
            )
        assert exit_code == 1
        # It refused for the toolchain reason, not incidentally on a missing arg.
        assert "unsanitised" in caplog.text
        assert not (tmp_path / "clips" / "video" / "h01_kitchen_s001_c001.mp4").exists()


class TestRuleHashLookup:
    @pytest.mark.unit
    def test_absent_artifact_degrades_rather_than_crashing(self, tmp_path: Path) -> None:
        cli = _load_cli()
        assert cli._rule_hashes(tmp_path / "absent.json") == {}

    @pytest.mark.unit
    def test_rejected_scenarios_are_included(self, tmp_path: Path) -> None:
        """A clip may exist for a scenario that has since been rejected; it must
        still be detectable as stale rather than reported as unknown."""
        cli = _load_cli()
        path = tmp_path / "artifact.json"
        path.write_text(
            json.dumps(
                {
                    "scenarios": [{"scenario_id": "SC-KIT-001", "rule_hash": "sha256:a"}],
                    "rejected": [{"scenario_id": "SC-SAF-001", "rule_hash": "sha256:b"}],
                }
            ),
            encoding="utf-8",
        )
        assert cli._rule_hashes(path) == {"SC-KIT-001": "sha256:a", "SC-SAF-001": "sha256:b"}

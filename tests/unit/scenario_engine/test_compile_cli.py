"""
Unit tests for scripts/scenarios/30_compile_scenarios.py.

The `--check` mode is what CI runs, so its exit codes are a contract: 0 current,
1 validation failure, 2 drift. A drifted or hand-edited artifact must fail there
rather than in production.

Also asserts the committed artifact is current, so a scenario edited without
recompiling fails the suite rather than shipping a stale rule set.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

_CLI_PATH = Path("scripts/scenarios/30_compile_scenarios.py")


def _load_cli() -> Any:
    """Import the numbered CLI module, whose name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location("compile_scenarios_cli", _CLI_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO: dict[str, Any] = {
    "scenario_id": "SC-KIT-001",
    "schema_version": 1,
    "scenario_name": "Cooking left unattended",
    "category": "kitchen",
    "trigger": {
        "all": [
            {"op": "detected", "args": {"class_name": "stove"}},
            {"op": "absent_for", "args": {"class_name": "person", "seconds": 120}},
        ]
    },
    "risk_level": "CRITICAL",
    "detectability": "direct",
    "claim_class": "observation",
    "status": "active",
    "caregiver_channel": "push",
    "next_best_action": "Ask someone to check the kitchen",
    "messages": {"en": "The stove is visible and nobody has been nearby for a while."},
    "reviewed_by": "clinical-review",
    "reviewed_on": "2026-08-07",
    "evidence": [
        {"source_id": "NFPA-CookingSafety", "strength": "guideline", "supports": "hazard"}
    ],
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An isolated config + scenario dir, never the real repo files."""
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "SC-KIT-001.yaml").write_text(
        yaml.safe_dump(_SCENARIO, allow_unicode=True), encoding="utf-8"
    )
    build = tmp_path / "build"
    config = tmp_path / "scenario_engine.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "compile": {
                    "scenario_dir": str(scenario_dir),
                    "scenarios_version": "scenarios-v0.1.0",
                    "outputs": {
                        "artifact": str(build / "scenarios.compiled.json"),
                        "view_csv": str(build / "scenarios.view.csv"),
                        "editable_csv": str(build / "scenarios.editable.csv"),
                    },
                    "capability": {
                        "data_config": "configs/data.yaml",
                        "feature_flags": "configs/feature_flags.yaml",
                        "wet_floor_decision": str(tmp_path / "no_decision.json"),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config


class TestCli:
    @pytest.mark.unit
    def test_compiles_and_writes_all_three_outputs(self, workspace: Path) -> None:
        cli = _load_cli()
        assert cli.run(workspace, check_only=False) == 0
        build = workspace.parent / "build"
        assert (build / "scenarios.compiled.json").exists()
        assert (build / "scenarios.view.csv").exists()
        assert (build / "scenarios.editable.csv").exists()

    @pytest.mark.unit
    def test_check_passes_on_a_current_artifact(self, workspace: Path) -> None:
        cli = _load_cli()
        cli.run(workspace, check_only=False)
        assert cli.run(workspace, check_only=True) == 0

    @pytest.mark.unit
    def test_check_reports_drift_as_exit_2(self, workspace: Path) -> None:
        cli = _load_cli()
        cli.run(workspace, check_only=False)
        artifact_path = workspace.parent / "build" / "scenarios.compiled.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["scenarios"][0]["risk_level"] = "INFO"
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        assert cli.run(workspace, check_only=True) == 2

    @pytest.mark.unit
    def test_check_without_a_committed_artifact_is_exit_2(self, workspace: Path) -> None:
        assert _load_cli().run(workspace, check_only=True) == 2

    @pytest.mark.unit
    def test_validation_failure_is_exit_1(self, workspace: Path, tmp_path: Path) -> None:
        """A static-only trigger — the cooking metronome — must not compile."""
        bad = dict(_SCENARIO)
        bad["trigger"] = {
            "all": [
                {"op": "detected", "args": {"class_name": "knife"}},
                {"op": "detected", "args": {"class_name": "person"}},
            ]
        }
        (tmp_path / "scenarios" / "SC-KIT-001.yaml").write_text(
            yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8"
        )
        assert _load_cli().run(workspace, check_only=False) == 1

    @pytest.mark.unit
    def test_compilation_is_byte_reproducible(self, workspace: Path) -> None:
        cli = _load_cli()
        cli.run(workspace, check_only=False)
        first = (workspace.parent / "build" / "scenarios.compiled.json").read_bytes()
        cli.run(workspace, check_only=False)
        second = (workspace.parent / "build" / "scenarios.compiled.json").read_bytes()
        assert first == second


class TestCommittedArtifact:
    @pytest.mark.unit
    def test_repo_artifact_is_current(self) -> None:
        """A scenario edited without recompiling fails here, not in production."""
        assert _load_cli().run(Path("configs/scenario_engine.yaml"), check_only=True) == 0

    @pytest.mark.unit
    def test_repo_artifact_verifies(self) -> None:
        from src.scenario_engine.compile import verify_artifact

        artifact = json.loads(
            Path("data/scenario_engine/build/scenarios.compiled.json").read_text(encoding="utf-8")
        )
        verify_artifact(artifact)

    @pytest.mark.unit
    def test_passport_is_recorded_as_unusable(self) -> None:
        """The live instance of the blindness the capability map exists to fix."""
        artifact = json.loads(
            Path("data/scenario_engine/build/scenarios.compiled.json").read_text(encoding="utf-8")
        )
        assert artifact["classes"]["passport"]["enabled"] is False

"""
Unit tests for scripts.qa.model_landing_check.

M9's acceptance is "no further code changes are required when the model lands".
`model_landing_check.py` is the falsifiable form of that sentence, so these
tests are mostly about the ways it must be *able to fail* — a landing check that
cannot fail is worse than none, because it certifies the thing it never tested.

The distinction between `fail` and `blocked` is load-bearing and is asserted
directly: a pending clinical review must never read as a broken pipeline, and a
broken pipeline must never hide behind a pending clinical review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.qa import model_landing_check as mlc
from src.config.config_loader import SystemConfig
from src.pipeline.detector import YOLODetector

_NAMES = {0: "person", 1: "face", 5: "knife"}


class _FakeModel:
    def __init__(self, names: dict[int, str]) -> None:
        self.names = names


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.write_bytes(b"not-a-real-model")
    return path


def _config(names: dict[int, str]) -> SystemConfig:
    return SystemConfig(class_names=dict(names))


class TestL1Weights:
    @pytest.mark.unit
    def test_missing_weights_fail(self, tmp_path: Path) -> None:
        check = mlc.check_l1_weights_present(tmp_path / "absent.pt")
        assert check.status == mlc.FAIL

    @pytest.mark.unit
    def test_present_weights_pass(self, weights: Path) -> None:
        check = mlc.check_l1_weights_present(weights)
        assert check.status == mlc.PASS
        assert check.details["size_mb"] is not None


class TestL2Taxonomy:
    @pytest.mark.unit
    def test_no_taxonomy_is_a_failure_not_a_skip(self) -> None:
        """Everything downstream is vacuous without it."""
        assert mlc.check_l2_taxonomy_declared(_config({})).status == mlc.FAIL

    @pytest.mark.unit
    def test_the_repository_taxonomy_passes(self) -> None:
        check = mlc.check_l2_taxonomy_declared(SystemConfig.load())
        assert check.status == mlc.PASS
        assert check.details["class_count"] == 23


class TestL3WeightsMatchTaxonomy:
    @pytest.mark.unit
    def test_matching_weights_pass(self, weights: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(YOLODetector, "_load_model", lambda self: _FakeModel(_NAMES))
        assert mlc.check_l3_weights_match_taxonomy(weights, _config(_NAMES)).status == mlc.PASS

    @pytest.mark.unit
    def test_a_different_class_list_fails_with_the_difference_named(
        self, weights: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(YOLODetector, "_load_model", lambda self: _FakeModel(_NAMES))
        check = mlc.check_l3_weights_match_taxonomy(
            weights, _config({**_NAMES, 5: "cutting_knife"})
        )
        assert check.status == mlc.FAIL
        assert "cutting_knife" in check.details["error"]

    @pytest.mark.unit
    def test_it_skips_rather_than_passes_when_prerequisites_are_missing(
        self, tmp_path: Path
    ) -> None:
        check = mlc.check_l3_weights_match_taxonomy(tmp_path / "absent.pt", _config(_NAMES))
        assert check.status == mlc.SKIPPED


class TestL5EngineCanStart:
    @pytest.mark.unit
    def test_all_draft_reports_blocked_not_failed(self) -> None:
        """The repository's real state: engineering is done, review is not."""
        scenarios = _load_repository_scenarios()
        check = mlc.check_l5_engine_can_start(scenarios, SystemConfig.load())
        assert check.status == mlc.BLOCKED
        assert "Clinical review" in check.details["blocked_on"]

    @pytest.mark.unit
    def test_active_scenarios_pass(self, active_scenarios: Path) -> None:
        from src.scenario_engine.compile import load_scenario_files

        check = mlc.check_l5_engine_can_start(
            load_scenario_files(active_scenarios), SystemConfig.load()
        )
        assert check.status == mlc.PASS
        assert check.details["runnable"] > 0

    @pytest.mark.unit
    def test_active_but_every_flag_off_is_a_failure(self, active_scenarios: Path) -> None:
        """That is a configuration mistake, not a pending human step."""
        from src.scenario_engine.compile import load_scenario_files

        scenarios = load_scenario_files(active_scenarios)
        config = SystemConfig(rules={s.scenario_id: False for s in scenarios})
        assert mlc.check_l5_engine_can_start(scenarios, config).status == mlc.FAIL


class TestL4ScenarioClasses:
    @pytest.mark.unit
    def test_the_repository_scenarios_only_use_usable_classes(self) -> None:
        check = mlc.check_l4_scenario_classes_usable(_load_repository_scenarios())
        assert check.status == mlc.PASS, check.details
        assert check.details["taxonomy_fingerprint"].startswith("sha256:")


class TestVerdict:
    @pytest.mark.unit
    def test_blocked_exits_zero_and_failed_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pending human step is not a build break; a broken build is."""
        monkeypatch.setattr(mlc, "REPORT_PATH", tmp_path / "report.json")

        blocked_exit = mlc.run(model_path=Path("models/does-not-exist.pt"), skip_assemble=True)
        assert blocked_exit == 1  # L1 fails: no weights

        def _all_blocked(*args: Any, **kwargs: Any) -> mlc.Check:
            return mlc.Check("L1", "stub", mlc.PASS)

        monkeypatch.setattr(mlc, "check_l1_weights_present", _all_blocked)
        monkeypatch.setattr(mlc, "check_l3_weights_match_taxonomy", _all_blocked)
        assert mlc.run(model_path=tmp_path / "any.pt", skip_assemble=True) == 0


def _load_repository_scenarios() -> list[Any]:
    from src.scenario_engine.compile import load_scenario_files

    return load_scenario_files(mlc.SCENARIO_DIR)

"""
Unit tests for src.pipeline.rule_engine.

The rule engine had **zero** test coverage before Phase 6 (a `risk_rules_path`
fixture existed at tests/integration/conftest.py but nothing used it). These
tests establish the behaviour baseline that milestone M8's supersession must
preserve, and pin the fail-closed contract that replaced the previous
silent-`False` / silent-`KeyError` behaviour.

See docs/08_scenario_engineering/architecture_review.md §3 (D6, D7).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipeline import BoundingBox, Detection, Severity
from src.pipeline.event_memory import EventMemory
from src.pipeline.rule_engine import RuleConfigError, RuleEngine

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_rules(tmp_path: Path, rules: list[dict]) -> str:
    path = tmp_path / "risk_rules.yaml"
    path.write_text(yaml.safe_dump({"rules": rules}, allow_unicode=True), encoding="utf-8")
    return str(path)


def _rule(**overrides: object) -> dict:
    base: dict = {
        "id": "test_rule",
        "condition": "detected(knife)",
        "severity": "HIGH",
        "cooldown_seconds": 60,
        "message_en": "A knife is visible.",
    }
    base.update(overrides)
    return base


def _detection(class_name: str, class_id: int = 0) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.2),
        frame_id=7,
        timestamp_ms=1000.0,
    )


@pytest.fixture
def memory() -> EventMemory:
    return EventMemory(window_size=150)


# ─── Loading ──────────────────────────────────────────────────────────────────


class TestLoading:
    @pytest.mark.unit
    def test_loads_valid_rules(self, tmp_path: Path) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule()]))
        assert len(engine._rules) == 1
        assert engine._rules[0].severity is Severity.HIGH

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuleConfigError, match="not found"):
            RuleEngine(str(tmp_path / "nope.yaml"))

    @pytest.mark.unit
    def test_empty_rule_set_is_refused(self, tmp_path: Path) -> None:
        """A safety pipeline loading zero rules used to log INFO and continue."""
        with pytest.raises(RuleConfigError, match="no rules"):
            RuleEngine(_write_rules(tmp_path, []))

    @pytest.mark.unit
    def test_missing_rules_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "risk_rules.yaml"
        path.write_text("something_else: []\n", encoding="utf-8")
        with pytest.raises(RuleConfigError, match="rules"):
            RuleEngine(str(path))

    @pytest.mark.unit
    def test_invalid_severity_fails_at_load_not_at_evaluate(self, tmp_path: Path) -> None:
        """D7: `Severity[...]` raised KeyError mid-frame, zeroing every alert."""
        with pytest.raises(RuleConfigError) as excinfo:
            RuleEngine(_write_rules(tmp_path, [_rule(severity="URGENT")]))
        assert "URGENT" in str(excinfo.value)
        assert "CRITICAL" in str(excinfo.value)

    @pytest.mark.unit
    def test_malformed_condition_fails_at_load(self, tmp_path: Path) -> None:
        with pytest.raises(RuleConfigError, match="test_rule"):
            RuleEngine(_write_rules(tmp_path, [_rule(condition="detected(knife) AND")]))

    @pytest.mark.unit
    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(RuleConfigError, match="duplicate"):
            RuleEngine(_write_rules(tmp_path, [_rule(), _rule()]))

    @pytest.mark.unit
    def test_missing_message_rejected(self, tmp_path: Path) -> None:
        rule = _rule()
        del rule["message_en"]
        with pytest.raises(RuleConfigError, match="message_en"):
            RuleEngine(_write_rules(tmp_path, [rule]))

    @pytest.mark.unit
    def test_unknown_class_rejected_when_taxonomy_supplied(self, tmp_path: Path) -> None:
        """`detected(knive)` used to be a rule that simply never fired."""
        path = _write_rules(tmp_path, [_rule(condition="detected(knive)")])
        with pytest.raises(RuleConfigError, match="knive"):
            RuleEngine(path, valid_class_names={"knife", "person"})

    @pytest.mark.unit
    def test_known_classes_accepted_when_taxonomy_supplied(self, tmp_path: Path) -> None:
        engine = RuleEngine(
            _write_rules(tmp_path, [_rule()]), valid_class_names={"knife", "person"}
        )
        assert len(engine._rules) == 1


# ─── Evaluation ───────────────────────────────────────────────────────────────


class TestEvaluation:
    @pytest.mark.unit
    def test_fires_when_condition_met(self, tmp_path: Path, memory: EventMemory) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule()]))
        alerts = engine.evaluate([_detection("knife")], memory)
        assert len(alerts) == 1
        assert alerts[0].rule_id == "test_rule"
        assert alerts[0].severity is Severity.HIGH
        assert alerts[0].message == "A knife is visible."
        assert alerts[0].frame_id == 7

    @pytest.mark.unit
    def test_does_not_fire_when_condition_unmet(self, tmp_path: Path, memory: EventMemory) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule()]))
        assert engine.evaluate([_detection("person")], memory) == []

    @pytest.mark.unit
    def test_cooldown_suppresses_immediate_refire(
        self, tmp_path: Path, memory: EventMemory
    ) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule(cooldown_seconds=3600)]))
        detections = [_detection("knife")]
        assert len(engine.evaluate(detections, memory)) == 1
        assert engine.evaluate(detections, memory) == []

    @pytest.mark.unit
    def test_first_fire_is_not_suppressed_on_a_freshly_booted_device(
        self, tmp_path: Path, memory: EventMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A never-fired rule must fire immediately, whatever the clock reads.

        `time.monotonic()` counts from boot, so a 0.0 "last fired" sentinel meant
        `now - 0.0 < cooldown` was TRUE on a freshly-powered device -- every rule
        suppressed for its full cooldown after a power cut. This asserts the
        boot case directly rather than depending on how long the test machine
        happens to have been up.
        """
        monkeypatch.setattr("src.pipeline.rule_engine.time.monotonic", lambda: 3.0)
        engine = RuleEngine(_write_rules(tmp_path, [_rule(cooldown_seconds=1800)]))
        alerts = engine.evaluate([_detection("knife")], memory)
        assert len(alerts) == 1, "a never-fired rule was suppressed just after boot"

    @pytest.mark.unit
    def test_zero_cooldown_allows_refire(self, tmp_path: Path, memory: EventMemory) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule(cooldown_seconds=0)]))
        detections = [_detection("knife")]
        assert len(engine.evaluate(detections, memory)) == 1
        assert len(engine.evaluate(detections, memory)) == 1

    @pytest.mark.unit
    def test_alerts_sorted_by_descending_severity(
        self, tmp_path: Path, memory: EventMemory
    ) -> None:
        """rule_engine.md:37-43 documented this ordering; it was never implemented."""
        rules = [
            _rule(id="info_rule", severity="INFO", cooldown_seconds=0),
            _rule(id="critical_rule", severity="CRITICAL", cooldown_seconds=0),
            _rule(id="medium_rule", severity="MEDIUM", cooldown_seconds=0),
        ]
        engine = RuleEngine(_write_rules(tmp_path, rules))
        alerts = engine.evaluate([_detection("knife")], memory)
        assert [a.severity for a in alerts] == [
            Severity.CRITICAL,
            Severity.MEDIUM,
            Severity.INFO,
        ]

    @pytest.mark.unit
    def test_any_of_conjunct_is_honoured(self, tmp_path: Path, memory: EventMemory) -> None:
        """D6: the old evaluator dropped everything after an any_of(...)."""
        rule = _rule(condition="any_of([medicine_strip, medicine_bottle]) AND detected(person)")
        engine = RuleEngine(_write_rules(tmp_path, [rule]))
        assert engine.evaluate([_detection("medicine_strip")], memory) == []
        alerts = engine.evaluate([_detection("medicine_strip"), _detection("person")], memory)
        assert len(alerts) == 1

    @pytest.mark.unit
    def test_explanation_records_condition_and_classes(
        self, tmp_path: Path, memory: EventMemory
    ) -> None:
        engine = RuleEngine(_write_rules(tmp_path, [_rule()]))
        alert = engine.evaluate([_detection("knife"), _detection("person")], memory)[0]
        assert alert.explanation["condition"] == "detected(knife)"
        assert alert.explanation["detected_classes"] == ["knife", "person"]
        assert alert.explanation["vlm_available"] is False


class TestTemporalEvaluation:
    @pytest.mark.unit
    def test_absent_for_uses_measured_fps(self, tmp_path: Path, memory: EventMemory) -> None:
        """A throttled device must not silently rescale temporal thresholds."""
        rule = _rule(id="stove", condition="detected(stove) AND absent_for(person, 30)")
        engine = RuleEngine(_write_rules(tmp_path, [rule]), fps=15.0)

        memory.update([_detection("person", class_id=0)])
        for _ in range(120):
            memory.update([_detection("stove", class_id=6)])

        # 120 frames since person. At 15 FPS that is 8s — under the 30s threshold.
        assert engine.evaluate([_detection("stove", class_id=6)], memory) == []
        # At a throttled 2 FPS the same 120 frames is 60s — over the threshold.
        assert len(engine.evaluate([_detection("stove", class_id=6)], memory, None, 2.0)) == 1


# ─── Hot reload ───────────────────────────────────────────────────────────────


class TestReload:
    @pytest.mark.unit
    def test_reload_picks_up_changes(self, tmp_path: Path, memory: EventMemory) -> None:
        path = _write_rules(tmp_path, [_rule()])
        engine = RuleEngine(path)
        _write_rules(tmp_path, [_rule(id="other", condition="detected(stove)")])
        engine.reload_rules()
        assert [r.rule_id for r in engine._rules] == ["other"]

    @pytest.mark.unit
    def test_bad_reload_keeps_previous_rules_active(
        self, tmp_path: Path, memory: EventMemory
    ) -> None:
        """A bad edit must never disarm the system."""
        path = _write_rules(tmp_path, [_rule()])
        engine = RuleEngine(path)
        Path(path).write_text("rules:\n  - id: broken\n    severity: NOPE\n", encoding="utf-8")

        with pytest.raises(RuleConfigError):
            engine.reload_rules()

        assert [r.rule_id for r in engine._rules] == ["test_rule"]
        assert len(engine.evaluate([_detection("knife")], memory)) == 1

    @pytest.mark.unit
    def test_reload_prunes_cooldowns_for_removed_rules(self, tmp_path: Path) -> None:
        """Stale cooldown keys used to leak forever across reloads."""
        path = _write_rules(tmp_path, [_rule()])
        engine = RuleEngine(path)
        engine._cooldowns["test_rule"] = 123.0
        engine._cooldowns["deleted_rule"] = 456.0

        _write_rules(tmp_path, [_rule()])
        engine.reload_rules()

        assert "test_rule" in engine._cooldowns
        assert "deleted_rule" not in engine._cooldowns


# ─── The shipped rule file ────────────────────────────────────────────────────


class TestShippedRules:
    @pytest.mark.unit
    def test_the_retired_legacy_rule_file_still_loads(self) -> None:
        """The committed rule set must satisfy every load-time invariant."""
        engine = RuleEngine("tests/fixtures/legacy_risk_rules.yaml")
        assert len(engine._rules) >= 1
        assert all(r.message_en for r in engine._rules)

    @pytest.mark.unit
    def test_shipped_rules_reference_only_taxonomy_classes(self) -> None:
        """Guards against a rule silently never firing because of a class typo."""
        data = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
        names = frozenset(data["names"].values())
        RuleEngine("tests/fixtures/legacy_risk_rules.yaml", valid_class_names=names)

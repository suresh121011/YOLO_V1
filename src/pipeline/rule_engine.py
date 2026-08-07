"""
YAML-Driven Rule Engine
========================
Evaluates safety rules against YOLO detections, Event Memory state,
and optional SmolVLM2 context. Single decision point in the pipeline.

Rules are defined in configs/risk_rules.yaml and can be hot-reloaded
at runtime without restarting the pipeline.

Conditions are parsed into an AST **once at load time** by
``src/pipeline/condition_parser.py``. A malformed condition, an unknown
predicate, or an invalid severity is therefore a loud load-time failure rather
than a rule that silently never fires — see
``docs/08_scenario_engineering/architecture_review.md`` §3 (defects D6, D7).

This engine and its string DSL are retired at milestone M8 in favour of the
structured predicate trees compiled by ``src/scenario_engine``
(ADR-P6-03). Until then it must be kept *correct*, but must not be *extended*
with new predicates.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import Alert, Detection, SceneContext, Severity
from .condition_parser import (
    ConditionSyntaxError,
    Node,
    evaluate_node,
    parse_condition,
    referenced_classes,
)
from .event_memory import EventMemory

logger = logging.getLogger(__name__)

#: Severity names accepted in risk_rules.yaml, validated at load time.
VALID_SEVERITIES: frozenset[str] = frozenset(s.name for s in Severity)


class RuleConfigError(ValueError):
    """Raised when configs/risk_rules.yaml cannot be loaded into valid rules.

    Fail-closed by design: a safety component that quietly loads zero rules is
    indistinguishable from one that is working perfectly and seeing nothing.
    """


@dataclass(frozen=True)
class CompiledRule:
    """A rule with its condition already parsed."""

    rule_id: str
    condition_text: str
    condition: Node
    severity: Severity
    #: Whole seconds — matches ``Alert.cooldown_seconds``, which is a LOCKED
    #: contract field typed ``int``. Sub-second cooldowns are not expressible.
    cooldown_seconds: int
    message_en: str
    message_hi: str | None


class RuleEngine:
    """YAML-driven safety rule evaluation engine.

    Evaluates all enabled rules against:
      - current YOLO detections
      - Event Memory (temporal state)
      - SmolVLM2 scene context (optional)

    Manages per-rule cooldowns internally.
    Thread-safe via RLock.
    """

    def __init__(
        self,
        rules_path: str,
        fps: float = 15.0,
        valid_class_names: frozenset[str] | set[str] | None = None,
        rule_enabled: Callable[[str], bool] | None = None,
    ) -> None:
        """
        Args:
            rules_path: Path to risk_rules.yaml.
            fps: Expected pipeline FPS (used for time-based rule conditions).
            valid_class_names: Optional taxonomy. When supplied, a condition
                referencing an unknown class fails at load rather than never
                firing.
            rule_enabled: Optional predicate deciding whether a rule id is active,
                normally ``SystemConfig.is_rule_enabled``. Disabled rules are
                still fully validated, so re-enabling one later cannot surface a
                latent syntax error.

        Raises:
            RuleConfigError: If the file is missing, malformed, or contains no rules.
        """
        self._rules_path = Path(rules_path)
        self._lock = threading.RLock()
        self._cooldowns: dict[str, float] = {}
        self._rules: list[CompiledRule] = []
        self._fps = fps
        self._valid_class_names = frozenset(valid_class_names) if valid_class_names else None
        self._rule_enabled = rule_enabled
        self._load_rules()

    # ─────────────────────────────────────────
    # Rule management
    # ─────────────────────────────────────────

    def _compile(self, raw_rules: list[object]) -> list[CompiledRule]:
        """Validate and parse raw YAML rule mappings. Raises on the first defect."""
        compiled: list[CompiledRule] = []
        seen: set[str] = set()

        for index, raw in enumerate(raw_rules):
            where = f"{self._rules_path} rule #{index}"
            if not isinstance(raw, dict):
                raise RuleConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

            rule_id = str(raw.get("id", "")).strip()
            if not rule_id:
                raise RuleConfigError(f"{where}: missing required field 'id'")
            if rule_id in seen:
                raise RuleConfigError(f"{where}: duplicate rule id {rule_id!r}")
            seen.add(rule_id)

            severity_name = str(raw.get("severity", "INFO"))
            if severity_name not in VALID_SEVERITIES:
                valid = ", ".join(sorted(VALID_SEVERITIES))
                raise RuleConfigError(
                    f"{where} ({rule_id}): invalid severity {severity_name!r}. Valid: {valid}"
                )

            condition_text = str(raw.get("condition", "")).strip()
            try:
                condition = parse_condition(condition_text)
            except ConditionSyntaxError as exc:
                raise RuleConfigError(f"{where} ({rule_id}): {exc}") from exc

            if self._valid_class_names is not None:
                unknown = sorted(referenced_classes(condition) - self._valid_class_names)
                if unknown:
                    raise RuleConfigError(
                        f"{where} ({rule_id}): condition references unknown "
                        f"class(es) {unknown}; check configs/data.yaml"
                    )

            message_en = str(raw.get("message_en", "")).strip()
            if not message_en:
                raise RuleConfigError(f"{where} ({rule_id}): missing required field 'message_en'")

            raw_cooldown = raw.get("cooldown_seconds", 60)
            try:
                cooldown_seconds = int(raw_cooldown)
            except (TypeError, ValueError) as exc:
                raise RuleConfigError(
                    f"{where} ({rule_id}): cooldown_seconds must be a whole "
                    f"number of seconds, got {raw_cooldown!r}"
                ) from exc
            if cooldown_seconds < 0:
                raise RuleConfigError(
                    f"{where} ({rule_id}): cooldown_seconds must be >= 0, got {cooldown_seconds}"
                )

            message_hi_raw = raw.get("message_hi")
            compiled.append(
                CompiledRule(
                    rule_id=rule_id,
                    condition_text=condition_text,
                    condition=condition,
                    severity=Severity[severity_name],
                    cooldown_seconds=cooldown_seconds,
                    message_en=message_en,
                    message_hi=str(message_hi_raw) if message_hi_raw else None,
                )
            )

        if not compiled:
            raise RuleConfigError(
                f"{self._rules_path} defines no rules. Refusing to run a safety "
                f"pipeline with an empty rule set."
            )
        return compiled

    def _load_rules(self) -> None:
        """Load, validate, and parse the rule file, then swap it in atomically."""
        if not self._rules_path.exists():
            raise RuleConfigError(f"Rules file not found: {self._rules_path}")

        try:
            data = yaml.safe_load(self._rules_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RuleConfigError(f"{self._rules_path} is not valid YAML: {exc}") from exc

        if not isinstance(data, dict) or "rules" not in data:
            raise RuleConfigError(f"{self._rules_path} must contain a top-level 'rules:' key")
        raw_rules = data["rules"]
        if not isinstance(raw_rules, list):
            raise RuleConfigError(f"{self._rules_path}: 'rules' must be a list")

        # Compile fully before mutating state, so a bad reload leaves the
        # previously-loaded (working) rule set active.
        compiled = self._compile(raw_rules)

        if self._rule_enabled is not None:
            active = [rule for rule in compiled if self._rule_enabled(rule.rule_id)]
            disabled = len(compiled) - len(active)
            if disabled:
                logger.info(f"{disabled} rule(s) disabled via feature flags")
            if not active:
                logger.warning(
                    f"Every rule in {self._rules_path} is disabled via feature flags — "
                    f"the pipeline will raise no alerts."
                )
            compiled = active

        with self._lock:
            self._rules = compiled
            live_ids = {rule.rule_id for rule in compiled}
            # Drop cooldown entries for rules that no longer exist, otherwise the
            # dict grows without bound across reloads and renamed ids can never
            # cool down again.
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if k in live_ids}
            logger.info(f"Loaded {len(compiled)} rules from {self._rules_path}")

    def reload_rules(self) -> None:
        """Hot-reload rules from YAML without restarting the pipeline.

        Raises:
            RuleConfigError: If the new file is invalid. The previously loaded
                rule set stays active — a bad edit must never disarm the system.
        """
        self._load_rules()
        logger.info("Rules hot-reloaded")

    # ─────────────────────────────────────────
    # Core evaluation
    # ─────────────────────────────────────────

    def evaluate(
        self,
        detections: list[Detection],
        memory: EventMemory,
        context: SceneContext | None = None,
        current_fps: float | None = None,
    ) -> list[Alert]:
        """Evaluate all rules against the current pipeline state.

        Args:
            detections: YOLO detections for the current frame.
            memory: Event Memory with temporal state.
            context: Optional SmolVLM2 scene context.
            current_fps: Measured FPS for time calculations. Pass the real rate —
                using the nominal target rescales every temporal threshold when
                the device throttles. Falls back to the init FPS if None.

        Returns:
            Alerts for rules that triggered and are off cooldown, ordered by
            descending severity (documented in rule_engine.md, previously
            unimplemented).
        """
        fps = current_fps or self._fps
        now = time.monotonic()
        detected_names = {d.class_name for d in detections}
        alerts: list[Alert] = []

        with self._lock:
            for rule in self._rules:
                last_fired = self._cooldowns.get(rule.rule_id, 0.0)
                if now - last_fired < rule.cooldown_seconds:
                    continue  # Rule still cooling down

                if not evaluate_node(rule.condition, detected_names, memory, fps):
                    continue  # Condition not met

                alerts.append(
                    Alert(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        message=rule.message_en,
                        message_hi=rule.message_hi,
                        triggering_detections=detections,
                        timestamp_ms=time.time() * 1000,
                        cooldown_seconds=rule.cooldown_seconds,
                        frame_id=detections[0].frame_id if detections else 0,
                        explanation={
                            "rule_id": rule.rule_id,
                            "condition": rule.condition_text,
                            "detected_classes": sorted(detected_names),
                            "cooldown_seconds": rule.cooldown_seconds,
                            "vlm_available": context is not None,
                            "vlm_activity": context.activity if context else None,
                        },
                    )
                )
                self._cooldowns[rule.rule_id] = now
                logger.info(f"Alert: {rule.rule_id} [{rule.severity.name}]")

        alerts.sort(key=lambda a: a.severity.value, reverse=True)
        return alerts

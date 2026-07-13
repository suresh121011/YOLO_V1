"""
YAML-Driven Rule Engine
========================
Evaluates safety rules against YOLO detections, Event Memory state,
and optional SmolVLM2 context. Single decision point in the pipeline.

Rules are defined in configs/risk_rules.yaml and can be hot-reloaded
at runtime without restarting the pipeline.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import yaml

from . import Alert, Detection, SceneContext, Severity
from .event_memory import EventMemory

logger = logging.getLogger(__name__)


class RuleEngine:
    """YAML-driven safety rule evaluation engine.

    Evaluates all enabled rules against:
      - current YOLO detections
      - Event Memory (temporal state)
      - SmolVLM2 scene context (optional)

    Manages per-rule cooldowns internally.
    Thread-safe via RLock.
    """

    def __init__(self, rules_path: str, fps: float = 15.0) -> None:
        """
        Args:
            rules_path: Path to risk_rules.yaml.
            fps: Expected pipeline FPS (used for time-based rule conditions).
        """
        self._rules_path = Path(rules_path)
        self._lock = threading.RLock()
        self._cooldowns: dict[str, float] = {}
        self._rules: list[dict] = []
        self._fps = fps
        self._load_rules()

    # ─────────────────────────────────────────
    # Rule management
    # ─────────────────────────────────────────

    def _load_rules(self) -> None:
        with self._lock:
            data = yaml.safe_load(self._rules_path.read_text(encoding="utf-8"))
            self._rules = data.get("rules", [])
            logger.info(f"Loaded {len(self._rules)} rules from {self._rules_path}")

    def reload_rules(self) -> None:
        """Hot-reload rules from YAML without restarting the pipeline."""
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
            current_fps: Override FPS for time calculations (uses init FPS if None).

        Returns:
            List of Alert objects for rules that triggered and are off-cooldown.
        """
        fps = current_fps or self._fps
        now = time.monotonic()
        detected_names = {d.class_name for d in detections}
        alerts: list[Alert] = []

        with self._lock:
            for rule in self._rules:
                rule_id = rule.get("id", "")
                if not rule_id:
                    continue

                cooldown = rule.get("cooldown_seconds", 60)
                last_fired = self._cooldowns.get(rule_id, 0.0)

                if now - last_fired < cooldown:
                    continue  # Rule still cooling down

                condition = rule.get("condition", "")
                if not self._evaluate_condition(condition, detected_names, memory, fps):
                    continue  # Condition not met

                severity = Severity[rule.get("severity", "INFO")]
                alert = Alert(
                    rule_id=rule_id,
                    severity=severity,
                    message=rule.get("message_en", ""),
                    message_hi=rule.get("message_hi"),
                    triggering_detections=detections,
                    timestamp_ms=time.time() * 1000,
                    cooldown_seconds=cooldown,
                    frame_id=detections[0].frame_id if detections else 0,
                    explanation={
                        "rule_id": rule_id,
                        "condition": condition,
                        "detected_classes": sorted(detected_names),
                        "cooldown_seconds": cooldown,
                        "vlm_available": context is not None,
                        "vlm_activity": context.activity if context else None,
                    },
                )
                alerts.append(alert)
                self._cooldowns[rule_id] = now
                logger.info(f"Alert: {rule_id} [{severity.name}]")

        return alerts

    # ─────────────────────────────────────────
    # Condition evaluation DSL
    # ─────────────────────────────────────────

    def _evaluate_condition(
        self,
        condition: str,
        detected_names: set[str],
        memory: EventMemory,
        fps: float,
    ) -> bool:
        """Parse and evaluate a rule condition string.

        Supported DSL tokens:
          detected(class)                — class in current detections
          NOT detected(class)            — class NOT in current detections
          absent_for(class, seconds)     — class absent from memory for ≥ N seconds
          any_of([class1, class2, ...])  — any of the listed classes detected
          AND, OR                        — logical operators (left-to-right)

        Example conditions:
          "detected(knife) AND detected(person)"
          "detected(stove) AND absent_for(person, 30)"
          "any_of([medicine_strip, medicine_bottle])"
          "detected(wire) AND detected(person)"
        """
        condition = condition.strip()

        # any_of([class1, class2, ...])
        any_of_match = re.search(r"any_of\(\[([^\]]+)\]\)", condition)
        if any_of_match:
            classes = [c.strip() for c in any_of_match.group(1).split(",")]
            return any(cls in detected_names for cls in classes)

        # Split on AND (evaluate all sub-conditions with AND logic)
        if " AND " in condition:
            parts = condition.split(" AND ")
            return all(
                self._evaluate_condition(p.strip(), detected_names, memory, fps) for p in parts
            )

        # Split on OR
        if " OR " in condition:
            parts = condition.split(" OR ")
            return any(
                self._evaluate_condition(p.strip(), detected_names, memory, fps) for p in parts
            )

        # NOT detected(class)
        not_match = re.match(r"NOT\s+detected\((\w+)\)", condition)
        if not_match:
            return not_match.group(1) not in detected_names

        # absent_for(class, seconds)
        absent_match = re.match(r"absent_for\((\w+),\s*(\d+(?:\.\d+)?)\)", condition)
        if absent_match:
            cls = absent_match.group(1)
            seconds = float(absent_match.group(2))
            return memory.is_absent_for_by_name(cls, seconds, fps)

        # detected(class)
        detected_match = re.match(r"detected\((\w+)\)", condition)
        if detected_match:
            return detected_match.group(1) in detected_names

        logger.warning(f"Unrecognised rule condition: '{condition}'")
        return False

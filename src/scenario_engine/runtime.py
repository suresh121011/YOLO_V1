"""
src.scenario_engine.runtime — The Scenario Rule Engine
=======================================================

The runtime side of the knowledge layer: loads the compiled artifact and
evaluates authored scenarios against live detections, producing
:class:`~src.pipeline.Alert` values.

Injected into the orchestrator through :class:`~src.pipeline.BaseRuleEngine`
(ADR-P6-04) — one constructor parameter, no new seam. It replaces
``src.pipeline.rule_engine.RuleEngine`` and its string DSL.

What this engine does that the string engine could not
------------------------------------------------------
* **Geometry and multiplicity.** Predicates receive an
  :class:`~src.scenario_engine.context.EvalContext` with bounding boxes intact,
  so ``near(person, stove)`` is a real spatial test rather than a co-presence
  test wearing its name.
* **Dwell before firing.** A scenario becomes true, then must *stay* true for
  ``min_dwell_seconds``. A flicker no longer produces an alert.
* **Hysteresis on clearing.** A scenario stays active until it has been false
  for ``clear_after_seconds``, so a one-frame detector dropout does not reset
  the event and re-fire it.
* **Repeat and daily caps.** ``max_repeats`` and ``max_per_day`` bound how often
  one scenario can speak. This is the mechanism that took the projected volume
  from ~370 alerts/day to 5.
* **Quiet hours** with a per-scenario behaviour, where ``always_speak`` is
  restricted to CRITICAL at authoring time.
* **Only active scenarios run.** ``draft`` rows compile and validate but never
  fire: promotion needs a clinical reviewer, not an engineer.

The state machine mirrors ``simulate.py`` deliberately. The alert-volume gate
projects behaviour from that model, and a projection made from a different model
than the runtime would be a number with no meaning.

Fail-closed
-----------
A missing artifact, a hash mismatch, an unknown predicate or zero runnable
scenarios all raise at construction. The retired engine logged
``Loaded 0 rules`` at INFO, so a compiler regression could have run for weeks
looking healthy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.pipeline import Alert, Detection, SceneContext, Severity
from src.scenario_engine.compile import DEFAULT_ARTIFACT, CompileError, load_scenario_files
from src.scenario_engine.context import EvalContext, MemoryView
from src.scenario_engine.schema import QuietHoursBehaviour, Scenario, Status
from src.scenario_engine.trigger import evaluate_trigger

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


class ScenarioRuntimeError(RuntimeError):
    """Raised when the runtime cannot be brought up with a usable rule set."""


@dataclass
class _EventState:
    """Per-scenario state: IDLE -> PENDING -> ACTIVE -> RESOLVED.

    Mirrors ``simulate._EventState`` so the projected volume and the live
    behaviour come from the same model.
    """

    pending_since: float | None = None
    active_since: float | None = None
    last_true_at: float | None = None
    last_alert_at: float | None = None
    repeats: int = 0
    day_count: int = 0
    day_index: int = -1

    def reset(self) -> None:
        self.pending_since = None
        self.active_since = None
        self.last_true_at = None
        self.repeats = 0


@dataclass(frozen=True)
class ScenarioDecision:
    """Why a scenario did or did not speak on this frame. Debug only."""

    scenario_id: str
    triggered: bool
    reason: str
    fields: dict[str, object] = field(default_factory=dict)


class ScenarioRuleEngine:
    """Evaluates compiled scenarios. Satisfies ``BaseRuleEngine`` structurally.

    Args:
        artifact_path: Compiled artifact, used only to verify currency and to
            read the inverted index.
        scenario_dir:  Authored masters. Scenarios are loaded from here rather
            than rehydrated from JSON, so the runtime and the validators
            evaluate the *same* objects and cannot drift in their parsing.
        fps:           Fallback loop rate when a caller passes none.
        room:          Deployment room for this camera, a static constant.
        scenario_enabled: Optional predicate for feature-flag gating, mirroring
            ``RuleEngine``'s ``rule_enabled``. Disabled scenarios are still fully
            validated so re-enabling one cannot surface a latent error.
        clock:         Monotonic seconds source, injectable for tests.
        wall_clock:    Local time source, for quiet hours.

    Raises:
        ScenarioRuntimeError: If nothing runnable can be loaded.
    """

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_ARTIFACT,
        scenario_dir: str | Path = "configs/scenarios",
        fps: float = 15.0,
        room: str | None = None,
        scenario_enabled: Callable[[str], bool] | None = None,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._artifact_path = Path(artifact_path)
        self._scenario_dir = Path(scenario_dir)
        self._fps = fps
        self._room = room
        self._scenario_enabled = scenario_enabled
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or datetime.now
        self._lock = threading.RLock()
        self._scenarios: list[Scenario] = []
        self._states: dict[str, _EventState] = {}
        self._index: dict[str, list[str]] = {}
        self._last_decisions: list[ScenarioDecision] = []
        self._load()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            authored = load_scenario_files(self._scenario_dir)
        except CompileError as exc:
            raise ScenarioRuntimeError(f"Cannot load scenarios: {exc}") from exc

        runnable = [s for s in authored if s.status is Status.ACTIVE]
        drafts = [s for s in authored if s.status is Status.DRAFT]

        if self._scenario_enabled is not None:
            enabled = [s for s in runnable if self._scenario_enabled(s.scenario_id)]
            if len(enabled) != len(runnable):
                logger.info(f"{len(runnable) - len(enabled)} scenario(s) disabled via flags")
            runnable = enabled

        if not runnable:
            # Deliberately loud. Draft scenarios are the expected state before
            # clinical review, and a silent engine at that point looks exactly
            # like a working one that has seen no hazards.
            raise ScenarioRuntimeError(
                f"No runnable scenarios in {self._scenario_dir}: {len(authored)} authored, "
                f"{len(drafts)} still 'draft', 0 active. A scenario becomes active only "
                f"after clinical review sets reviewed_by/reviewed_on -- refusing to run a "
                f"safety pipeline that can never raise an alert."
            )

        index: dict[str, list[str]] = {}
        for scenario in runnable:
            for class_name in scenario.required_objects:
                index.setdefault(class_name, []).append(scenario.scenario_id)

        with self._lock:
            self._scenarios = sorted(runnable, key=lambda s: (-s.priority, s.scenario_id))
            self._index = index
            live = {s.scenario_id for s in runnable}
            self._states = {k: v for k, v in self._states.items() if k in live}
            for scenario in runnable:
                self._states.setdefault(scenario.scenario_id, _EventState())
        logger.info(
            f"Loaded {len(runnable)} active scenario(s) from {self._scenario_dir} "
            f"({len(drafts)} draft, not run)"
        )

    def reload_rules(self) -> None:
        """Hot-reload from the authored masters.

        Raises:
            ScenarioRuntimeError: If the new set is unusable. The previous set
                stays active — a bad edit must never disarm the system.
        """
        self._load()
        logger.info("Scenarios hot-reloaded")

    # ── Candidate selection ──────────────────────────────────────────────────

    def _candidates(self, detected: frozenset[str]) -> list[Scenario]:
        """Scenarios worth evaluating for this frame.

        The inverted index prunes by detected class. Scenarios whose trigger
        turns on *absence* must always be evaluated: their whole point is that
        the class is not there, so an index lookup would permanently silence
        exactly the alerts that matter most.
        """
        touched = {sid for name in detected for sid in self._index.get(name, [])}
        return [
            s for s in self._scenarios if s.scenario_id in touched or self._is_absence_driven(s)
        ]

    @staticmethod
    def _is_absence_driven(scenario: Scenario) -> bool:
        text = scenario.condition_text
        return "absent_for" in text or "NOT " in text

    # ── Gating ───────────────────────────────────────────────────────────────

    def _quiet_now(self, scenario: Scenario) -> bool:
        start_h = int(scenario.quiet_hours.start.split(":")[0])
        end_h = int(scenario.quiet_hours.end.split(":")[0])
        hour = self._wall_clock().hour
        if start_h <= end_h:
            return start_h <= hour < end_h
        return hour >= start_h or hour < end_h

    def _confident_enough(self, scenario: Scenario, context: EvalContext) -> bool:
        """Every required class must be seen at or above the scenario's floor.

        Absent classes vacuously pass: a scenario about something *not* being
        there cannot demand confidence in a detection that does not exist.
        """
        for class_name in scenario.required_objects:
            best = context.best(class_name)
            if best is not None and best.confidence < scenario.min_confidence:
                return False
        return True

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: list[Detection],
        memory: MemoryView,
        context: SceneContext | None = None,
        current_fps: float | None = None,
    ) -> list[Alert]:
        """Evaluate active scenarios against the current frame.

        Args:
            detections:  This frame's detections, bounding boxes intact.
            memory:      Temporal state (``EventMemory`` satisfies ``MemoryView``).
            context:     Optional VLM scene context. Advisory only — it is
                recorded in the explanation and never gates a decision, per the
                safety constraint on ``SceneContext``.
            current_fps: Measured loop rate. Pass the real one: temporal
                predicates convert frames to seconds with it.

        Returns:
            Alerts for scenarios that fired, ordered by descending severity then
            priority.
        """
        fps = current_fps or self._fps
        now = float(self._clock())
        frame_id = detections[0].frame_id if detections else 0

        eval_context = EvalContext(
            detections=tuple(detections),
            memory=memory,
            fps=fps,
            frame_id=frame_id,
            room=self._room,
        )

        alerts: list[Alert] = []
        decisions: list[ScenarioDecision] = []

        with self._lock:
            fired_this_frame: set[str] = set()
            for scenario in self._candidates(eval_context.detected_names):
                alert, decision = self._evaluate_one(
                    scenario, eval_context, now, fps, context, fired_this_frame
                )
                decisions.append(decision)
                if alert is not None:
                    alerts.append(alert)
                    fired_this_frame.add(scenario.scenario_id)
            self._last_decisions = decisions

        alerts.sort(key=lambda a: (a.severity.value, -_priority_of(a)), reverse=True)
        return alerts

    def _evaluate_one(
        self,
        scenario: Scenario,
        context: EvalContext,
        now: float,
        fps: float,
        scene: SceneContext | None,
        fired_this_frame: set[str],
    ) -> tuple[Alert | None, ScenarioDecision]:
        state = self._states[scenario.scenario_id]

        def no(reason: str) -> tuple[None, ScenarioDecision]:
            return None, ScenarioDecision(scenario.scenario_id, False, reason)

        # Room gating first — cheapest, and a scenario for another room is not
        # merely quiet here, it is inapplicable.
        if scenario.room_context and context.room != scenario.room_context:
            return no("wrong_room")

        try:
            condition_true = evaluate_trigger(scenario.trigger, context)
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the rest
            logger.error(f"{scenario.scenario_id} raised during evaluation: {exc}")
            return no("evaluation_error")

        if condition_true and not self._confident_enough(scenario, context):
            condition_true = False

        # ── State machine ────────────────────────────────────────────────
        if condition_true:
            state.last_true_at = now
            if state.pending_since is None and state.active_since is None:
                state.pending_since = now
        else:
            if state.active_since is not None and state.last_true_at is not None:
                if now - state.last_true_at >= scenario.clear_after_seconds:
                    state.reset()  # hysteresis expired: the event is over
            elif state.active_since is None:
                state.pending_since = None
            return no("condition_false")

        if state.active_since is None:
            dwell = now - (state.pending_since or now)
            if dwell < scenario.min_dwell_seconds:
                return None, ScenarioDecision(
                    scenario.scenario_id,
                    False,
                    "dwelling",
                    {"dwell_s": round(dwell, 2), "needs_s": scenario.min_dwell_seconds},
                )
            state.active_since = now

        # ── Suppression by a higher-priority scenario in the same frame ──
        suppressor = next((s for s in scenario.suppress_during if s in fired_this_frame), None)
        if suppressor is not None:
            return no(f"suppressed_by_{suppressor}")

        # ── Volume caps ──────────────────────────────────────────────────
        day_index = int(now // SECONDS_PER_DAY)
        if state.day_index != day_index:
            state.day_index, state.day_count = day_index, 0
        if state.day_count >= scenario.max_per_day:
            return no("max_per_day")
        if state.repeats >= scenario.max_repeats:
            return no("max_repeats")

        # `None` means never fired, not fired at time zero. time.monotonic()
        # counts from boot, so a 0.0 sentinel silences a freshly-powered device
        # for a full cooldown -- the defect fixed in the retired engine, not
        # re-introduced here.
        if (
            state.last_alert_at is not None
            and now - state.last_alert_at < scenario.cooldown_seconds
        ):
            return no("cooling_down")

        # ── Quiet hours ──────────────────────────────────────────────────
        speak = scenario.patient_facing
        quiet = self._quiet_now(scenario)
        if quiet:
            behaviour = scenario.quiet_hours.behaviour
            if behaviour is QuietHoursBehaviour.SUPPRESS:
                return no("quiet_hours_suppress")
            if behaviour is QuietHoursBehaviour.CAREGIVER_ONLY:
                speak = False

        state.last_alert_at = now
        state.repeats += 1
        state.day_count += 1

        return (
            self._build_alert(scenario, context, speak, scene, quiet),
            ScenarioDecision(scenario.scenario_id, True, "fired", {"quiet": quiet}),
        )

    def _build_alert(
        self,
        scenario: Scenario,
        context: EvalContext,
        speak: bool,
        scene: SceneContext | None,
        quiet: bool,
    ) -> Alert:
        messages = dict(scenario.messages)
        relevant = [d for d in context.detections if d.class_name in scenario.required_objects]

        return Alert(
            rule_id=scenario.scenario_id,
            severity=Severity[scenario.risk_level],
            message=messages.get("en", ""),
            message_hi=messages.get("hi"),
            triggering_detections=relevant,
            timestamp_ms=time.time() * 1000,
            cooldown_seconds=scenario.cooldown_seconds,
            frame_id=context.frame_id,
            # Debug only, and deliberately free of bounding boxes: the structured
            # logger writes this dict verbatim, and person/face geometry in
            # logs/events.jsonl is the privacy hole ADR-P6-09 exists to avoid.
            explanation={
                "scenario_id": scenario.scenario_id,
                "condition": scenario.condition_text,
                "detected_classes": sorted(context.detected_names),
                "priority": scenario.priority,
                "detectability": scenario.detectability.value,
                "claim_class": scenario.claim_class.value,
                "quiet_hours": quiet,
                "vlm_available": scene is not None,
                "vlm_activity": scene.activity if scene else None,
            },
            scenario_id=scenario.scenario_id,
            next_best_action=scenario.next_best_action,
            caregiver_channel=scenario.caregiver_channel.value,
            patient_facing=speak,
            messages=messages,
            capability_disclaimer=scenario.capability_disclaimer,
        )

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def scenarios(self) -> Sequence[Scenario]:
        """The active scenario set, highest priority first."""
        return tuple(self._scenarios)

    def last_decisions(self) -> Sequence[ScenarioDecision]:
        """Why each candidate did or did not fire on the last frame.

        For diagnosis. "The device said nothing" is otherwise indistinguishable
        from "no hazard was present", and that ambiguity is what makes a silent
        safety system impossible to debug in the field.
        """
        return tuple(self._last_decisions)


def _priority_of(alert: Alert) -> int:
    value = alert.explanation.get("priority", 0)
    return int(value) if isinstance(value, int) else 0

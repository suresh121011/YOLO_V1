"""
src.scenario_engine.schema — The Scenario Row
=============================================

Frozen dataclasses plus a hand-rolled ``_validate`` — the house pattern
(``src/dataset/sources_config.py``, ``src/dataset/release/gates.py``). No
pydantic: it would be a new dependency and a break from convention, and would
need its own ADR.

What is authored vs. derived
----------------------------
The **trigger tree is the only authored condition**. ``required_objects`` is
*derived* from it, so the CSV view and the compiled artifact carry that column
without a second source of truth that can drift from the executable one.
``optional_objects`` is authored and purely documentary — objects that may be
present without affecting the decision.

A separate ``forbidden_objects`` column is deliberately **not** carried here.
Negation lives inside the trigger (``not: {op: detected, ...}``), where it
composes with everything else; splitting it back out into a flat list would
reintroduce exactly the drift this derivation exists to prevent. The compiler
renders the full condition for review instead.

``rule_hash`` covers semantics only
-----------------------------------
Per ADR-P6-06 the hash spans the trigger and the behavioural scalars, and
deliberately excludes prompts, notes, evidence and the display name — so fixing
a Hindi typo invalidates zero labelled clips, while downgrading a risk level or
adding a required object invalidates exactly the clips that were adjudicated
against the old behaviour.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .trigger import Trigger, parse_trigger, render_trigger, trigger_classes

SCENARIO_ID_PATTERN = r"^SC-[A-Z]{3}-\d{3}$"

#: Severity names, mirroring src.pipeline.Severity. Duplicated rather than
#: imported so the knowledge layer states its own vocabulary, exactly as
#: src/dataset/release/gates.py duplicates GateResult rather than importing
#: from src/training — the layering direction matters more than the DRY win.
RISK_LEVELS = ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")


class ScenarioError(ValueError):
    """Raised when a scenario definition is invalid. Always names the field."""


class Detectability(str, Enum):
    """How the system arrives at this scenario."""

    DIRECT = "direct"
    INFERRED = "inferred"
    REJECTED = "rejected"


class ClaimClass(str, Enum):
    """What kind of statement the scenario makes about the world."""

    OBSERVATION = "observation"
    REMINDER = "reminder"
    INFERENCE = "inference"


class CaregiverChannel(str, Enum):
    """How a caregiver is told.

    An enum rather than a boolean: as a boolean it gets set true everywhere,
    which fatigues the caregiver — the person whose attention actually protects
    the resident.
    """

    NONE = "none"
    DIGEST = "digest"
    PUSH = "push"
    PUSH_AND_CALL = "push_and_call"


class Status(str, Enum):
    """Lifecycle state. Deprecated and rejected rows stay on disk."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class EvidenceStrength(str, Enum):
    """How well-founded the claim behind a scenario is."""

    GUIDELINE = "guideline"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"
    OBSERVATIONAL = "observational"
    EXPERT_OPINION = "expert_opinion"
    NONE = "none"


class EvidenceSupports(str, Enum):
    """What the citation actually backs.

    ``hazard`` never licenses ``detectability: direct``. Published guidance backs
    the *hazard*; nothing in the literature backs our *detector*. Keeping these
    separate blocks the commonest self-deception in this work — citing a
    guideline to legitimise a detection that cannot be made.
    """

    HAZARD = "hazard"
    INTERVENTION = "intervention"
    DETECTION_METHOD = "detection_method"


class QuietHoursBehaviour(str, Enum):
    """What happens overnight. ``always_speak`` is CRITICAL-only."""

    SUPPRESS = "suppress"
    CAREGIVER_ONLY = "caregiver_only"
    ALWAYS_SPEAK = "always_speak"


def _enum(value: Any, enum_cls: type[Enum], where: str) -> Any:
    """Coerce a YAML scalar to an enum member with an enumerating error."""
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = [member.value for member in enum_cls]
        raise ScenarioError(f"{where} must be one of {valid}, got {value!r}") from exc


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ScenarioError(f"{where} must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ScenarioError(f"{where} must contain only non-empty strings, got {item!r}")
    return tuple(value)


@dataclass(frozen=True)
class Evidence:
    """One citation behind a scenario."""

    source_id: str
    strength: EvidenceStrength
    supports: EvidenceSupports
    section: str = ""
    item: str = ""
    note: str = ""

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> Evidence:
        if not isinstance(raw, dict):
            raise ScenarioError(f"{where} must be a mapping, got {type(raw).__name__}")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ScenarioError(f"{where}.source_id is required")
        return cls(
            source_id=source_id,
            strength=_enum(raw.get("strength"), EvidenceStrength, f"{where}.strength"),
            supports=_enum(raw.get("supports"), EvidenceSupports, f"{where}.supports"),
            section=str(raw.get("section", "")),
            item=str(raw.get("item", "")),
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True)
class QuietHours:
    """Night-time policy. Night is a mode, not a modifier."""

    start: str = "21:00"
    end: str = "06:00"
    behaviour: QuietHoursBehaviour = QuietHoursBehaviour.SUPPRESS

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> QuietHours:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ScenarioError(f"{where} must be a mapping, got {type(raw).__name__}")
        return cls(
            start=str(raw.get("start", "21:00")),
            end=str(raw.get("end", "06:00")),
            behaviour=_enum(
                raw.get("behaviour", "suppress"), QuietHoursBehaviour, f"{where}.behaviour"
            ),
        )


@dataclass(frozen=True)
class EscalationStep:
    """One rung of the ladder. Escalation reworded, never repeated louder."""

    after_seconds: float
    action: str
    messages: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> EscalationStep:
        if not isinstance(raw, dict):
            raise ScenarioError(f"{where} must be a mapping, got {type(raw).__name__}")
        after = raw.get("after_seconds")
        if not isinstance(after, int | float) or isinstance(after, bool) or after <= 0:
            raise ScenarioError(f"{where}.after_seconds must be a positive number, got {after!r}")
        action = raw.get("action")
        if not isinstance(action, str) or not action:
            raise ScenarioError(f"{where}.action is required")
        messages = raw.get("messages", {}) or {}
        if not isinstance(messages, dict):
            raise ScenarioError(f"{where}.messages must be a mapping of locale -> text")
        return cls(
            after_seconds=float(after),
            action=action,
            messages={str(k): str(v) for k, v in messages.items()},
        )


@dataclass(frozen=True)
class Scenario:
    """One authored, reviewable, versioned scenario."""

    scenario_id: str
    schema_version: int
    scenario_name: str
    category: str
    trigger: Trigger
    risk_level: str
    detectability: Detectability
    claim_class: ClaimClass
    status: Status
    caregiver_channel: CaregiverChannel
    next_best_action: str
    messages: Mapping[str, str]

    min_confidence: float
    min_dwell_seconds: float
    cooldown_seconds: int
    max_repeats: int
    max_per_day: int
    priority: int
    clear_after_seconds: float

    quiet_hours: QuietHours
    evidence: tuple[Evidence, ...]
    escalation: tuple[EscalationStep, ...]

    patient_facing: bool = True
    room_context: str | None = None
    optional_objects: tuple[str, ...] = ()
    suppress_during: tuple[str, ...] = ()
    capability_disclaimer: str = ""
    rejection_reason: str = ""
    false_positive_notes: str = ""
    notes: str = ""
    reviewed_by: str = ""
    reviewed_on: str = ""
    deprecated_in: str = ""
    superseded_by: str = ""
    sample_video_needed: bool = True

    # ── Derived ──────────────────────────────────────────────────────────────

    @property
    def condition_text(self) -> str:
        """One-line rendering of the trigger, for the CSV view and logs."""
        return render_trigger(self.trigger)

    @property
    def required_objects(self) -> tuple[str, ...]:
        """Classes the trigger references. Derived — never authored separately."""
        return tuple(sorted(trigger_classes(self.trigger)))

    def semantic_fields(self) -> dict[str, Any]:
        """Exactly the fields ``rule_hash`` covers.

        Prompts, notes, evidence and the display name are excluded so
        translation and documentation work never invalidates a labelled clip.
        """
        return {
            "trigger": self.condition_text,
            "risk_level": self.risk_level,
            "caregiver_channel": self.caregiver_channel.value,
            "min_confidence": self.min_confidence,
            "min_dwell_seconds": self.min_dwell_seconds,
            "clear_after_seconds": self.clear_after_seconds,
            "room_context": self.room_context,
            "optional_objects": list(self.optional_objects),
            "suppress_during": list(self.suppress_during),
            "patient_facing": self.patient_facing,
        }

    def rule_hash(self) -> str:
        """Stable digest of the scenario's *behaviour*.

        Computed exactly like the two existing fingerprints in this repo
        (``completeness.taxonomy_fingerprint``, ``annotation.base``).
        """
        canonical = json.dumps(
            self.semantic_fields(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "") -> Scenario:
        """Build and validate a scenario from its authored YAML mapping.

        Raises:
            ScenarioError: On any missing, malformed, or out-of-range field.
                Structural validity only — cross-scenario checks (duplicate ids,
                class capability, evidence-gates-risk) belong to the compiler.
        """
        if not isinstance(raw, dict):
            raise ScenarioError(f"{where or 'scenario'} must be a mapping")

        scenario_id = str(raw.get("scenario_id", "")).strip()
        prefix = where or scenario_id or "scenario"

        if not re.match(SCENARIO_ID_PATTERN, scenario_id):
            raise ScenarioError(
                f"{prefix}.scenario_id must match {SCENARIO_ID_PATTERN} "
                f"(e.g. SC-KIT-001), got {scenario_id!r}"
            )

        risk_level = str(raw.get("risk_level", "")).upper()
        if risk_level not in RISK_LEVELS:
            raise ScenarioError(
                f"{prefix}.risk_level must be one of {list(RISK_LEVELS)}, got {risk_level!r}"
            )

        trigger_raw = raw.get("trigger")
        if trigger_raw is None:
            raise ScenarioError(f"{prefix}.trigger is required")
        trigger = parse_trigger(trigger_raw, f"{prefix}.trigger")

        messages_raw = raw.get("messages", {}) or {}
        if not isinstance(messages_raw, dict):
            raise ScenarioError(f"{prefix}.messages must be a mapping of locale -> text")
        messages = {str(k): str(v) for k, v in messages_raw.items()}

        evidence_raw = raw.get("evidence", []) or []
        if not isinstance(evidence_raw, list):
            raise ScenarioError(f"{prefix}.evidence must be a list")
        evidence = tuple(
            Evidence.from_mapping(item, f"{prefix}.evidence[{i}]")
            for i, item in enumerate(evidence_raw)
        )

        escalation_raw = raw.get("escalation", []) or []
        if not isinstance(escalation_raw, list):
            raise ScenarioError(f"{prefix}.escalation must be a list")
        escalation = tuple(
            EscalationStep.from_mapping(item, f"{prefix}.escalation[{i}]")
            for i, item in enumerate(escalation_raw)
        )

        def _number(key: str, default: Any, minimum: float, maximum: float | None = None) -> float:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ScenarioError(f"{prefix}.{key} must be a number, got {value!r}")
            if value < minimum or (maximum is not None and value > maximum):
                bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise ScenarioError(f"{prefix}.{key} must be {bound}, got {value}")
            return float(value)

        def _whole(key: str, default: int, minimum: int) -> int:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScenarioError(f"{prefix}.{key} must be a whole number, got {value!r}")
            if value < minimum:
                raise ScenarioError(f"{prefix}.{key} must be >= {minimum}, got {value}")
            return int(value)

        scenario = cls(
            scenario_id=scenario_id,
            schema_version=_whole("schema_version", 1, 1),
            scenario_name=str(raw.get("scenario_name", "")).strip(),
            category=str(raw.get("category", "")).strip(),
            trigger=trigger,
            risk_level=risk_level,
            detectability=_enum(raw.get("detectability"), Detectability, f"{prefix}.detectability"),
            claim_class=_enum(raw.get("claim_class"), ClaimClass, f"{prefix}.claim_class"),
            status=_enum(raw.get("status", "draft"), Status, f"{prefix}.status"),
            caregiver_channel=_enum(
                raw.get("caregiver_channel", "none"),
                CaregiverChannel,
                f"{prefix}.caregiver_channel",
            ),
            next_best_action=str(raw.get("next_best_action", "")).strip(),
            messages=messages,
            min_confidence=_number("min_confidence", 0.5, 0.0, 1.0),
            min_dwell_seconds=_number("min_dwell_seconds", 0.0, 0.0),
            cooldown_seconds=_whole("cooldown_seconds", 60, 0),
            max_repeats=_whole("max_repeats", 2, 0),
            max_per_day=_whole("max_per_day", 6, 0),
            priority=_whole("priority", 100, 0),
            clear_after_seconds=_number("clear_after_seconds", 30.0, 0.0),
            quiet_hours=QuietHours.from_mapping(raw.get("quiet_hours"), f"{prefix}.quiet_hours"),
            evidence=evidence,
            escalation=escalation,
            patient_facing=bool(raw.get("patient_facing", True)),
            room_context=(str(raw["room_context"]) if raw.get("room_context") else None),
            optional_objects=_str_list(raw.get("optional_objects"), f"{prefix}.optional_objects"),
            suppress_during=_str_list(raw.get("suppress_during"), f"{prefix}.suppress_during"),
            capability_disclaimer=str(raw.get("capability_disclaimer", "")).strip(),
            rejection_reason=str(raw.get("rejection_reason", "")).strip(),
            false_positive_notes=str(raw.get("false_positive_notes", "")).strip(),
            notes=str(raw.get("notes", "")).strip(),
            reviewed_by=str(raw.get("reviewed_by", "")).strip(),
            reviewed_on=str(raw.get("reviewed_on", "")).strip(),
            deprecated_in=str(raw.get("deprecated_in", "")).strip(),
            superseded_by=str(raw.get("superseded_by", "")).strip(),
            sample_video_needed=bool(raw.get("sample_video_needed", True)),
        )
        scenario._validate(prefix)
        return scenario

    def _validate(self, prefix: str) -> None:
        """Structural invariants that need the whole row assembled."""
        if not self.scenario_name:
            raise ScenarioError(f"{prefix}.scenario_name is required")
        if not self.category:
            raise ScenarioError(f"{prefix}.category is required")

        if self.detectability is Detectability.REJECTED:
            if not self.rejection_reason:
                raise ScenarioError(
                    f"{prefix}: detectability 'rejected' requires a rejection_reason naming "
                    f"the missing capability, not merely a refusal (ADR-P6-08)"
                )
            return  # A rejected row never runs; the rest of the contract is moot.

        if self.detectability is Detectability.INFERRED and not self.capability_disclaimer:
            raise ScenarioError(
                f"{prefix}: detectability 'inferred' requires a capability_disclaimer, "
                f"surfaced with the alert so an inference is never read as an observation"
            )

        if self.status is Status.ACTIVE and not (self.reviewed_by and self.reviewed_on):
            raise ScenarioError(
                f"{prefix}: status 'active' requires reviewed_by and reviewed_on. This "
                f"project already demands two annotators at IAA >= 0.75 to accept a bounding "
                f"box; an instruction spoken to an elderly person deserves at least as much."
            )

        if self.patient_facing and not self.messages:
            raise ScenarioError(f"{prefix}: a patient-facing scenario needs at least one message")

        if (
            self.quiet_hours.behaviour is QuietHoursBehaviour.ALWAYS_SPEAK
            and self.risk_level != "CRITICAL"
        ):
            raise ScenarioError(
                f"{prefix}: quiet_hours.behaviour 'always_speak' is permitted only for "
                f"CRITICAL, got {self.risk_level}. Waking someone at night creates the "
                f"rushed, disoriented state that causes falls."
            )

        if self.max_repeats > 3:
            raise ScenarioError(
                f"{prefix}.max_repeats is capped at 3, got {self.max_repeats}. Repetition is "
                f"not escalation — reword or hand off to the caregiver instead."
            )

        if self.escalation:
            offsets = [step.after_seconds for step in self.escalation]
            if offsets != sorted(offsets):
                raise ScenarioError(f"{prefix}.escalation steps must be ordered by after_seconds")


def scenario_sort_key(scenario: Scenario) -> tuple[int, str]:
    """Total order for arbitration and for deterministic artifact output."""
    return (scenario.priority, scenario.scenario_id)


def sorted_scenarios(scenarios: Sequence[Scenario]) -> list[Scenario]:
    """Scenarios in their canonical order, independent of file iteration order."""
    return sorted(scenarios, key=scenario_sort_key)

"""
src.scenario_engine.validators — Report-Level Scenario Checks
=============================================================

The compiler hard-fails on anything that makes a scenario *unrunnable*
(unknown class, static-only trigger, uncited alarm). These validators cover
what makes a scenario *wrong* rather than broken, and produce a report with
ERROR/WARN severities rather than raising — following the
``run_full_qa.py --exit-zero-on-warnings`` convention.

The two that do the most work
-----------------------------
**No unobservable assertions.** The taxonomy is static objects plus
``person``/``face``: no pose, no on/off state, no tracking. So a message may
describe appearance, never assert state. ``configs/risk_rules.yaml:54`` says the
stove "appears to be **on**", which the system cannot see; a caregiver who drives
across the city to find a cold stove stops responding, and stops responding the
fourth time too. The same check catches questions the system cannot hear the
answer to — there is no ASR, and an unanswerable question invites a
resolve-the-uncertainty second dose.

**Exhaustive determinism.** The set that fires, and its order, must be a pure
function of the detections — independent of file iteration order. Enumerating
subsets of the *referenced* classes only makes this cheap: a dozen distinct
classes is ~4096 evaluations, not 2^23.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..pipeline import BoundingBox, Detection
from .context import EvalContext
from .schema import Scenario, Status
from .trigger import evaluate_trigger, trigger_classes

ERROR = "ERROR"
WARN = "WARN"

#: Phrases asserting state this taxonomy cannot observe. Matched case-insensitively
#: on word boundaries in every user-facing string.
UNOBSERVABLE_PHRASES: tuple[str, ...] = (
    "is on",
    "is off",
    # "appears to be on" — the exact hedge risk_rules.yaml:54 used. Hedging the
    # verb does not make an unobservable state observable.
    "be on",
    "be off",
    "turned on",
    "left on",
    "has fallen",
    "is running",
    "you took",
    "did not take",
    "have taken",
    "is unconscious",
    "not moving",
)

#: Words whose startle profile is itself a fall mechanism. Not a style preference.
STARTLE_PHRASES: tuple[str, ...] = (
    "careful",
    "watch out",
    "danger",
    "warning",
    "stop!",
    "hurry",
)

#: Verbs that read as a medical-device claim.
REGULATED_VERBS: tuple[str, ...] = (
    "detect",
    "diagnose",
    "monitor",
    "prevent",
    "ensure",
    "guarantee",
    "protect",
    "emergency",
)

#: Presbycusis costs sentence-final content first, and working memory is finite.
MAX_PATIENT_MESSAGE_WORDS = 14

#: Devanagari block, for checking a `hi` string is actually Hindi.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass(frozen=True)
class Finding:
    """One validation result."""

    code: str
    severity: str
    scenario_id: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "scenario_id": self.scenario_id,
            "message": self.message,
        }


def _phrase_re(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


def _patient_strings(scenario: Scenario) -> list[tuple[str, str]]:
    """Every string spoken to the resident, with its field path."""
    strings = [(f"messages.{lang}", text) for lang, text in sorted(scenario.messages.items())]
    for index, step in enumerate(scenario.escalation):
        strings.extend(
            (f"escalation[{index}].messages.{lang}", text)
            for lang, text in sorted(step.messages.items())
        )
    return strings


# ─── Message checks ───────────────────────────────────────────────────────────


def check_unobservable_assertions(scenarios: Sequence[Scenario]) -> list[Finding]:
    """No message may assert state the taxonomy cannot observe."""
    findings: list[Finding] = []
    for scenario in scenarios:
        for where, text in _patient_strings(scenario):
            for phrase in UNOBSERVABLE_PHRASES:
                if _phrase_re(phrase).search(text):
                    findings.append(
                        Finding(
                            "unobservable-assertion",
                            ERROR,
                            scenario.scenario_id,
                            f"{where} asserts {phrase!r}, which no class in this taxonomy "
                            f"encodes. Describe appearance instead ('looks wet', 'is "
                            f"visible'), never state.",
                        )
                    )
    return findings


def check_unanswerable_questions(scenarios: Sequence[Scenario]) -> list[Finding]:
    """There is no ASR, so a question is a prompt the system cannot act on."""
    findings: list[Finding] = []
    for scenario in scenarios:
        if not scenario.patient_facing:
            continue
        for where, text in _patient_strings(scenario):
            if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
                findings.append(
                    Finding(
                        "unanswerable-question",
                        ERROR,
                        scenario.scenario_id,
                        f"{where} asks a question, but there is no speech recognition — the "
                        f"answer is never heard. An unanswerable question from a machine is "
                        f"distressing with cognitive impairment, and invites a "
                        f"resolve-the-uncertainty second dose.",
                    )
                )
    return findings


def check_startle_language(scenarios: Sequence[Scenario]) -> list[Finding]:
    """Startle is a fall mechanism, so startling wording is an ERROR."""
    findings: list[Finding] = []
    for scenario in scenarios:
        if not scenario.patient_facing:
            continue
        for where, text in _patient_strings(scenario):
            for phrase in STARTLE_PHRASES:
                if _phrase_re(phrase).search(text):
                    findings.append(
                        Finding(
                            "startle-language",
                            ERROR,
                            scenario.scenario_id,
                            f"{where} contains {phrase!r}. A sudden loud warning to someone "
                            f"holding a knife or mid-transfer causes the injury it warns "
                            f"about. Lead with the observation instead.",
                        )
                    )
    return findings


def check_regulated_claims(scenarios: Sequence[Scenario]) -> list[Finding]:
    """Regulated verbs turn a wellness product into a medical-device claim."""
    findings: list[Finding] = []
    for scenario in scenarios:
        for where, text in _patient_strings(scenario):
            for verb in REGULATED_VERBS:
                if _phrase_re(verb).search(text):
                    findings.append(
                        Finding(
                            "regulated-claim",
                            WARN,
                            scenario.scenario_id,
                            f"{where} uses {verb!r}. Claiming to detect, monitor or prevent a "
                            f"condition is what crosses the medical-device line; keep the "
                            f"wording observational.",
                        )
                    )
    return findings


def check_message_length(scenarios: Sequence[Scenario]) -> list[Finding]:
    """Long prompts are heard as noise, and the tail is lost first."""
    findings: list[Finding] = []
    for scenario in scenarios:
        if not scenario.patient_facing:
            continue
        for where, text in _patient_strings(scenario):
            words = len(text.split())
            if words > MAX_PATIENT_MESSAGE_WORDS:
                findings.append(
                    Finding(
                        "message-too-long",
                        WARN,
                        scenario.scenario_id,
                        f"{where} is {words} words (limit {MAX_PATIENT_MESSAGE_WORDS}). "
                        f"Presbycusis costs sentence-final content first.",
                    )
                )
    return findings


def check_locale_completeness(scenarios: Sequence[Scenario]) -> list[Finding]:
    """Both locales present, and `hi` actually written in Devanagari."""
    findings: list[Finding] = []
    for scenario in scenarios:
        if not scenario.patient_facing:
            continue
        for locale in ("en", "hi"):
            if not scenario.messages.get(locale, "").strip():
                findings.append(
                    Finding(
                        "missing-locale",
                        WARN,
                        scenario.scenario_id,
                        f"messages.{locale} is missing. Hindi alone is also insufficient "
                        f"coverage for the target population; the map is open-ended.",
                    )
                )
        hindi = scenario.messages.get("hi", "")
        if hindi.strip() and not _DEVANAGARI.search(hindi):
            findings.append(
                Finding(
                    "locale-not-translated",
                    ERROR,
                    scenario.scenario_id,
                    "messages.hi contains no Devanagari — an English string was pasted "
                    "into the Hindi field.",
                )
            )
    return findings


# ─── Reachability ─────────────────────────────────────────────────────────────


def check_confidence_reachable(
    scenarios: Sequence[Scenario], class_thresholds: Mapping[str, float], global_floor: float
) -> list[Finding]:
    """A `min_confidence` below the detector's floor is a dead rule."""
    findings: list[Finding] = []
    for scenario in scenarios:
        for class_name in sorted(trigger_classes(scenario.trigger)):
            floor = float(class_thresholds.get(class_name, global_floor))
            if scenario.min_confidence < floor:
                findings.append(
                    Finding(
                        "confidence-unreachable",
                        WARN,
                        scenario.scenario_id,
                        f"min_confidence {scenario.min_confidence} is below the effective "
                        f"detector floor for {class_name!r} ({floor}); the detector never "
                        f"emits a box that low, so the threshold has no effect.",
                    )
                )
    return findings


def check_temporal_within_memory(
    scenarios: Sequence[Scenario], memory_window_frames: int, target_fps: float
) -> list[Finding]:
    """A dwell longer than the memory window can never be satisfied."""
    ceiling = memory_window_frames / max(target_fps, 0.1)
    findings: list[Finding] = []
    for scenario in scenarios:
        if scenario.min_dwell_seconds > ceiling:
            findings.append(
                Finding(
                    "dwell-exceeds-memory",
                    ERROR,
                    scenario.scenario_id,
                    f"min_dwell_seconds {scenario.min_dwell_seconds} exceeds the Event "
                    f"Memory ceiling of {ceiling:.1f}s ({memory_window_frames} frames at "
                    f"{target_fps} FPS). Raise memory_window_frames or lower the dwell.",
                )
            )
    return findings


def check_safety_class_coverage(
    scenarios: Sequence[Scenario], required_classes: Iterable[str]
) -> list[Finding]:
    """Every safety-critical class needs at least one active scenario.

    The asymmetry is the point: zero references for ``book`` or ``laptop`` is
    fine; zero for a safety-critical class means a hazard the system can see and
    has nothing to say about.
    """
    covered: set[str] = set()
    for scenario in scenarios:
        if scenario.status is Status.ACTIVE:
            covered |= trigger_classes(scenario.trigger)

    return [
        Finding(
            "safety-class-uncovered",
            WARN,
            "",
            f"Safety-critical class {class_name!r} is referenced by no active scenario.",
        )
        for class_name in sorted(set(required_classes) - covered)
    ]


# ─── Determinism ──────────────────────────────────────────────────────────────


class _DeterministicMemory:
    """Memory stub whose answers depend only on the class name.

    Determinism is a property of the *ordering*, so any reproducible temporal
    answer works — what matters is that two runs see identical state.
    """

    def __init__(self, present: frozenset[str]) -> None:
        self._present = present

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        return class_name not in self._present

    def consecutive_frames(self, class_id: int) -> int:
        return 10_000  # any dwell is satisfied

    def has_ever_seen(self, class_name: str) -> bool:
        return class_name in self._present


def _context_for(present: frozenset[str], room: str | None) -> EvalContext:
    detections = [
        Detection(
            class_id=index,
            class_name=name,
            confidence=0.9,
            bbox=BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1),
            frame_id=0,
            timestamp_ms=0.0,
        )
        for index, name in enumerate(sorted(present))
    ]
    return EvalContext.from_frame(detections, _DeterministicMemory(present), 15.0, room=room)


def fired_scenarios(scenarios: Sequence[Scenario], ctx: EvalContext) -> list[str]:
    """Ids that fire for a context, in canonical (priority, id) order."""
    fired = [s for s in scenarios if evaluate_trigger(s.trigger, ctx)]
    return [s.scenario_id for s in sorted(fired, key=lambda s: (s.priority, s.scenario_id))]


def check_determinism(scenarios: Sequence[Scenario], max_classes: int = 14) -> list[Finding]:
    """The fired set must not depend on scenario iteration order.

    Enumerates subsets of the union of *referenced* classes only. Above
    ``max_classes`` the space is truncated to the most-referenced classes rather
    than sampled randomly, so the check stays reproducible.
    """
    referenced: set[str] = set()
    for scenario in scenarios:
        referenced |= trigger_classes(scenario.trigger)

    ordered = sorted(referenced)[:max_classes]
    # None sorts first; a plain sorted() over {str, None} raises TypeError.
    named_rooms = sorted({s.room_context for s in scenarios if s.room_context})
    rooms: list[str | None] = [None, *named_rooms]

    reversed_scenarios = list(reversed(scenarios))
    findings: list[Finding] = []

    for size in range(len(ordered) + 1):
        for combo in itertools.combinations(ordered, size):
            present = frozenset(combo)
            for room in rooms:
                ctx = _context_for(present, room)
                forward = fired_scenarios(scenarios, ctx)
                backward = fired_scenarios(reversed_scenarios, ctx)
                if forward != backward:
                    findings.append(
                        Finding(
                            "non-deterministic-firing",
                            ERROR,
                            ",".join(sorted(set(forward) ^ set(backward))),
                            f"For classes {sorted(present)} in room {room!r} the fired set "
                            f"depends on scenario order: {forward} vs {backward}.",
                        )
                    )
                    return findings  # One counterexample is enough.
    return findings


def check_conflicting_arbitration(scenarios: Sequence[Scenario]) -> list[Finding]:
    """Two scenarios that can fire together must not share a priority.

    Priority is the total order used for arbitration. Ties are resolved by id,
    which is stable but arbitrary — fine for unrelated scenarios, misleading
    when both genuinely fire on the same frame.
    """
    findings: list[Finding] = []
    for a, b in itertools.combinations(scenarios, 2):
        if a.priority != b.priority:
            continue
        if not (trigger_classes(a.trigger) & trigger_classes(b.trigger)):
            continue
        if a.risk_level == b.risk_level:
            continue
        findings.append(
            Finding(
                "ambiguous-arbitration",
                WARN,
                f"{a.scenario_id},{b.scenario_id}",
                f"{a.scenario_id} ({a.risk_level}) and {b.scenario_id} ({b.risk_level}) "
                f"share priority {a.priority} and overlapping classes, so which is spoken "
                f"first is decided by id order. Give them distinct priorities.",
            )
        )
    return findings


# ─── Orchestration ────────────────────────────────────────────────────────────


def run_all(
    scenarios: Sequence[Scenario],
    *,
    class_thresholds: Mapping[str, float] | None = None,
    global_confidence_floor: float = 0.25,
    memory_window_frames: int = 150,
    target_fps: float = 15.0,
    safety_classes: Iterable[str] = (),
) -> list[Finding]:
    """Run every report-level validator and return the findings."""
    runnable = [s for s in scenarios if s.status in (Status.ACTIVE, Status.DRAFT)]
    findings: list[Finding] = []
    findings += check_unobservable_assertions(runnable)
    findings += check_unanswerable_questions(runnable)
    findings += check_startle_language(runnable)
    findings += check_regulated_claims(runnable)
    findings += check_message_length(runnable)
    findings += check_locale_completeness(runnable)
    findings += check_confidence_reachable(
        runnable, class_thresholds or {}, global_confidence_floor
    )
    findings += check_temporal_within_memory(runnable, memory_window_frames, target_fps)
    findings += check_safety_class_coverage(runnable, safety_classes)
    findings += check_determinism(runnable)
    findings += check_conflicting_arbitration(runnable)
    return findings


def errors(findings: Sequence[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == ERROR]


def warnings(findings: Sequence[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == WARN]

"""
src.scenario_engine.compile — Scenario Compiler
===============================================

Globs ``configs/scenarios/SC-*.yaml``, validates each row and the set as a
whole, and emits the single artifact the runtime loads.

One artifact, not two
---------------------
An earlier draft emitted both a compiled JSON and a regenerated
``risk_rules.yaml``. Two generated artifacts can drift, so ``risk_rules.yaml`` is
retired at M8 and ``scenarios.compiled.json`` is the only output (ADR-P6-01/02).

What the artifact carries beyond the rows
-----------------------------------------
* ``rule_count`` and ``content_hash`` so the loader can **hard-fail** rather than
  run happily on an empty or truncated file. The legacy loader did
  ``data.get("rules", [])`` and logged ``Loaded 0 rules`` at INFO, so a compiler
  regression could have run for weeks looking healthy.
* the **class capability map**, because a taxonomy fingerprint provably cannot
  catch the scheduled ``wet_floor`` demotion (ADR-P6-05).
* an **inverted index** class -> scenario ids, so an edge device evaluates only
  the scenarios that touch a class it actually detected. The rule engine's
  budget is 5 ms/frame and spatial predicates are O(n²) over detections.

Set-level correctness
---------------------
A rule set's correctness is a property of the *set*, not of any row: duplicate
ids, deleted ids and conflicting arbitration are invisible in a one-file diff.
That is why the compiled artifact is committed and reviewed (ADR-P6-01).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import Detectability, Scenario, ScenarioError, Status, sorted_scenarios
from .taxonomy import CapabilityMap, build_capability_map
from .trigger import is_static_only, trigger_classes

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_ARTIFACT = Path("data/scenario_engine/build/scenarios.compiled.json")

#: Bumped when the artifact's *shape* changes, independently of the scenario
#: dataset's own semver.
ARTIFACT_SCHEMA_VERSION = 1


class CompileError(ValueError):
    """Raised when the scenario set cannot be compiled. Names the offender."""


@dataclass(frozen=True)
class CompileResult:
    """The compiled artifact plus the scenarios behind it."""

    artifact: dict[str, Any]
    scenarios: tuple[Scenario, ...]
    capability: CapabilityMap

    @property
    def active(self) -> tuple[Scenario, ...]:
        return tuple(s for s in self.scenarios if s.status is Status.ACTIVE)


# ─── Loading ──────────────────────────────────────────────────────────────────


def load_scenario_files(scenario_dir: Path | str = DEFAULT_SCENARIO_DIR) -> list[Scenario]:
    """Parse every ``SC-*.yaml`` under a directory, in filename order.

    Raises:
        CompileError: If the directory is missing, a file is unreadable, or a
            row fails schema validation. The error names the file.
    """
    directory = Path(scenario_dir)
    if not directory.is_dir():
        raise CompileError(
            f"Scenario directory not found: {directory}. Authored scenarios live in "
            f"configs/ (knowledge is source), not data/ (the DVC plane) — ADR-P6-01."
        )

    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("SC-*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CompileError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise CompileError(f"{path} must contain a single scenario mapping")
        try:
            scenarios.append(Scenario.from_mapping(raw, where=path.name))
        except ScenarioError as exc:
            raise CompileError(str(exc)) from exc

    return scenarios


# ─── Set-level validation ─────────────────────────────────────────────────────


def _check_unique_ids(scenarios: Sequence[Scenario]) -> None:
    seen: dict[str, int] = {}
    for scenario in scenarios:
        seen[scenario.scenario_id] = seen.get(scenario.scenario_id, 0) + 1
    duplicates = sorted(sid for sid, count in seen.items() if count > 1)
    if duplicates:
        raise CompileError(
            f"Duplicate scenario id(s) {duplicates}. Ids are immutable and never "
            f"reused, including after deprecation (ADR-P6-06)."
        )


def _check_classes(scenarios: Sequence[Scenario], capability: CapabilityMap) -> None:
    """Every referenced class must exist, be bbox-detectable, and be enabled."""
    for scenario in scenarios:
        if scenario.detectability is Detectability.REJECTED:
            continue  # A rejected row never runs; its classes need not be live.

        referenced = trigger_classes(scenario.trigger)

        unknown = capability.unknown(referenced)
        if unknown:
            raise CompileError(
                f"{scenario.scenario_id}: trigger references class(es) {unknown} that are "
                f"not in the taxonomy. Class names come from configs/data.yaml."
            )

        unusable = capability.unusable(referenced)
        if unusable:
            details = ", ".join(
                f"{c.name} (detection_level={c.detection_level}, enabled={c.enabled}"
                + (f", {c.reason}" if c.reason else "")
                + ")"
                for c in unusable
            )
            raise CompileError(
                f"{scenario.scenario_id}: trigger requires class(es) the deployed system "
                f"cannot supply: {details}. A scenario keyed on a demoted or disabled class "
                f"would compile green and never fire (ADR-P6-05)."
            )


def _check_not_static_only(scenarios: Sequence[Scenario]) -> None:
    """Reject triggers satisfiable by object presence alone."""
    for scenario in scenarios:
        if scenario.detectability is Detectability.REJECTED:
            continue
        if is_static_only(scenario.trigger):
            raise CompileError(
                f"{scenario.scenario_id}: trigger is satisfiable by object presence alone "
                f"({scenario.condition_text}). For a permanently-visible object that is a "
                f"metronome, not a hazard signal. Add a temporal, counting or spatial term "
                f"— dwell is nearly free for slow hazards and buys most of the precision."
            )


def _check_evidence_gates_risk(scenarios: Sequence[Scenario]) -> None:
    """``strength: none`` (or no evidence) forbids alarming."""
    for scenario in scenarios:
        if scenario.detectability is Detectability.REJECTED:
            continue
        strengths = {e.strength.value for e in scenario.evidence}
        well_founded = bool(strengths - {"none"})
        if well_founded:
            continue
        if scenario.risk_level in ("HIGH", "CRITICAL"):
            raise CompileError(
                f"{scenario.scenario_id}: risk_level {scenario.risk_level} requires at least "
                f"one evidence entry with strength other than 'none'. If it cannot be cited, "
                f"it cannot alarm."
            )
        if scenario.caregiver_channel.value in ("push", "push_and_call"):
            raise CompileError(
                f"{scenario.scenario_id}: caregiver_channel "
                f"'{scenario.caregiver_channel.value}' requires cited evidence."
            )


def _check_direct_detectability_is_earned(scenarios: Sequence[Scenario]) -> None:
    """``supports: hazard`` never licenses ``detectability: direct`` on its own."""
    for scenario in scenarios:
        if scenario.detectability is not Detectability.DIRECT or not scenario.evidence:
            continue
        supports = {e.supports.value for e in scenario.evidence}
        if supports == {"hazard"}:
            continue  # Permitted: the hazard is cited, the detection is ours to justify.
        unknown = supports - {"hazard", "intervention", "detection_method"}
        if unknown:  # pragma: no cover — enum-guarded upstream
            raise CompileError(f"{scenario.scenario_id}: unknown evidence.supports {unknown}")


def _check_no_rejected_scenarios_run(scenarios: Sequence[Scenario]) -> None:
    """A rejected row must not also claim to be active."""
    for scenario in scenarios:
        rejected = scenario.detectability is Detectability.REJECTED
        if rejected and scenario.status is Status.ACTIVE:
            raise CompileError(
                f"{scenario.scenario_id}: detectability 'rejected' cannot have status "
                f"'active'. Rejected rows stay in the repo as a versioned negative "
                f"register and never run (ADR-P6-08)."
            )


def validate_set(scenarios: Sequence[Scenario], capability: CapabilityMap) -> None:
    """Run every set-level check, raising on the first failure."""
    _check_unique_ids(scenarios)
    _check_no_rejected_scenarios_run(scenarios)
    _check_classes(scenarios, capability)
    _check_not_static_only(scenarios)
    _check_evidence_gates_risk(scenarios)
    _check_direct_detectability_is_earned(scenarios)


# ─── Artifact ─────────────────────────────────────────────────────────────────


def build_inverted_index(scenarios: Iterable[Scenario]) -> dict[str, list[str]]:
    """Map each class to the scenarios whose trigger mentions it.

    Lets the runtime skip every scenario that cannot possibly fire this frame,
    which is what keeps a few hundred scenarios inside the 5 ms rule budget.
    """
    index: dict[str, set[str]] = {}
    for scenario in scenarios:
        for class_name in trigger_classes(scenario.trigger):
            index.setdefault(class_name, set()).add(scenario.scenario_id)
    return {name: sorted(ids) for name, ids in sorted(index.items())}


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "schema_version": scenario.schema_version,
        "scenario_name": scenario.scenario_name,
        "category": scenario.category,
        "status": scenario.status.value,
        "detectability": scenario.detectability.value,
        "claim_class": scenario.claim_class.value,
        "risk_level": scenario.risk_level,
        "priority": scenario.priority,
        "condition": scenario.condition_text,
        "required_objects": list(scenario.required_objects),
        "optional_objects": list(scenario.optional_objects),
        "suppress_during": list(scenario.suppress_during),
        "room_context": scenario.room_context,
        "min_confidence": scenario.min_confidence,
        "min_dwell_seconds": scenario.min_dwell_seconds,
        "clear_after_seconds": scenario.clear_after_seconds,
        "cooldown_seconds": scenario.cooldown_seconds,
        "max_repeats": scenario.max_repeats,
        "max_per_day": scenario.max_per_day,
        "quiet_hours": {
            "start": scenario.quiet_hours.start,
            "end": scenario.quiet_hours.end,
            "behaviour": scenario.quiet_hours.behaviour.value,
        },
        "patient_facing": scenario.patient_facing,
        "messages": dict(sorted(scenario.messages.items())),
        "next_best_action": scenario.next_best_action,
        "caregiver_channel": scenario.caregiver_channel.value,
        "escalation": [
            {
                "after_seconds": step.after_seconds,
                "action": step.action,
                "messages": dict(sorted(step.messages.items())),
            }
            for step in scenario.escalation
        ],
        "evidence": [
            {
                "source_id": e.source_id,
                "section": e.section,
                "item": e.item,
                "strength": e.strength.value,
                "supports": e.supports.value,
                "note": e.note,
            }
            for e in scenario.evidence
        ],
        "capability_disclaimer": scenario.capability_disclaimer,
        "rejection_reason": scenario.rejection_reason,
        "reviewed_by": scenario.reviewed_by,
        "reviewed_on": scenario.reviewed_on,
        "rule_hash": scenario.rule_hash(),
    }


def _content_hash(payload: dict[str, Any]) -> str:
    """Digest over everything except the hash field itself."""
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compile_scenarios(
    scenario_dir: Path | str = DEFAULT_SCENARIO_DIR,
    capability: CapabilityMap | None = None,
    scenarios_version: str = "scenarios-v0.1.0",
) -> CompileResult:
    """Load, validate and assemble the compiled artifact.

    Args:
        scenario_dir:      Directory of authored ``SC-*.yaml`` files.
        capability:        Pre-built capability map; derived from the live config
            when omitted.
        scenarios_version: Semver of the scenario dataset itself, distinct from
            ``ARTIFACT_SCHEMA_VERSION``.

    Raises:
        CompileError: On any row-level or set-level failure.
    """
    capability = capability or build_capability_map()
    scenarios = tuple(sorted_scenarios(load_scenario_files(scenario_dir)))
    validate_set(scenarios, capability)

    runnable = tuple(
        s
        for s in scenarios
        if s.status in (Status.ACTIVE, Status.DRAFT)
        and s.detectability is not Detectability.REJECTED
    )
    deprecated = tuple(s for s in scenarios if s.status is Status.DEPRECATED)
    rejected = tuple(s for s in scenarios if s.detectability is Detectability.REJECTED)

    payload: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenarios_version": scenarios_version,
        "taxonomy_fingerprint": capability.fingerprint,
        "rule_count": len(runnable),
        "classes": capability.to_dict(),
        "inverted_index": build_inverted_index(runnable),
        "scenarios": [_scenario_to_dict(s) for s in runnable],
        "deprecated": [_scenario_to_dict(s) for s in deprecated],
        "rejected": [_scenario_to_dict(s) for s in rejected],
    }
    payload["content_hash"] = _content_hash(payload)

    return CompileResult(artifact=payload, scenarios=scenarios, capability=capability)


def write_artifact(artifact: dict[str, Any], path: Path | str = DEFAULT_ARTIFACT) -> Path:
    """Write the compiled artifact with byte-reproducible formatting.

    LF line endings and ``ensure_ascii=False``, matching
    ``src.utils.report_utils.save_json_report`` — DVC hashes the bytes on disk
    and ``.gitattributes`` normalises to LF, so anything else gives the same
    artifact two hashes across the CI matrix.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    return target


def verify_artifact(artifact: dict[str, Any]) -> None:
    """Fail closed on an artifact the runtime must not trust.

    Args:
        artifact: A loaded ``scenarios.compiled.json``.

    Raises:
        CompileError: On a missing/!= content hash, a rule-count mismatch, or an
            unsupported artifact schema version.
    """
    version = artifact.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise CompileError(
            f"Artifact schema version {version!r} is not supported "
            f"(expected {ARTIFACT_SCHEMA_VERSION})."
        )

    declared = artifact.get("content_hash")
    if not declared:
        raise CompileError("Artifact has no content_hash — refusing to load.")
    actual = _content_hash(artifact)
    if declared != actual:
        raise CompileError(
            f"Artifact content_hash mismatch: declared {declared}, computed {actual}. "
            f"The file has been edited by hand or truncated; recompile it."
        )

    scenarios = artifact.get("scenarios")
    if not isinstance(scenarios, list):
        raise CompileError("Artifact 'scenarios' must be a list.")
    if artifact.get("rule_count") != len(scenarios):
        raise CompileError(
            f"Artifact rule_count {artifact.get('rule_count')} does not match "
            f"{len(scenarios)} scenarios present."
        )

"""
src.scenario_engine.trigger — Structured Trigger Trees
======================================================

A scenario's trigger is authored as a nested YAML structure, validated against
the predicate registry, and compiled to an immutable tree the runtime walks.
The one-line string form is *emitted* for the CSV view and for logs, and is
never parsed back (ADR-P6-03).

Authored shape::

    trigger:
      all:
        - op: detected
          args: {class_name: wet_floor}
        - op: present_for
          args: {class_name: wet_floor, seconds: 3}
        - not:
            op: in_room
            args: {room: balcony}

Every error names the field path rather than a character offset::

    SC-KIT-001: trigger.all[1].args.seconds must be > 0 seconds, got -3

Kinds, not just truth values
----------------------------
:func:`trigger_kinds` reports which predicate kinds a trigger uses. A trigger
built only from ``static`` predicates is satisfiable by object presence alone
and is rejected by the compiler — the single check that disqualifies
``knife_near_person``, ``medicine_reminder`` and ``gas_cylinder_check`` at
authoring time rather than after they have generated ~370 alerts in a day.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .context import EvalContext
from .predicates import PredicateError, check_args, get_predicate, referenced_classes


class TriggerError(ValueError):
    """Raised when a trigger tree is malformed. Always names the field path."""


# ─── Tree ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Predicate:
    """A single registered predicate invocation with validated arguments."""

    op: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class AllOf:
    """Conjunction. Empty is rejected at parse time, never silently true."""

    nodes: tuple[Trigger, ...]


@dataclass(frozen=True)
class AnyOf:
    """Disjunction. Empty is rejected at parse time, never silently false."""

    nodes: tuple[Trigger, ...]


@dataclass(frozen=True)
class Not:
    """Negation. Composes with every predicate, unlike the legacy DSL."""

    node: Trigger


Trigger = Predicate | AllOf | AnyOf | Not

#: Keys that introduce a combinator rather than a predicate invocation.
_COMBINATORS = ("all", "any", "not")


# ─── Parsing ──────────────────────────────────────────────────────────────────


def parse_trigger(raw: Any, where: str = "trigger") -> Trigger:
    """Validate and compile an authored trigger structure.

    Args:
        raw:   The mapping loaded from a scenario's ``trigger:`` block.
        where: Field path used in error messages.

    Returns:
        The immutable trigger tree.

    Raises:
        TriggerError: On any malformed node, unknown predicate, or bad argument.
            Never returns a node that silently evaluates to a constant.
    """
    if not isinstance(raw, dict):
        raise TriggerError(f"{where} must be a mapping, got {type(raw).__name__}")

    present = [key for key in (*_COMBINATORS, "op") if key in raw]
    if not present:
        raise TriggerError(
            f"{where} must contain exactly one of 'op', 'all', 'any' or 'not'; "
            f"got keys {sorted(raw)}"
        )
    if len(present) > 1:
        raise TriggerError(f"{where} sets {present} — exactly one is allowed")

    key = present[0]

    if key == "not":
        return Not(parse_trigger(raw["not"], f"{where}.not"))

    if key in ("all", "any"):
        nodes_raw = raw[key]
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise TriggerError(
                f"{where}.{key} must be a non-empty list of trigger nodes, got {nodes_raw!r}"
            )
        nodes = tuple(
            parse_trigger(node, f"{where}.{key}[{i}]") for i, node in enumerate(nodes_raw)
        )
        return AllOf(nodes) if key == "all" else AnyOf(nodes)

    op = raw["op"]
    if not isinstance(op, str) or not op:
        raise TriggerError(f"{where}.op must be a non-empty string, got {op!r}")

    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise TriggerError(f"{where}.args must be a mapping, got {type(args).__name__}")

    try:
        spec = get_predicate(op)
        resolved = check_args(spec, args, where=where)
    except PredicateError as exc:
        raise TriggerError(str(exc)) from exc

    return Predicate(op=op, args=resolved)


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_trigger(node: Trigger, ctx: EvalContext) -> bool:
    """Evaluate a compiled trigger against one frame's context."""
    if isinstance(node, Predicate):
        return get_predicate(node.op).fn(ctx, node.args)
    if isinstance(node, AllOf):
        return all(evaluate_trigger(child, ctx) for child in node.nodes)
    if isinstance(node, AnyOf):
        return any(evaluate_trigger(child, ctx) for child in node.nodes)
    return not evaluate_trigger(node.node, ctx)


# ─── Introspection ────────────────────────────────────────────────────────────


def trigger_kinds(node: Trigger) -> frozenset[str]:
    """Predicate kinds used anywhere in a trigger.

    Drives the no-static-only-trigger rule: a trigger whose kinds are a subset
    of ``{"static"}`` is satisfiable by object presence alone.
    """
    if isinstance(node, Predicate):
        return frozenset({get_predicate(node.op).kind})
    if isinstance(node, Not):
        return trigger_kinds(node.node)
    kinds: set[str] = set()
    for child in node.nodes:
        kinds |= trigger_kinds(child)
    return frozenset(kinds)


def trigger_classes(node: Trigger) -> frozenset[str]:
    """Taxonomy class names referenced anywhere in a trigger.

    Feeds class resolution against configs/data.yaml and the capability map, so
    a scenario naming a demoted or disabled class fails the build rather than
    never firing (ADR-P6-05).
    """
    if isinstance(node, Predicate):
        return referenced_classes(get_predicate(node.op), node.args)
    if isinstance(node, Not):
        return trigger_classes(node.node)
    names: set[str] = set()
    for child in node.nodes:
        names |= trigger_classes(child)
    return frozenset(names)


def is_static_only(node: Trigger) -> bool:
    """Is this trigger satisfiable by object presence alone?

    Such a trigger fires on every frame the objects happen to be visible, which
    for a permanently-present object is a metronome, not a hazard signal.
    """
    return trigger_kinds(node) <= frozenset({"static"})


def render_trigger(node: Trigger) -> str:
    """Render a trigger as a one-line human-readable string.

    Emitted into the compiled artifact and the CSV view so both stay readable.
    **Never parsed back** — the structured tree is the only source of truth.
    """
    if isinstance(node, Predicate):
        spec = get_predicate(node.op)
        ordered = [*spec.required, *(a for a in sorted(node.args) if a not in spec.required)]
        rendered = ", ".join(f"{name}={node.args[name]!r}" for name in ordered if name in node.args)
        return f"{node.op}({rendered})"
    if isinstance(node, Not):
        return f"NOT {render_trigger(node.node)}"
    joiner = " AND " if isinstance(node, AllOf) else " OR "
    parts = [render_trigger(child) for child in node.nodes]
    body = joiner.join(parts)
    return body if len(parts) == 1 else f"({body})"

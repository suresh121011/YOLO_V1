"""
src.scenario_engine.predicates.registry — Predicate Registry
============================================================

Decorator-based predicate registry, fourth instance of the house pattern
(src/dataset/completeness_policies.py, src/dataset/splitting/registry.py,
src/dataset/annotation/registry.py).

Adding a predicate: write a pure function of ``(EvalContext, args)``, decorate
with ``@register_predicate(...)`` declaring its argument types and kind — the
compiler, the validators and the runtime evaluator need no changes.

Argument types are declarative so that one generic checker produces every error
message, and so those messages name the scenario field rather than a character
offset:

    SC-KIT-001.yaml: trigger.all[1].args.max_dist must be a fraction of frame
    width in (0, 1], got '0.2m'

That precision is the whole reason triggers are structured trees rather than
condition strings (ADR-P6-03).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..context import EvalContext

#: A predicate is a pure function of the frame context plus its own arguments.
PredicateFn = Callable[[EvalContext, Mapping[str, Any]], bool]

#: Argument value types the generic checker understands.
#:   class      — a taxonomy class name
#:   class_list — a non-empty list of taxonomy class names
#:   fraction   — a float in (0, 1]; frame-normalised distance or IoU
#:   seconds    — a positive number of seconds
#:   count      — a non-negative integer
#:   room       — a room name from the capture vocabulary
ARG_TYPES = frozenset({"class", "class_list", "fraction", "seconds", "count", "room"})

#: What a predicate contributes to a trigger. Used by the "no static-only
#: trigger" validator: a scenario satisfiable by object presence alone is
#: rejected, which is what disqualifies three of the six legacy rules at
#: authoring time (see negative_register.md).
KINDS = frozenset({"static", "counting", "temporal", "spatial", "context"})


class PredicateError(ValueError):
    """Raised for an unknown predicate or a malformed argument."""


@dataclass(frozen=True)
class PredicateSpec:
    """Everything the compiler needs to know about a predicate."""

    name: str
    fn: PredicateFn
    kind: str
    arg_types: Mapping[str, str]
    required: tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    doc: str = ""

    @property
    def optional(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.arg_types) - set(self.required)))

    def class_args(self) -> tuple[str, ...]:
        """Argument names that carry taxonomy class references."""
        return tuple(sorted(n for n, t in self.arg_types.items() if t in ("class", "class_list")))


_PREDICATES: dict[str, PredicateSpec] = {}


def register_predicate(
    name: str,
    *,
    kind: str,
    arg_types: Mapping[str, str],
    required: Sequence[str],
    defaults: Mapping[str, Any] | None = None,
) -> Callable[[PredicateFn], PredicateFn]:
    """Register a predicate implementation under a name.

    Args:
        name:      The ``op`` value used in a scenario's trigger tree.
        kind:      One of :data:`KINDS`.
        arg_types: Argument name -> type from :data:`ARG_TYPES`.
        required:  Argument names that must be present.
        defaults:  Values for optional arguments.

    Raises:
        PredicateError: If the name is already registered, or the declaration
            itself is malformed. Declaration errors surface at import time.
    """
    if name in _PREDICATES:
        raise PredicateError(
            f"Predicate '{name}' already registered by {_PREDICATES[name].fn.__name__}"
        )
    if kind not in KINDS:
        raise PredicateError(f"Predicate '{name}': unknown kind '{kind}'. Valid: {sorted(KINDS)}")
    for arg, arg_type in arg_types.items():
        if arg_type not in ARG_TYPES:
            raise PredicateError(
                f"Predicate '{name}': argument '{arg}' has unknown type "
                f"'{arg_type}'. Valid: {sorted(ARG_TYPES)}"
            )
    unknown_required = set(required) - set(arg_types)
    if unknown_required:
        raise PredicateError(
            f"Predicate '{name}': required argument(s) {sorted(unknown_required)} "
            f"are not declared in arg_types"
        )

    def _register(fn: PredicateFn) -> PredicateFn:
        _PREDICATES[name] = PredicateSpec(
            name=name,
            fn=fn,
            kind=kind,
            arg_types=dict(arg_types),
            required=tuple(required),
            defaults=dict(defaults or {}),
            doc=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
        )
        return fn

    return _register


def available_predicates() -> list[str]:
    """Return the sorted list of registered predicate names."""
    return sorted(_PREDICATES)


def get_predicate(name: str) -> PredicateSpec:
    """Look up a predicate spec by name.

    Raises:
        PredicateError: If the name is unknown, listing the registered ones.
            The legacy engine returned ``False`` for an unrecognised condition,
            so a typo produced a rule that silently never fired.
    """
    if name not in _PREDICATES:
        raise PredicateError(
            f"Unknown predicate '{name}'. Registered predicates: {available_predicates()}. "
            f"Register one via @register_predicate or fix the scenario's trigger."
        )
    return _PREDICATES[name]


# ─── Argument checking ────────────────────────────────────────────────────────


def _check_value(
    predicate: str,
    arg: str,
    arg_type: str,
    value: Any,
    where: str,
) -> None:
    """Validate one argument value, raising with the full field path."""
    prefix = f"{where}.args.{arg}" if where else f"{predicate}.{arg}"

    if arg_type in ("class", "room"):
        if not isinstance(value, str) or not value:
            raise PredicateError(f"{prefix} must be a non-empty string, got {value!r}")
        return

    if arg_type == "class_list":
        if not isinstance(value, list | tuple) or not value:
            raise PredicateError(f"{prefix} must be a non-empty list of class names, got {value!r}")
        for item in value:
            if not isinstance(item, str) or not item:
                raise PredicateError(f"{prefix} must contain only class names, got {item!r}")
        return

    if arg_type == "fraction":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PredicateError(
                f"{prefix} must be a fraction of the frame in (0, 1], got {value!r}. "
                f"Distances are frame-normalised, not pixels or metres (ADR-P6-07)."
            )
        if not 0 < float(value) <= 1:
            raise PredicateError(
                f"{prefix} must be in (0, 1], got {value}. Distances are frame-normalised."
            )
        return

    if arg_type == "seconds":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PredicateError(f"{prefix} must be a number of seconds, got {value!r}")
        if float(value) <= 0:
            raise PredicateError(f"{prefix} must be > 0 seconds, got {value}")
        return

    # count
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredicateError(f"{prefix} must be a whole number, got {value!r}")
    if value < 0:
        raise PredicateError(f"{prefix} must be >= 0, got {value}")


def check_args(
    spec: PredicateSpec,
    args: Mapping[str, Any],
    where: str = "",
) -> dict[str, Any]:
    """Validate a predicate's arguments and apply defaults.

    Args:
        spec:  The predicate being invoked.
        args:  Raw argument mapping from the scenario YAML.
        where: Field path for error messages, e.g. ``trigger.all[1]``.

    Returns:
        The argument mapping with defaults filled in.

    Raises:
        PredicateError: On a missing required argument, an unknown argument, or
            a value that fails its declared type.
    """
    missing = sorted(set(spec.required) - set(args))
    if missing:
        raise PredicateError(
            f"{where or spec.name}: predicate '{spec.name}' is missing required "
            f"argument(s) {missing}. Required: {list(spec.required)}"
        )

    unknown = sorted(set(args) - set(spec.arg_types))
    if unknown:
        raise PredicateError(
            f"{where or spec.name}: predicate '{spec.name}' got unknown argument(s) "
            f"{unknown}. Accepted: {sorted(spec.arg_types)}"
        )

    resolved = {**spec.defaults, **args}
    for arg, value in resolved.items():
        _check_value(spec.name, arg, spec.arg_types[arg], value, where)
    return resolved


def referenced_classes(spec: PredicateSpec, args: Mapping[str, Any]) -> frozenset[str]:
    """Taxonomy class names an invocation refers to.

    Feeds the compiler's class-resolution and capability checks, so a scenario
    requiring a demoted or disabled class fails the build instead of never
    firing (ADR-P6-05).
    """
    names: set[str] = set()
    for arg in spec.class_args():
        value = args.get(arg)
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, list | tuple):
            names.update(str(item) for item in value)
    return frozenset(names)

"""
src.scenario_engine.predicates — Predicate Registry and Vocabulary
==================================================================

Importing this package registers every built-in predicate, so
``available_predicates()`` is populated for the compiler, the validators and the
error messages that enumerate valid names.
"""

from __future__ import annotations

from . import builtins as _builtins  # noqa: F401 — import registers the built-ins
from .registry import (
    ARG_TYPES,
    KINDS,
    PredicateError,
    PredicateFn,
    PredicateSpec,
    available_predicates,
    check_args,
    get_predicate,
    referenced_classes,
    register_predicate,
)

__all__ = [
    "ARG_TYPES",
    "KINDS",
    "PredicateError",
    "PredicateFn",
    "PredicateSpec",
    "available_predicates",
    "check_args",
    "get_predicate",
    "referenced_classes",
    "register_predicate",
]

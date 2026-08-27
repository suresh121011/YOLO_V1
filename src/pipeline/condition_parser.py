"""
Condition Parser — Tokenizer, Recursive-Descent Parser, and AST Evaluator
=========================================================================
Parses the ``configs/risk_rules.yaml`` condition DSL into a small immutable AST
and evaluates it against the current frame's detections plus Event Memory.

Why this module exists
----------------------
The previous inline evaluator in ``rule_engine.py`` used ``re.search`` plus
``str.split`` and was wrong in four ways (see
``docs/08_scenario_engineering/architecture_review.md`` §3, defect D6):

1. ``any_of([...])`` was matched across the *whole* condition and returned
   immediately, so ``any_of([a, b]) AND detected(person)`` silently discarded
   the conjunct.
2. AND was split before OR, giving AND *lower* precedence than OR — inverted
   from standard boolean semantics.
3. Parentheses were unsupported; a leading ``(`` defeated every anchored
   predicate regex and the condition degraded to ``False``.
4. ``NOT`` composed only with ``detected``.

It also failed **silently**: an unrecognised condition logged a warning and
returned ``False``, so a malformed CRITICAL rule simply never fired.

Parsing happens once at rule-load time, not once per frame. That makes a
malformed condition a loud load-time failure instead of a permanent silent
false, and keeps the per-frame cost inside the 5 ms rule-engine budget in
``docs/02_technical_architecture_specification/performance_budget.md``.

Grammar
-------
::

    expr      := or_expr
    or_expr   := and_expr ( "OR" and_expr )*
    and_expr  := unary ( "AND" unary )*
    unary     := "NOT" unary | primary
    primary   := "(" expr ")" | predicate
    predicate := detected( name )
               | absent_for( name , number )
               | any_of( [ name , ... ] )

``AND`` binds tighter than ``OR``, as in every other boolean language.

Scope note
----------
This module deliberately supports **only** the predicates the shipped rule set
already uses. New predicates (``near``, ``count``, ``present_for``, …) belong to
the structured predicate trees of ``src/scenario_engine`` — see
``docs/08_scenario_engineering/adr/ADR-P6-03-structured-ast-over-string-dsl.md``.
This file is retired together with the legacy string DSL at milestone M8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ─── Errors ───────────────────────────────────────────────────────────────────


class ConditionSyntaxError(ValueError):
    """Raised when a rule condition cannot be parsed.

    Always names the offending condition and the position, because these
    surface at rule-load time where the operator can act on them.
    """


# ─── Memory protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class AbsenceQueryable(Protocol):
    """The single Event Memory capability the DSL needs.

    Declared as a Protocol so this module does not import ``EventMemory`` and
    stays trivially testable with a stub.
    """

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        """Has ``class_name`` been continuously absent for at least ``seconds``?"""
        ...


# ─── AST ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detected:
    """``detected(class_name)`` — class present in the current frame."""

    class_name: str


@dataclass(frozen=True)
class AbsentFor:
    """``absent_for(class_name, seconds)`` — class absent for >= N seconds."""

    class_name: str
    seconds: float


@dataclass(frozen=True)
class AnyOf:
    """``any_of([a, b, ...])`` — any listed class present in the current frame."""

    class_names: tuple[str, ...]


@dataclass(frozen=True)
class Not:
    """Logical negation."""

    operand: Node


@dataclass(frozen=True)
class And:
    """Logical conjunction over two or more operands."""

    operands: tuple[Node, ...]


@dataclass(frozen=True)
class Or:
    """Logical disjunction over two or more operands."""

    operands: tuple[Node, ...]


Node = Detected | AbsentFor | AnyOf | Not | And | Or

#: Predicate names this parser accepts. Anything else is a load-time error.
KNOWN_PREDICATES: frozenset[str] = frozenset({"detected", "absent_for", "any_of"})


# ─── Tokenizer ────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<punct>[()\[\],])
    """,
    re.VERBOSE,
)

_KEYWORDS = frozenset({"AND", "OR", "NOT"})


@dataclass(frozen=True)
class _Token:
    kind: str  # "number" | "ident" | "keyword" | "punct" | "end"
    value: str
    pos: int


def _tokenize(condition: str) -> list[_Token]:
    """Split a condition string into tokens, rejecting unknown characters."""
    tokens: list[_Token] = []
    index = 0
    length = len(condition)

    while index < length:
        match = _TOKEN_RE.match(condition, index)
        if match is None:
            raise ConditionSyntaxError(
                f"unexpected character {condition[index]!r} at position {index} "
                f"in condition {condition!r}"
            )
        index = match.end()

        if match.lastgroup == "ws":
            continue
        value = match.group()
        if match.lastgroup == "ident":
            kind = "keyword" if value in _KEYWORDS else "ident"
        else:
            kind = match.lastgroup or "punct"
        tokens.append(_Token(kind=kind, value=value, pos=match.start()))

    tokens.append(_Token(kind="end", value="", pos=length))
    return tokens


# ─── Parser ───────────────────────────────────────────────────────────────────


class _Parser:
    """Recursive-descent parser for the condition grammar."""

    def __init__(self, condition: str) -> None:
        self._condition = condition
        self._tokens = _tokenize(condition)
        self._i = 0

    # -- token helpers --

    @property
    def _current(self) -> _Token:
        return self._tokens[self._i]

    def _advance(self) -> _Token:
        token = self._tokens[self._i]
        self._i += 1
        return token

    def _expect(self, kind: str, value: str | None = None) -> _Token:
        token = self._current
        if token.kind != kind or (value is not None and token.value != value):
            wanted = value if value is not None else kind
            found = token.value or "end of condition"
            raise ConditionSyntaxError(
                f"expected {wanted!r} but found {found!r} at position {token.pos} "
                f"in condition {self._condition!r}"
            )
        return self._advance()

    # -- grammar --

    def parse(self) -> Node:
        """Parse the full condition and assert the input is fully consumed."""
        if self._current.kind == "end":
            raise ConditionSyntaxError(f"empty condition: {self._condition!r}")
        node = self._parse_or()
        if self._current.kind != "end":
            token = self._current
            raise ConditionSyntaxError(
                f"unexpected trailing token {token.value!r} at position {token.pos} "
                f"in condition {self._condition!r}"
            )
        return node

    def _parse_or(self) -> Node:
        operands = [self._parse_and()]
        while self._current.kind == "keyword" and self._current.value == "OR":
            self._advance()
            operands.append(self._parse_and())
        return operands[0] if len(operands) == 1 else Or(tuple(operands))

    def _parse_and(self) -> Node:
        operands = [self._parse_unary()]
        while self._current.kind == "keyword" and self._current.value == "AND":
            self._advance()
            operands.append(self._parse_unary())
        return operands[0] if len(operands) == 1 else And(tuple(operands))

    def _parse_unary(self) -> Node:
        if self._current.kind == "keyword" and self._current.value == "NOT":
            self._advance()
            return Not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        if self._current.kind == "punct" and self._current.value == "(":
            self._advance()
            node = self._parse_or()
            self._expect("punct", ")")
            return node
        return self._parse_predicate()

    def _parse_predicate(self) -> Node:
        name_token = self._expect("ident")
        name = name_token.value
        if name not in KNOWN_PREDICATES:
            valid = ", ".join(sorted(KNOWN_PREDICATES))
            raise ConditionSyntaxError(
                f"unknown predicate {name!r} at position {name_token.pos} "
                f"in condition {self._condition!r}. Valid predicates: {valid}"
            )
        self._expect("punct", "(")

        if name == "detected":
            class_name = self._expect("ident").value
            self._expect("punct", ")")
            return Detected(class_name)

        if name == "absent_for":
            class_name = self._expect("ident").value
            self._expect("punct", ",")
            seconds = float(self._expect("number").value)
            self._expect("punct", ")")
            return AbsentFor(class_name, seconds)

        # any_of([a, b, ...])
        self._expect("punct", "[")
        names: list[str] = [self._expect("ident").value]
        while self._current.kind == "punct" and self._current.value == ",":
            self._advance()
            names.append(self._expect("ident").value)
        self._expect("punct", "]")
        self._expect("punct", ")")
        return AnyOf(tuple(names))


def parse_condition(condition: str) -> Node:
    """Parse a rule condition string into an AST.

    Args:
        condition: A condition such as ``"detected(stove) AND absent_for(person, 30)"``.

    Returns:
        The root AST node.

    Raises:
        ConditionSyntaxError: If the condition is empty, malformed, references an
            unknown predicate, or carries trailing tokens. Never returns a node
            that silently evaluates to ``False``.
    """
    return _Parser(condition.strip()).parse()


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_node(
    node: Node,
    detected_names: frozenset[str] | set[str],
    memory: AbsenceQueryable,
    fps: float,
) -> bool:
    """Evaluate a parsed condition against the current frame state.

    Args:
        node: Root of a parsed condition AST.
        detected_names: Class names present in the current frame.
        memory: Event Memory (anything satisfying ``AbsenceQueryable``).
        fps: Frame rate used to convert the memory's frame counts to seconds.
            Pass the **measured** rate, not the nominal target — otherwise
            thermal throttling silently rescales every temporal threshold.

    Returns:
        Whether the condition holds.
    """
    if isinstance(node, Detected):
        return node.class_name in detected_names
    if isinstance(node, AnyOf):
        return any(name in detected_names for name in node.class_names)
    if isinstance(node, AbsentFor):
        return memory.is_absent_for_by_name(node.class_name, node.seconds, fps)
    if isinstance(node, Not):
        return not evaluate_node(node.operand, detected_names, memory, fps)
    if isinstance(node, And):
        return all(evaluate_node(o, detected_names, memory, fps) for o in node.operands)
    return any(evaluate_node(o, detected_names, memory, fps) for o in node.operands)


def referenced_classes(node: Node) -> frozenset[str]:
    """Every class name a condition refers to.

    Used by the rule loader to validate conditions against the live taxonomy,
    so a typo like ``detected(knive)`` fails at load rather than never firing.
    """
    if isinstance(node, Detected | AbsentFor):
        return frozenset({node.class_name})
    if isinstance(node, AnyOf):
        return frozenset(node.class_names)
    if isinstance(node, Not):
        return referenced_classes(node.operand)
    names: set[str] = set()
    for operand in node.operands:
        names |= referenced_classes(operand)
    return frozenset(names)

"""
Unit tests for src.pipeline.condition_parser.

Covers the grammar, the four defects the previous regex evaluator had
(docs/08_scenario_engineering/architecture_review.md §3, D6), and the
fail-loud contract that replaces its silent ``False``.
"""

from __future__ import annotations

import pytest

from src.pipeline.condition_parser import (
    AbsentFor,
    And,
    AnyOf,
    ConditionSyntaxError,
    Detected,
    Not,
    Or,
    evaluate_node,
    parse_condition,
    referenced_classes,
)


class _StubMemory:
    """Minimal AbsenceQueryable: classes in ``absent`` are absent long enough."""

    def __init__(self, absent: set[str] | None = None) -> None:
        self._absent = absent or set()
        self.calls: list[tuple[str, float, float]] = []

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        self.calls.append((class_name, seconds, fps))
        return class_name in self._absent


def _eval(condition: str, present: set[str], absent: set[str] | None = None) -> bool:
    return evaluate_node(parse_condition(condition), present, _StubMemory(absent), 15.0)


class TestParsing:
    """Grammar and AST shape."""

    @pytest.mark.unit
    def test_detected(self) -> None:
        assert parse_condition("detected(knife)") == Detected("knife")

    @pytest.mark.unit
    def test_absent_for_accepts_int_and_float(self) -> None:
        assert parse_condition("absent_for(person, 30)") == AbsentFor("person", 30.0)
        assert parse_condition("absent_for(person, 2.5)") == AbsentFor("person", 2.5)

    @pytest.mark.unit
    def test_any_of(self) -> None:
        node = parse_condition("any_of([medicine_strip, medicine_bottle])")
        assert node == AnyOf(("medicine_strip", "medicine_bottle"))

    @pytest.mark.unit
    def test_not_composes_with_any_predicate(self) -> None:
        """D6.4: NOT previously composed only with detected()."""
        assert parse_condition("NOT any_of([knife])") == Not(AnyOf(("knife",)))
        assert parse_condition("NOT absent_for(person, 5)") == Not(AbsentFor("person", 5.0))
        assert parse_condition("NOT NOT detected(knife)") == Not(Not(Detected("knife")))

    @pytest.mark.unit
    def test_and_binds_tighter_than_or(self) -> None:
        """D6.2: the old splitter gave AND *lower* precedence than OR."""
        node = parse_condition("detected(a) OR detected(b) AND detected(c)")
        assert node == Or((Detected("a"), And((Detected("b"), Detected("c")))))

    @pytest.mark.unit
    def test_parentheses_override_precedence(self) -> None:
        """D6.3: parentheses were unsupported and degraded the rule to False."""
        node = parse_condition("(detected(a) OR detected(b)) AND detected(c)")
        assert node == And((Or((Detected("a"), Detected("b"))), Detected("c")))

    @pytest.mark.unit
    def test_n_ary_flattening(self) -> None:
        node = parse_condition("detected(a) AND detected(b) AND detected(c)")
        assert node == And((Detected("a"), Detected("b"), Detected("c")))

    @pytest.mark.unit
    def test_whitespace_is_insignificant(self) -> None:
        assert parse_condition("  detected( knife )  ") == Detected("knife")


class TestParseErrors:
    """Malformed conditions must fail loudly, never evaluate to False."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "condition",
        [
            "",
            "   ",
            "detected(knife) AND",
            "detected(knife) AND_typo detected(person)",
            "(detected(knife)",
            "detected(knife))",
            "detected()",
            "absent_for(person)",
            "absent_for(person, thirty)",
            "any_of(knife)",
            "any_of([])",
            "detected(knife) detected(person)",
            "detected(knife) @ detected(person)",
        ],
    )
    def test_malformed_conditions_raise(self, condition: str) -> None:
        with pytest.raises(ConditionSyntaxError):
            parse_condition(condition)

    @pytest.mark.unit
    def test_unknown_predicate_names_the_valid_set(self) -> None:
        with pytest.raises(ConditionSyntaxError) as excinfo:
            parse_condition("near(person, stove, 0.2)")
        message = str(excinfo.value)
        assert "near" in message
        assert "detected" in message and "absent_for" in message and "any_of" in message

    @pytest.mark.unit
    def test_trailing_garbage_is_not_silently_ignored(self) -> None:
        """The old regexes were unanchored at the end, so typos became valid rules."""
        with pytest.raises(ConditionSyntaxError):
            parse_condition("detected(knife) AND_typo detected(person)")


class TestEvaluation:
    """Semantics against a frame's detected class names."""

    @pytest.mark.unit
    def test_detected_and_not(self) -> None:
        assert _eval("detected(knife)", {"knife"}) is True
        assert _eval("detected(knife)", {"person"}) is False
        assert _eval("NOT detected(knife)", {"person"}) is True

    @pytest.mark.unit
    def test_any_of_with_conjunct_is_not_discarded(self) -> None:
        """D6.1: the old evaluator returned on any_of and dropped the AND."""
        condition = "any_of([medicine_strip, medicine_bottle]) AND detected(person)"
        assert _eval(condition, {"medicine_strip", "person"}) is True
        assert _eval(condition, {"medicine_strip"}) is False, "conjunct was discarded"

    @pytest.mark.unit
    def test_precedence_evaluates_as_parsed(self) -> None:
        condition = "detected(a) OR detected(b) AND detected(c)"
        assert _eval(condition, {"a"}) is True
        assert _eval(condition, {"b"}) is False
        assert _eval(condition, {"b", "c"}) is True

    @pytest.mark.unit
    def test_absent_for_consults_memory_with_given_fps(self) -> None:
        memory = _StubMemory(absent={"person"})
        node = parse_condition("absent_for(person, 30)")
        assert evaluate_node(node, set(), memory, 7.5) is True
        assert memory.calls == [("person", 30.0, 7.5)]

    @pytest.mark.unit
    def test_shipped_stove_rule(self) -> None:
        condition = "detected(stove) AND absent_for(person, 30)"
        assert _eval(condition, {"stove"}, absent={"person"}) is True
        assert _eval(condition, {"stove"}, absent=set()) is False
        assert _eval(condition, set(), absent={"person"}) is False


class TestReferencedClasses:
    """Class references drive load-time taxonomy validation."""

    @pytest.mark.unit
    def test_collects_from_every_node_type(self) -> None:
        node = parse_condition(
            "(detected(stove) OR any_of([knife, sink])) AND NOT absent_for(person, 5)"
        )
        assert referenced_classes(node) == frozenset({"stove", "knife", "sink", "person"})

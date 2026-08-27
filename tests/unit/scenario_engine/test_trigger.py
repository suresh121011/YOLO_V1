"""
Unit tests for src.scenario_engine.trigger.

Covers parsing, evaluation, and the introspection the compiler's validators
depend on — in particular `is_static_only`, the check that disqualifies the
three legacy rules being deleted rather than migrated.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.pipeline import BoundingBox, Detection
from src.scenario_engine import EvalContext
from src.scenario_engine.trigger import (
    AllOf,
    Not,
    Predicate,
    TriggerError,
    evaluate_trigger,
    is_static_only,
    parse_trigger,
    render_trigger,
    trigger_classes,
    trigger_kinds,
)


class _StubMemory:
    def __init__(self, absent: set[str] | None = None) -> None:
        self._absent = absent or set()

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        return class_name in self._absent

    def consecutive_frames(self, class_id: int) -> int:
        return 0

    def has_ever_seen(self, class_name: str) -> bool:
        return True


def _det(class_name: str, cx: float = 0.5) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(cx=cx, cy=0.5, w=0.1, h=0.1),
        frame_id=1,
        timestamp_ms=0.0,
    )


def _ctx(*detections: Detection, absent: set[str] | None = None, room: str | None = None):
    return EvalContext.from_frame(list(detections), _StubMemory(absent), 15.0, room=room)


_DETECTED_KNIFE: dict[str, Any] = {"op": "detected", "args": {"class_name": "knife"}}
_DETECTED_PERSON: dict[str, Any] = {"op": "detected", "args": {"class_name": "person"}}


class TestParsing:
    @pytest.mark.unit
    def test_single_predicate(self) -> None:
        node = parse_trigger(_DETECTED_KNIFE)
        assert node == Predicate(op="detected", args={"class_name": "knife"})

    @pytest.mark.unit
    def test_all_and_any(self) -> None:
        node = parse_trigger({"all": [_DETECTED_KNIFE, _DETECTED_PERSON]})
        assert isinstance(node, AllOf)
        assert len(node.nodes) == 2

    @pytest.mark.unit
    def test_not_composes_with_any_predicate(self) -> None:
        node = parse_trigger(
            {"not": {"op": "absent_for", "args": {"class_name": "p", "seconds": 5}}}
        )
        assert isinstance(node, Not)

    @pytest.mark.unit
    def test_defaults_are_resolved_at_parse_time(self) -> None:
        node = parse_trigger({"op": "count", "args": {"class_name": "person"}})
        assert isinstance(node, Predicate)
        assert node.args["at_least"] == 1

    @pytest.mark.unit
    def test_nesting(self) -> None:
        node = parse_trigger(
            {
                "all": [
                    {"any": [_DETECTED_KNIFE, _DETECTED_PERSON]},
                    {"not": _DETECTED_PERSON},
                ]
            }
        )
        assert isinstance(node, AllOf)


class TestParseErrors:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw",
        [
            {},
            {"all": []},
            {"any": []},
            {"all": "not-a-list"},
            {"op": ""},
            {"op": 3},
            {"op": "detected", "args": "not-a-mapping"},
            "not-a-mapping",
            {"op": "detected", "all": [_DETECTED_KNIFE]},
        ],
    )
    def test_malformed_nodes_raise(self, raw: Any) -> None:
        with pytest.raises(TriggerError):
            parse_trigger(raw)

    @pytest.mark.unit
    def test_empty_all_is_not_silently_true(self) -> None:
        with pytest.raises(TriggerError, match="non-empty"):
            parse_trigger({"all": []})

    @pytest.mark.unit
    def test_unknown_predicate_names_the_valid_set(self) -> None:
        with pytest.raises(TriggerError) as excinfo:
            parse_trigger({"op": "teleports", "args": {}})
        assert "teleports" in str(excinfo.value)
        assert "detected" in str(excinfo.value)

    @pytest.mark.unit
    def test_error_carries_the_nested_field_path(self) -> None:
        with pytest.raises(TriggerError) as excinfo:
            parse_trigger(
                {
                    "all": [
                        _DETECTED_KNIFE,
                        {
                            "op": "near",
                            "args": {"class_a": "person", "class_b": "stove", "max_dist": 4},
                        },
                    ]
                }
            )
        assert "trigger.all[1].args.max_dist" in str(excinfo.value)


class TestEvaluation:
    @pytest.mark.unit
    def test_all_requires_every_child(self) -> None:
        node = parse_trigger({"all": [_DETECTED_KNIFE, _DETECTED_PERSON]})
        assert evaluate_trigger(node, _ctx(_det("knife"), _det("person"))) is True
        assert evaluate_trigger(node, _ctx(_det("knife"))) is False

    @pytest.mark.unit
    def test_any_requires_one_child(self) -> None:
        node = parse_trigger({"any": [_DETECTED_KNIFE, _DETECTED_PERSON]})
        assert evaluate_trigger(node, _ctx(_det("person"))) is True
        assert evaluate_trigger(node, _ctx(_det("stove"))) is False

    @pytest.mark.unit
    def test_not_inverts(self) -> None:
        node = parse_trigger({"not": _DETECTED_KNIFE})
        assert evaluate_trigger(node, _ctx(_det("person"))) is True
        assert evaluate_trigger(node, _ctx(_det("knife"))) is False

    @pytest.mark.unit
    def test_precedence_is_explicit_in_the_structure(self) -> None:
        """Nesting removes the ambiguity the string DSL got backwards."""
        node = parse_trigger(
            {
                "all": [
                    {"any": [_DETECTED_KNIFE, _DETECTED_PERSON]},
                    {"op": "detected", "args": {"class_name": "stove"}},
                ]
            }
        )
        assert evaluate_trigger(node, _ctx(_det("knife"), _det("stove"))) is True
        assert evaluate_trigger(node, _ctx(_det("knife"))) is False

    @pytest.mark.unit
    def test_realistic_unattended_stove_trigger(self) -> None:
        node = parse_trigger(
            {
                "all": [
                    {"op": "detected", "args": {"class_name": "stove"}},
                    {"op": "absent_for", "args": {"class_name": "person", "seconds": 120}},
                    {"op": "in_room", "args": {"room": "kitchen"}},
                ]
            }
        )
        fires = _ctx(_det("stove"), absent={"person"}, room="kitchen")
        wrong_room = _ctx(_det("stove"), absent={"person"}, room="hall")
        person_there = _ctx(_det("stove"), _det("person"), room="kitchen")
        assert evaluate_trigger(node, fires) is True
        assert evaluate_trigger(node, wrong_room) is False
        assert evaluate_trigger(node, person_there) is False


class TestIntrospection:
    @pytest.mark.unit
    def test_kinds_are_collected_across_the_tree(self) -> None:
        node = parse_trigger(
            {
                "all": [
                    _DETECTED_KNIFE,
                    {"op": "absent_for", "args": {"class_name": "person", "seconds": 30}},
                ]
            }
        )
        assert trigger_kinds(node) == frozenset({"static", "temporal"})

    @pytest.mark.unit
    def test_static_only_detects_the_deleted_legacy_rules(self) -> None:
        """`detected(knife) AND detected(person)` — the cooking metronome."""
        knife_near_person = parse_trigger({"all": [_DETECTED_KNIFE, _DETECTED_PERSON]})
        medicine_reminder = parse_trigger(
            {"op": "any_of", "args": {"class_names": ["medicine_strip", "medicine_bottle"]}}
        )
        gas_cylinder_check = parse_trigger(
            {
                "all": [
                    {"op": "detected", "args": {"class_name": "gas_cylinder"}},
                    {"not": {"op": "detected", "args": {"class_name": "stove"}}},
                ]
            }
        )
        assert is_static_only(knife_near_person) is True
        assert is_static_only(medicine_reminder) is True
        assert is_static_only(gas_cylinder_check) is True

    @pytest.mark.unit
    def test_a_temporal_or_spatial_term_lifts_static_only(self) -> None:
        with_dwell = parse_trigger(
            {
                "all": [
                    {"op": "detected", "args": {"class_name": "wet_floor"}},
                    {"op": "present_for", "args": {"class_name": "wet_floor", "seconds": 3}},
                ]
            }
        )
        with_space = parse_trigger(
            {
                "all": [
                    _DETECTED_PERSON,
                    {
                        "op": "near",
                        "args": {"class_a": "person", "class_b": "stove", "max_dist": 0.2},
                    },
                ]
            }
        )
        assert is_static_only(with_dwell) is False
        assert is_static_only(with_space) is False

    @pytest.mark.unit
    def test_classes_are_collected_including_lists_and_negations(self) -> None:
        node = parse_trigger(
            {
                "all": [
                    {"op": "any_of", "args": {"class_names": ["knife", "sink"]}},
                    {"not": {"op": "absent_for", "args": {"class_name": "person", "seconds": 5}}},
                    {
                        "op": "near",
                        "args": {"class_a": "walking_stick", "class_b": "bed", "max_dist": 0.3},
                    },
                ]
            }
        )
        assert trigger_classes(node) == frozenset(
            {"knife", "sink", "person", "walking_stick", "bed"}
        )


class TestRendering:
    @pytest.mark.unit
    def test_predicate_renders_required_args_first(self) -> None:
        node = parse_trigger(
            {"op": "near", "args": {"max_dist": 0.2, "class_a": "person", "class_b": "stove"}}
        )
        assert render_trigger(node) == "near(class_a='person', class_b='stove', max_dist=0.2)"

    @pytest.mark.unit
    def test_combinators_render_with_explicit_parentheses(self) -> None:
        node = parse_trigger({"all": [_DETECTED_KNIFE, {"not": _DETECTED_PERSON}]})
        rendered = render_trigger(node)
        assert rendered.startswith("(") and rendered.endswith(")")
        assert " AND " in rendered
        assert "NOT detected(class_name='person')" in rendered

    @pytest.mark.unit
    def test_rendering_is_deterministic(self) -> None:
        raw = {
            "all": [
                {"op": "count", "args": {"class_name": "person", "at_most": 1, "at_least": 1}},
                _DETECTED_KNIFE,
            ]
        }
        assert render_trigger(parse_trigger(raw)) == render_trigger(parse_trigger(raw))

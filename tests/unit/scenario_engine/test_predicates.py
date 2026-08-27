"""
Unit tests for src.scenario_engine.predicates.

Two things these must prove, because both failure modes are silent:

1. `near` genuinely uses geometry. A spatial predicate implemented against the
   legacy `set[str]` signature degrades into plain co-presence — identical to
   the `knife_near_person` rule being deleted for firing throughout normal
   cooking — while looking implemented (ADR-P6-03).
2. An unknown predicate or a malformed argument is a loud error naming the
   field, not a rule that silently never fires.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.pipeline import BoundingBox, Detection
from src.scenario_engine import EvalContext
from src.scenario_engine.predicates import (
    PredicateError,
    available_predicates,
    check_args,
    get_predicate,
    referenced_classes,
    register_predicate,
)


class _StubMemory:
    def __init__(
        self,
        absent: set[str] | None = None,
        ever_seen: set[str] | None = None,
        consecutive: dict[int, int] | None = None,
    ) -> None:
        self._absent = absent or set()
        self._ever_seen = ever_seen or set()
        self._consecutive = consecutive or {}

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        return class_name in self._absent

    def consecutive_frames(self, class_id: int) -> int:
        return self._consecutive.get(class_id, 0)

    def has_ever_seen(self, class_name: str) -> bool:
        return class_name in self._ever_seen


def _det(
    class_name: str, cx: float = 0.5, cy: float = 0.5, class_id: int = 0, size: float = 0.1
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(cx=cx, cy=cy, w=size, h=size),
        frame_id=1,
        timestamp_ms=0.0,
    )


def _ctx(*detections: Detection, memory: Any = None, fps: float = 15.0, room: str | None = None):
    return EvalContext.from_frame(list(detections), memory or _StubMemory(), fps, room=room)


def _call(name: str, ctx: EvalContext, **args: Any) -> bool:
    spec = get_predicate(name)
    return spec.fn(ctx, check_args(spec, args))


# ─── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    @pytest.mark.unit
    def test_builtins_are_registered(self) -> None:
        for name in ("detected", "near", "overlaps", "count", "absent_for", "present_for"):
            assert name in available_predicates()

    @pytest.mark.unit
    def test_unknown_predicate_names_the_valid_set(self) -> None:
        with pytest.raises(PredicateError) as excinfo:
            get_predicate("teleports")
        message = str(excinfo.value)
        assert "teleports" in message
        assert "near" in message and "detected" in message

    @pytest.mark.unit
    def test_duplicate_registration_is_rejected(self) -> None:
        with pytest.raises(PredicateError, match="already registered"):
            register_predicate(
                "detected", kind="static", arg_types={"class_name": "class"}, required=()
            )

    @pytest.mark.unit
    def test_bad_declaration_fails_at_registration(self) -> None:
        with pytest.raises(PredicateError, match="unknown kind"):
            register_predicate("x1", kind="magic", arg_types={}, required=())
        with pytest.raises(PredicateError, match="unknown type"):
            register_predicate("x2", kind="static", arg_types={"a": "wombat"}, required=())
        with pytest.raises(PredicateError, match="not declared"):
            register_predicate("x3", kind="static", arg_types={}, required=("a",))

    @pytest.mark.unit
    def test_kinds_distinguish_static_from_the_rest(self) -> None:
        """The no-static-only-trigger validator depends on this partition."""
        assert get_predicate("detected").kind == "static"
        assert get_predicate("any_of").kind == "static"
        assert get_predicate("near").kind == "spatial"
        assert get_predicate("absent_for").kind == "temporal"
        assert get_predicate("count").kind == "counting"


class TestArgChecking:
    @pytest.mark.unit
    def test_missing_required_argument(self) -> None:
        with pytest.raises(PredicateError, match="missing required"):
            check_args(get_predicate("near"), {"class_a": "person"})

    @pytest.mark.unit
    def test_unknown_argument(self) -> None:
        with pytest.raises(PredicateError, match="unknown argument"):
            check_args(get_predicate("detected"), {"class_name": "person", "colour": "red"})

    @pytest.mark.unit
    def test_defaults_are_applied(self) -> None:
        resolved = check_args(get_predicate("count"), {"class_name": "person"})
        assert resolved["at_least"] == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [0, -0.2, 1.5, "0.2m", True, None])
    def test_fraction_rejects_out_of_range_and_wrong_types(self, bad: Any) -> None:
        with pytest.raises(PredicateError):
            check_args(
                get_predicate("near"),
                {"class_a": "person", "class_b": "stove", "max_dist": bad},
            )

    @pytest.mark.unit
    def test_fraction_error_explains_the_unit(self) -> None:
        with pytest.raises(PredicateError) as excinfo:
            check_args(
                get_predicate("near"),
                {"class_a": "person", "class_b": "stove", "max_dist": "0.2m"},
            )
        assert "frame-normalised" in str(excinfo.value)

    @pytest.mark.unit
    def test_error_message_carries_the_field_path(self) -> None:
        with pytest.raises(PredicateError) as excinfo:
            check_args(
                get_predicate("near"),
                {"class_a": "person", "class_b": "stove", "max_dist": 4},
                where="trigger.all[1]",
            )
        assert "trigger.all[1].args.max_dist" in str(excinfo.value)

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [0, -5, "thirty"])
    def test_seconds_must_be_positive(self, bad: Any) -> None:
        with pytest.raises(PredicateError):
            check_args(get_predicate("absent_for"), {"class_name": "person", "seconds": bad})

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [[], "person", [""], [3]])
    def test_class_list_must_be_non_empty_strings(self, bad: Any) -> None:
        with pytest.raises(PredicateError):
            check_args(get_predicate("any_of"), {"class_names": bad})

    @pytest.mark.unit
    def test_referenced_classes_collects_every_class_arg(self) -> None:
        spec = get_predicate("near")
        args = {"class_a": "person", "class_b": "stove", "max_dist": 0.2}
        assert referenced_classes(spec, args) == frozenset({"person", "stove"})

        list_spec = get_predicate("any_of")
        assert referenced_classes(
            list_spec, {"class_names": ["medicine_strip", "medicine_bottle"]}
        ) == frozenset({"medicine_strip", "medicine_bottle"})


# ─── Behaviour ────────────────────────────────────────────────────────────────


class TestStaticPredicates:
    @pytest.mark.unit
    def test_detected(self) -> None:
        ctx = _ctx(_det("knife"))
        assert _call("detected", ctx, class_name="knife") is True
        assert _call("detected", ctx, class_name="person") is False

    @pytest.mark.unit
    def test_any_of_and_all_of(self) -> None:
        ctx = _ctx(_det("medicine_strip"), _det("person"))
        assert _call("any_of", ctx, class_names=["medicine_strip", "medicine_bottle"]) is True
        assert _call("all_of", ctx, class_names=["medicine_strip", "person"]) is True
        assert _call("all_of", ctx, class_names=["medicine_strip", "medicine_bottle"]) is False


class TestCount:
    @pytest.mark.unit
    def test_at_least_and_at_most(self) -> None:
        ctx = _ctx(_det("person", cx=0.2), _det("person", cx=0.5), _det("person", cx=0.8))
        assert _call("count", ctx, class_name="person", at_least=3) is True
        assert _call("count", ctx, class_name="person", at_least=4) is False
        assert _call("count", ctx, class_name="person", at_least=1, at_most=2) is False
        assert _call("count", ctx, class_name="person", at_least=1, at_most=3) is True

    @pytest.mark.unit
    def test_alone_is_expressible(self) -> None:
        """`require_alone` needs at_most=1 — a grandchild must not be invisible."""
        alone = _ctx(_det("person"))
        crowd = _ctx(_det("person", cx=0.2), _det("person", cx=0.8))
        assert _call("count", alone, class_name="person", at_least=1, at_most=1) is True
        assert _call("count", crowd, class_name="person", at_least=1, at_most=1) is False


class TestTemporalPredicates:
    @pytest.mark.unit
    def test_absent_for_consults_memory(self) -> None:
        ctx = _ctx(_det("stove"), memory=_StubMemory(absent={"person"}))
        assert _call("absent_for", ctx, class_name="person", seconds=30) is True
        assert _call("absent_for", ctx, class_name="stove", seconds=30) is False

    @pytest.mark.unit
    def test_present_for_requires_sustained_frames(self) -> None:
        memory = _StubMemory(consecutive={20: 45})
        ctx = _ctx(_det("wet_floor", class_id=20), memory=memory, fps=15.0)
        # 45 consecutive frames at 15 FPS is 3.0s.
        assert _call("present_for", ctx, class_name="wet_floor", seconds=3) is True
        assert _call("present_for", ctx, class_name="wet_floor", seconds=4) is False

    @pytest.mark.unit
    def test_present_for_is_false_when_absent_this_frame(self) -> None:
        ctx = _ctx(memory=_StubMemory(consecutive={20: 999}))
        assert _call("present_for", ctx, class_name="wet_floor", seconds=1) is False

    @pytest.mark.unit
    def test_present_for_scales_with_measured_fps(self) -> None:
        """A throttled device must not shorten a dwell requirement."""
        memory = _StubMemory(consecutive={20: 45})
        throttled = _ctx(_det("wet_floor", class_id=20), memory=memory, fps=2.0)
        # 45 frames at 2 FPS is 22.5s, so a 3s dwell is satisfied.
        assert _call("present_for", throttled, class_name="wet_floor", seconds=3) is True
        assert _call("present_for", throttled, class_name="wet_floor", seconds=30) is False

    @pytest.mark.unit
    def test_ever_seen(self) -> None:
        ctx = _ctx(memory=_StubMemory(ever_seen={"walking_stick"}))
        assert _call("ever_seen", ctx, class_name="walking_stick") is True
        assert _call("ever_seen", ctx, class_name="support_handle") is False


class TestSpatialPredicates:
    @pytest.mark.unit
    def test_near_is_false_for_distant_same_frame_detections(self) -> None:
        """The acceptance criterion: `near` must not degrade into co-presence."""
        ctx = _ctx(_det("person", cx=0.05, cy=0.5), _det("stove", cx=0.95, cy=0.5))
        assert _call("detected", ctx, class_name="person") is True
        assert _call("detected", ctx, class_name="stove") is True
        assert _call("near", ctx, class_a="person", class_b="stove", max_dist=0.2) is False

    @pytest.mark.unit
    def test_near_is_true_when_actually_close(self) -> None:
        ctx = _ctx(_det("person", cx=0.50, cy=0.5), _det("stove", cx=0.58, cy=0.5))
        assert _call("near", ctx, class_a="person", class_b="stove", max_dist=0.2) is True

    @pytest.mark.unit
    def test_near_uses_the_closest_pair(self) -> None:
        ctx = _ctx(
            _det("person", cx=0.05, cy=0.5),
            _det("person", cx=0.55, cy=0.5),
            _det("stove", cx=0.60, cy=0.5),
        )
        assert _call("near", ctx, class_a="person", class_b="stove", max_dist=0.1) is True

    @pytest.mark.unit
    def test_near_is_false_when_either_class_is_missing(self) -> None:
        ctx = _ctx(_det("person"))
        assert _call("near", ctx, class_a="person", class_b="stove", max_dist=0.9) is False

    @pytest.mark.unit
    def test_overlaps_uses_iou(self) -> None:
        close = _ctx(
            _det("person", cx=0.50, cy=0.5, size=0.4),
            _det("chair", cx=0.52, cy=0.5, size=0.4),
        )
        apart = _ctx(
            _det("person", cx=0.10, cy=0.5, size=0.1),
            _det("chair", cx=0.90, cy=0.5, size=0.1),
        )
        assert _call("overlaps", close, class_a="person", class_b="chair", min_iou=0.5) is True
        assert _call("overlaps", apart, class_a="person", class_b="chair", min_iou=0.1) is False

    @pytest.mark.unit
    def test_adjacent_objects_are_near_but_do_not_overlap(self) -> None:
        """Why IoU cannot replace `near`: adjacent boxes have IoU 0."""
        ctx = _ctx(
            _det("person", cx=0.40, cy=0.5, size=0.1),
            _det("stove", cx=0.52, cy=0.5, size=0.1),
        )
        assert _call("overlaps", ctx, class_a="person", class_b="stove", min_iou=0.01) is False
        assert _call("near", ctx, class_a="person", class_b="stove", max_dist=0.2) is True

    @pytest.mark.unit
    def test_above_and_below_use_image_coordinates(self) -> None:
        ctx = _ctx(_det("cupboard", cy=0.2), _det("stove", cy=0.8))
        assert _call("above", ctx, class_a="cupboard", class_b="stove") is True
        assert _call("below", ctx, class_a="cupboard", class_b="stove") is False
        assert _call("below", ctx, class_a="stove", class_b="cupboard") is True


class TestContextPredicates:
    @pytest.mark.unit
    def test_in_room(self) -> None:
        kitchen = _ctx(_det("stove"), room="kitchen")
        bathroom = _ctx(_det("stove"), room="bathroom")
        unset = _ctx(_det("stove"))
        assert _call("in_room", kitchen, room="kitchen") is True
        assert _call("in_room", bathroom, room="kitchen") is False
        assert _call("in_room", unset, room="kitchen") is False

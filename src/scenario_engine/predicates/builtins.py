"""
src.scenario_engine.predicates.builtins — The Built-in Predicate Vocabulary
===========================================================================

Every predicate is a pure function of :class:`EvalContext` plus its own
validated arguments, so each has a table-driven unit test with no YAML in the
loop.

Distances are **fractions of the frame**, never pixels or metres.
``BoundingBox`` is frame-normalised and carries no depth, so ``near`` is a
screen-space proxy for physical proximity: it varies with field of view and
camera mounting, which is why installation guidance is a Track C deliverable
(ADR-P6-07).

Kinds matter as much as behaviour. ``static`` predicates alone cannot satisfy a
trigger — the "no static-only trigger" validator rejects a scenario whose
condition is nothing but object presence, which is exactly what makes
``detected(knife) AND detected(person)`` unwritable as a complete scenario.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...pipeline import BoundingBox
from ..context import EvalContext
from .registry import register_predicate

# ─── Static presence ──────────────────────────────────────────────────────────


@register_predicate(
    "detected",
    kind="static",
    arg_types={"class_name": "class"},
    required=("class_name",),
)
def detected(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Class is present in the current frame."""
    return ctx.present(str(args["class_name"]))


@register_predicate(
    "any_of",
    kind="static",
    arg_types={"class_names": "class_list"},
    required=("class_names",),
)
def any_of(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Any of the listed classes is present in the current frame."""
    return any(ctx.present(str(name)) for name in args["class_names"])


@register_predicate(
    "all_of",
    kind="static",
    arg_types={"class_names": "class_list"},
    required=("class_names",),
)
def all_of(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Every listed class is present in the current frame."""
    return all(ctx.present(str(name)) for name in args["class_names"])


# ─── Counting ─────────────────────────────────────────────────────────────────


@register_predicate(
    "count",
    kind="counting",
    arg_types={"class_name": "class", "at_least": "count", "at_most": "count"},
    required=("class_name",),
    defaults={"at_least": 1},
)
def count(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Instance count of a class falls within a range.

    Multiplicity the legacy `set[str]` evaluator could not express. Needed for
    multi-generational homes, where `absent_for(person, N)` is satisfied by
    *any* person — so a grandchild in the kitchen suppresses an
    unattended-stove scenario.
    """
    observed = ctx.count(str(args["class_name"]))
    if observed < int(args["at_least"]):
        return False
    at_most = args.get("at_most")
    return at_most is None or observed <= int(at_most)


# ─── Temporal ─────────────────────────────────────────────────────────────────


@register_predicate(
    "absent_for",
    kind="temporal",
    arg_types={"class_name": "class", "seconds": "seconds"},
    required=("class_name", "seconds"),
)
def absent_for(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Class has been continuously absent for at least N seconds.

    Uses the context's **measured** fps. A never-seen class counts as absent
    for the whole session rather than saturating at the memory window.
    """
    return ctx.memory.is_absent_for_by_name(
        str(args["class_name"]), float(args["seconds"]), ctx.fps
    )


@register_predicate(
    "present_for",
    kind="temporal",
    arg_types={"class_name": "class", "seconds": "seconds"},
    required=("class_name", "seconds"),
)
def present_for(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Class has been present in every frame for at least N seconds.

    The dwell primitive. Dwell is nearly free for slow hazards — a wet floor
    persists for minutes — while eliminating detector flicker, which is why the
    precision budget is spent here rather than on raising confidence thresholds
    (domain_research_report.md §3).
    """
    detection = ctx.best(str(args["class_name"]))
    if detection is None:
        return False
    required_frames = float(args["seconds"]) * max(ctx.fps, 0.1)
    return ctx.memory.consecutive_frames(detection.class_id) >= required_frames


@register_predicate(
    "ever_seen",
    kind="temporal",
    arg_types={"class_name": "class"},
    required=("class_name",),
)
def ever_seen(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Class has been detected at least once since start-up.

    Distinguishes "this home has a walking stick that is currently elsewhere"
    from "this home has no walking stick" — the two readings of an absence that
    the legacy engine collapsed together.
    """
    return ctx.memory.has_ever_seen(str(args["class_name"]))


# ─── Spatial ──────────────────────────────────────────────────────────────────


def _centre_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean distance between two box centres, in frame fractions."""
    return float(((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5)


def _pairs(
    ctx: EvalContext, args: Mapping[str, Any]
) -> tuple[tuple[BoundingBox, ...], tuple[BoundingBox, ...]]:
    return ctx.boxes(str(args["class_a"])), ctx.boxes(str(args["class_b"]))


@register_predicate(
    "near",
    kind="spatial",
    arg_types={"class_a": "class", "class_b": "class", "max_dist": "fraction"},
    required=("class_a", "class_b", "max_dist"),
)
def near(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Some instance of A is within max_dist of some instance of B.

    ``max_dist`` is a fraction of the frame diagonal-ish scale, not a physical
    distance. IoU cannot express this: two adjacent, non-overlapping objects
    have IoU 0, and "person standing beside the stove" is the single most
    important spatial relation in the set.
    """
    boxes_a, boxes_b = _pairs(ctx, args)
    limit = float(args["max_dist"])
    return any(_centre_distance(a, b) <= limit for a in boxes_a for b in boxes_b)


@register_predicate(
    "overlaps",
    kind="spatial",
    arg_types={"class_a": "class", "class_b": "class", "min_iou": "fraction"},
    required=("class_a", "class_b", "min_iou"),
)
def overlaps(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Some instance of A overlaps some instance of B by at least min_iou."""
    boxes_a, boxes_b = _pairs(ctx, args)
    threshold = float(args["min_iou"])
    return any(a.iou(b) >= threshold for a in boxes_a for b in boxes_b)


@register_predicate(
    "above",
    kind="spatial",
    arg_types={"class_a": "class", "class_b": "class"},
    required=("class_a", "class_b"),
)
def above(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Some instance of A sits higher in frame than some instance of B.

    Image coordinates: y increases downward, so "above" means a smaller cy.
    """
    boxes_a, boxes_b = _pairs(ctx, args)
    return any(a.cy < b.cy for a in boxes_a for b in boxes_b)


@register_predicate(
    "below",
    kind="spatial",
    arg_types={"class_a": "class", "class_b": "class"},
    required=("class_a", "class_b"),
)
def below(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """Some instance of A sits lower in frame than some instance of B."""
    boxes_a, boxes_b = _pairs(ctx, args)
    return any(a.cy > b.cy for a in boxes_a for b in boxes_b)


# ─── Context ──────────────────────────────────────────────────────────────────


@register_predicate(
    "in_room",
    kind="context",
    arg_types={"room": "room"},
    required=("room",),
)
def in_room(ctx: EvalContext, args: Mapping[str, Any]) -> bool:
    """The camera is deployed in the named room.

    A deployment-time constant from the capture room vocabulary — one camera per
    room — not a per-frame geometric zone. Zones need an authoring tool and a
    recalibration story, and are deliberately deferred (ADR-P6-07).
    """
    return ctx.room == str(args["room"])

"""
EvalContext — Everything a Predicate Is Allowed to Look At
==========================================================

The single input type for every scenario predicate.

Why this type exists
--------------------
The legacy rule engine handed its evaluator a bare ``set[str]`` of class names
(``rule_engine.py`` built ``detected_names = {d.class_name for d in detections}``
and passed only that). Bounding boxes and multiplicity were therefore destroyed
*before* any predicate ran, which makes ``near``, ``overlaps``, ``above`` and
``count`` unimplementable — not merely awkward.

That is the real reason "extend the existing DSL" was rejected rather than a
stylistic preference: the evaluator signature had to change regardless. See
docs/08_scenario_engineering/adr/ADR-P6-03-structured-ast-over-string-dsl.md.

The failure mode this type prevents is silent. A ``near(person, stove, 0.2)``
implemented against the old set signature degrades into
``detected(person) AND detected(stove)`` — semantically identical to the
``knife_near_person`` rule being deleted for firing throughout normal cooking —
while *looking* implemented.

Units
-----
``BoundingBox`` is normalised to the frame, so all distances here are
**fractions of frame width/height**, not pixels and not metres. A distance is a
screen-space proxy for physical proximity: it varies with field of view and
camera mounting, and carries no depth information. See
docs/08_scenario_engineering/adr/ADR-P6-07-near-units-and-room-vs-zone.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..pipeline import BoundingBox, Detection


@runtime_checkable
class MemoryView(Protocol):
    """The temporal-state surface predicates may consult.

    A Protocol rather than a direct ``EventMemory`` import, so predicates stay
    testable with a stub and the package keeps a single, explicit dependency
    surface. ``src.pipeline.event_memory.EventMemory`` satisfies it structurally.
    """

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        """Has ``class_name`` been continuously absent for at least ``seconds``?"""
        ...

    def consecutive_frames(self, class_id: int) -> int:
        """How many consecutive trailing frames contain this class?"""
        ...

    def has_ever_seen(self, class_name: str) -> bool:
        """Has this class been detected at least once since start-up?"""
        ...


@dataclass(frozen=True)
class EvalContext:
    """Immutable view of one frame, plus the temporal state behind it.

    Attributes:
        detections: Every detection in the current frame, bounding boxes intact.
        memory:     Temporal state (see :class:`MemoryView`).
        fps:        **Measured** loop rate, not the nominal target. Temporal
                    predicates convert frame counts to seconds with it, so a
                    nominal value silently rescales every threshold when the
                    device throttles.
        frame_id:   Current frame index, for provenance in explanations.
        room:       Deployment room (``kitchen``, ``bathroom``, …) from the
                    camera's configuration. A static per-camera constant — not a
                    per-frame geometric zone, which is deliberately deferred.
    """

    detections: tuple[Detection, ...]
    memory: MemoryView
    fps: float
    frame_id: int = 0
    room: str | None = None

    # Derived indexes, built once. Excluded from equality/repr because they are
    # a pure function of `detections`.
    _by_name: dict[str, tuple[Detection, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        index: dict[str, list[Detection]] = {}
        for detection in self.detections:
            index.setdefault(detection.class_name, []).append(detection)
        object.__setattr__(self, "_by_name", {name: tuple(items) for name, items in index.items()})

    # ── Presence and multiplicity ────────────────────────────────────────────

    @property
    def detected_names(self) -> frozenset[str]:
        """Class names present in this frame."""
        return frozenset(self._by_name)

    def present(self, class_name: str) -> bool:
        """Is at least one instance of this class in the current frame?"""
        return class_name in self._by_name

    def count(self, class_name: str) -> int:
        """How many instances of this class are in the current frame?

        Multiplicity the old ``set[str]`` signature could not express. Needed for
        multi-person homes, where ``absent_for(person, N)`` is satisfied by *any*
        person — so a grandchild in the kitchen suppresses an unattended-stove
        scenario.
        """
        return len(self._by_name.get(class_name, ()))

    def instances(self, class_name: str) -> tuple[Detection, ...]:
        """Every detection of this class in the current frame."""
        return self._by_name.get(class_name, ())

    def boxes(self, class_name: str) -> tuple[BoundingBox, ...]:
        """Bounding boxes for every instance of this class."""
        return tuple(d.bbox for d in self._by_name.get(class_name, ()))

    def best(self, class_name: str) -> Detection | None:
        """Highest-confidence instance of this class, or None."""
        instances = self._by_name.get(class_name, ())
        return max(instances, key=lambda d: d.confidence) if instances else None

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_frame(
        cls,
        detections: list[Detection] | tuple[Detection, ...],
        memory: MemoryView,
        fps: float,
        frame_id: int = 0,
        room: str | None = None,
    ) -> EvalContext:
        """Build a context from a frame's detections.

        Args:
            detections: Detections for this frame; copied into a tuple.
            memory:     Temporal state.
            fps:        Measured loop rate.
            frame_id:   Frame index.
            room:       Deployment room for this camera.
        """
        return cls(
            detections=tuple(detections),
            memory=memory,
            fps=fps,
            frame_id=frame_id,
            room=room,
        )

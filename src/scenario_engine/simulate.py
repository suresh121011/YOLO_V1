"""
src.scenario_engine.simulate — Alert-Volume Simulation
======================================================

Replays a day of detections through the scenario set and reports how many alerts
each scenario would actually produce.

Why this is a gate and not a report
-----------------------------------
Alarm fatigue is the dominant failure mode of home monitoring, and it is not a
comfort issue: the endpoint is a muted or unplugged device, which is a 100%
false-negative rate. The literature puts non-actionable alarm rates at 74–99%,
and telecare abandonment rising from 17.7% to 23.5% over twelve months *with
decreasing perceived safety*.

The six legacy rules projected roughly 370 alerts on day one with essentially
none actionable — ``knife_near_person`` firing throughout every meal,
``medicine_reminder`` every 300 s at a strip left on a table, and
``gas_cylinder_check`` every 600 s including overnight. Every one of them looked
reasonable in isolation. Volume is a property of the *set* against *real
occupancy*, so only a replay can see it.

What the model does and does not include
----------------------------------------
Included, because each materially changes volume: dwell before firing, the
per-scenario cooldown, ``max_repeats`` within one event, the per-scenario daily
budget, quiet hours, and event hysteresis via ``clear_after_seconds`` — a
condition that stays true is **one event**, not a metronome.

Excluded: the global rate limit and queue eviction. Those are runtime
back-pressure, and counting them here would let a scenario set be "in budget"
only because the queue was throwing its alerts away.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..pipeline import BoundingBox, Detection
from .context import EvalContext
from .schema import QuietHoursBehaviour, Scenario, Status
from .trigger import evaluate_trigger

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Frame:
    """One sampled moment: which classes are visible, and when."""

    t_seconds: float
    present: frozenset[str]
    room: str | None = None


@dataclass
class _EventState:
    """Per-scenario state machine: IDLE -> PENDING -> ACTIVE -> RESOLVED."""

    pending_since: float | None = None
    active_since: float | None = None
    last_true_at: float | None = None
    last_alert_at: float | None = None
    repeats_this_event: int = 0
    alerts_today: int = 0


@dataclass(frozen=True)
class ScenarioVolume:
    """Projected volume for one scenario."""

    scenario_id: str
    alerts: int
    budget: int
    suppressed_by_budget: int
    events: int

    @property
    def over_budget(self) -> bool:
        return self.alerts > self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "alerts": self.alerts,
            "max_per_day": self.budget,
            "suppressed_by_budget": self.suppressed_by_budget,
            "events": self.events,
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True)
class VolumeReport:
    """Whole-set projection."""

    per_scenario: tuple[ScenarioVolume, ...]
    total_alerts: int
    budget: int
    simulated_seconds: float
    frames: int

    @property
    def within_budget(self) -> bool:
        return self.total_alerts <= self.budget and not any(
            v.over_budget for v in self.per_scenario
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_alerts": self.total_alerts,
            "max_projected_alerts_per_day": self.budget,
            "within_budget": self.within_budget,
            "simulated_seconds": self.simulated_seconds,
            "frames": self.frames,
            "per_scenario": [v.to_dict() for v in self.per_scenario],
        }


class _ReplayMemory:
    """Temporal state derived from the replay timeline, not a live pipeline."""

    def __init__(self) -> None:
        self._last_seen: dict[str, float] = {}
        self._present_since: dict[str, float] = {}
        self._now = 0.0
        #: Owned by the replay loop, which knows the id<->name mapping. An
        #: instance attribute, not a class one — a shared mutable default would
        #: leak dwell state between simulations.
        self.consecutive_by_id: dict[int, int] = {}

    def observe(self, frame: Frame) -> None:
        self._now = frame.t_seconds
        for name in frame.present:
            self._last_seen[name] = frame.t_seconds
            self._present_since.setdefault(name, frame.t_seconds)
        for name in list(self._present_since):
            if name not in frame.present:
                del self._present_since[name]

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float) -> bool:
        last = self._last_seen.get(class_name)
        if last is None:
            return self._now >= seconds  # Never seen: absent for the whole session.
        return (self._now - last) >= seconds

    def consecutive_frames(self, class_id: int) -> int:
        return self.consecutive_by_id.get(class_id, 0)

    def has_ever_seen(self, class_name: str) -> bool:
        return class_name in self._last_seen


def _quiet(scenario: Scenario, t_seconds: float) -> bool:
    """Is this moment inside the scenario's quiet hours?"""
    start_h = int(scenario.quiet_hours.start.split(":")[0])
    end_h = int(scenario.quiet_hours.end.split(":")[0])
    hour = int((t_seconds % SECONDS_PER_DAY) // 3600)
    if start_h <= end_h:
        return start_h <= hour < end_h
    return hour >= start_h or hour < end_h  # window wraps midnight


def simulate(
    scenarios: Sequence[Scenario],
    frames: Iterable[Frame],
    budget: int = 20,
    fps: float = 15.0,
) -> VolumeReport:
    """Replay frames through the scenario set and project daily alert volume.

    Args:
        scenarios: The scenario set. Only active and draft rows are replayed.
        frames:    Sampled timeline, ascending in ``t_seconds``.
        budget:    Whole-set daily alert ceiling.
        fps:       Nominal rate, used to convert dwell seconds to frames.
    """
    runnable = [s for s in scenarios if s.status in (Status.ACTIVE, Status.DRAFT)]
    states: dict[str, _EventState] = {s.scenario_id: _EventState() for s in runnable}
    counts: dict[str, int] = {s.scenario_id: 0 for s in runnable}
    suppressed: dict[str, int] = {s.scenario_id: 0 for s in runnable}
    events: dict[str, int] = {s.scenario_id: 0 for s in runnable}

    memory = _ReplayMemory()
    consecutive = memory.consecutive_by_id
    class_ids: dict[str, int] = {}

    frame_list = list(frames)
    last_t = 0.0

    for frame in frame_list:
        last_t = frame.t_seconds
        memory.observe(frame)

        for name in sorted(frame.present):
            class_ids.setdefault(name, len(class_ids))
        for name, class_id in class_ids.items():
            consecutive[class_id] = consecutive.get(class_id, 0) + 1 if name in frame.present else 0

        detections = [
            Detection(
                class_id=class_ids[name],
                class_name=name,
                confidence=0.9,
                bbox=BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1),
                frame_id=0,
                timestamp_ms=frame.t_seconds * 1000,
            )
            for name in sorted(frame.present)
        ]
        ctx = EvalContext.from_frame(detections, memory, fps, room=frame.room)

        for scenario in runnable:
            state = states[scenario.scenario_id]
            holds = evaluate_trigger(scenario.trigger, ctx)

            if not holds:
                # Hysteresis: the event closes only after the clear window.
                if state.last_true_at is not None and (
                    frame.t_seconds - state.last_true_at >= scenario.clear_after_seconds
                ):
                    state.pending_since = None
                    state.active_since = None
                    state.last_true_at = None
                    state.repeats_this_event = 0
                continue

            state.last_true_at = frame.t_seconds
            if state.pending_since is None:
                state.pending_since = frame.t_seconds

            # Dwell must elapse before the first announcement.
            if frame.t_seconds - state.pending_since < scenario.min_dwell_seconds:
                continue

            if state.active_since is None:
                state.active_since = frame.t_seconds
                events[scenario.scenario_id] += 1
            elif state.repeats_this_event >= scenario.max_repeats:
                continue

            if state.last_alert_at is not None and (
                frame.t_seconds - state.last_alert_at < scenario.cooldown_seconds
            ):
                continue

            if scenario.patient_facing and _quiet(scenario, frame.t_seconds):
                if scenario.quiet_hours.behaviour is not QuietHoursBehaviour.ALWAYS_SPEAK:
                    continue

            if counts[scenario.scenario_id] >= scenario.max_per_day:
                suppressed[scenario.scenario_id] += 1
                continue

            counts[scenario.scenario_id] += 1
            state.last_alert_at = frame.t_seconds
            if state.repeats_this_event or state.active_since != frame.t_seconds:
                state.repeats_this_event += 1

    per_scenario = tuple(
        ScenarioVolume(
            scenario_id=s.scenario_id,
            alerts=counts[s.scenario_id],
            budget=s.max_per_day,
            suppressed_by_budget=suppressed[s.scenario_id],
            events=events[s.scenario_id],
        )
        for s in sorted(runnable, key=lambda s: s.scenario_id)
    )

    return VolumeReport(
        per_scenario=per_scenario,
        total_alerts=sum(counts.values()),
        budget=budget,
        simulated_seconds=last_t,
        frames=len(frame_list),
    )


def synthetic_day(
    occupancy: Mapping[str, Sequence[tuple[float, float]]],
    step_seconds: float = 30.0,
    room: str | None = None,
    duration_seconds: float = SECONDS_PER_DAY,
) -> list[Frame]:
    """Build a day's timeline from per-class visibility windows.

    Args:
        occupancy: ``{class_name: [(start_s, end_s), ...]}`` — when each class is
            visible. A permanently-visible object is ``[(0, 86400)]``, which is
            how a gas cylinder in an Indian kitchen actually behaves.
        step_seconds: Sampling interval. Coarser than the real frame rate on
            purpose; volume is driven by cooldowns measured in minutes.
        room: Deployment room for every frame.
        duration_seconds: Length of the simulated day.
    """
    frames: list[Frame] = []
    t = 0.0
    while t < duration_seconds:
        present = frozenset(
            name
            for name, windows in occupancy.items()
            if any(start <= t < end for start, end in windows)
        )
        frames.append(Frame(t_seconds=t, present=present, room=room))
        t += step_seconds
    return frames


@dataclass(frozen=True)
class OccupancyProfile:
    """A named day of household activity, for reproducible replays."""

    name: str
    occupancy: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    room: str | None = None


def indian_kitchen_day() -> OccupancyProfile:
    """A representative kitchen day, tuned to the failure modes we know about.

    The LPG cylinder is visible all day because it lives under the platform;
    the stove is visible all day because it is a fixed object; a person cooks
    twice; a knife is out during meal prep. This is the exact occupancy that
    turned three legacy rules into metronomes.
    """
    hour = 3600.0
    return OccupancyProfile(
        name="indian_kitchen_day",
        room="kitchen",
        occupancy={
            "gas_cylinder": [(0.0, SECONDS_PER_DAY)],
            "stove": [(0.0, SECONDS_PER_DAY)],
            "person": [(7 * hour, 8.5 * hour), (18 * hour, 20 * hour)],
            "knife": [(7.2 * hour, 8.0 * hour), (18.2 * hour, 19.2 * hour)],
            "medicine_strip": [(8 * hour, 22 * hour)],
        },
    )

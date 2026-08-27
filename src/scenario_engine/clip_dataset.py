"""
src.scenario_engine.clip_dataset — Clip Collection Accounting
==============================================================

Answers one question honestly: **how many clips do we actually have?**

Not how many files are on disk. A clip counts toward the collection target only
when a human has reviewed it and set ``review_status: accepted``. This module
exists because "100 clips complete" is the claim most likely to be made on the
strength of ``ls | wc -l``, and this repository has a recorded incident where a
release gate read green while structurally incapable of failing.

Everything here is pure aggregation over :class:`~src.scenario_engine.clips.ClipManifest`
values — no I/O, no ffmpeg — so the accounting is testable with synthetic
manifests and cannot drift from what the report claims.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.scenario_engine.clips import ClipManifest, ClipRequirements

_PRESENT_FOR = re.compile(r"present_for\([^)]*seconds=(\d+(?:\.\d+)?)")
_ABSENT_FOR = re.compile(r"absent_for\([^)]*seconds=(\d+(?:\.\d+)?)")


def min_clip_seconds(condition: str) -> float:
    """Shortest clip in which a scenario could possibly fire.

    Read off the compiled condition: the longest ``present_for`` must elapse,
    and any ``absent_for`` must elapse *after* whatever established the state,
    so the two add. ``SC-KIT-001`` needs ``present_for(stove, 60)`` followed by
    ``absent_for(person, 900)`` -- sixteen minutes.

    This is the arithmetic that decides whether a scenario can be demonstrated
    on video at all, and it is the reason most short clips of a hazard are
    correctly **negatives**: within twenty seconds the engine has not seen
    enough time pass, and staying silent is the right behaviour.

    Returns 0.0 for a scenario with no temporal predicate.
    """
    present = [float(m) for m in _PRESENT_FOR.findall(condition)]
    absent = [float(m) for m in _ABSENT_FOR.findall(condition)]
    return (max(present) if present else 0.0) + (max(absent) if absent else 0.0)


#: A scenario with fewer accepted clips than this cannot support a meaningful
#: claim about the engine's behaviour on it — one clip is an anecdote, and a
#: single camera angle plus a single household is not evidence of generality.
DEFAULT_MIN_CLIPS_PER_SCENARIO = 4

#: First-batch collection target.
DEFAULT_TARGET_ACCEPTED = 100


@dataclass(frozen=True)
class ScenarioCoverage:
    """Per-scenario accounting."""

    scenario_id: str
    accepted: int
    accepted_positive: int
    accepted_negative: int
    pending: int
    rejected: int
    sufficient: bool
    shortfall: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "accepted": self.accepted,
            "accepted_positive": self.accepted_positive,
            "accepted_negative": self.accepted_negative,
            "pending": self.pending,
            "rejected": self.rejected,
            "sufficient": self.sufficient,
            "shortfall": self.shortfall,
        }


@dataclass(frozen=True)
class DatasetReport:
    """The whole batch, counted."""

    total_manifests: int
    accepted: int
    pending: int
    rejected: int
    accepted_positive: int
    accepted_negative: int
    negative_fraction: float
    min_negative_fraction: float
    negative_fraction_met: bool
    target_accepted: int
    target_met: bool
    indian_home: int
    own_capture: int
    external: int
    privacy_stripped: int
    privacy_unstripped: int
    consent_complete: int
    consent_missing: int
    licence_complete: int
    licence_missing: int
    coverage: tuple[ScenarioCoverage, ...]
    uncovered_scenarios: tuple[str, ...]
    rejections: tuple[dict[str, str], ...]
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        return "PASS" if not self.problems else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "totals": {
                "manifests_on_disk": self.total_manifests,
                "accepted": self.accepted,
                "pending": self.pending,
                "rejected": self.rejected,
                "target_accepted": self.target_accepted,
                "target_met": self.target_met,
            },
            "polarity": {
                "accepted_positive": self.accepted_positive,
                "accepted_negative": self.accepted_negative,
                "negative_fraction": round(self.negative_fraction, 4),
                "min_negative_fraction": self.min_negative_fraction,
                "negative_fraction_met": self.negative_fraction_met,
            },
            "provenance": {
                "own_capture": self.own_capture,
                "external": self.external,
                "indian_home": self.indian_home,
                "licence_complete": self.licence_complete,
                "licence_missing": self.licence_missing,
            },
            "privacy": {
                "metadata_stripped": self.privacy_stripped,
                "metadata_not_stripped": self.privacy_unstripped,
                "consent_complete": self.consent_complete,
                "consent_missing": self.consent_missing,
            },
            "coverage": [c.to_dict() for c in self.coverage],
            "uncovered_scenarios": list(self.uncovered_scenarios),
            "rejections": list(self.rejections),
            "problems": list(self.problems),
        }


def impossible_positives(
    clips: Iterable[ClipManifest],
    conditions: dict[str, str],
) -> list[str]:
    """Positive clips too short for their scenario to have fired.

    The most likely data-quality failure in a short-clip collection batch, and
    the one a reviewer is least able to catch by eye: a 20-second clip of a
    person at a stove labelled as an ``SC-KIT-001`` positive. It cannot be one.
    The scenario needs fifteen minutes of absence, so the engine correctly stays
    silent, and a suite containing that clip fails for a reason nobody can find.

    Such a clip is not waste -- relabelled ``negative`` with
    ``negative_kind: pre_dwell`` it is a genuine and valuable test. This
    function exists to force that relabelling rather than to discard footage.
    """
    problems: list[str] = []
    for clip in clips:
        if clip.polarity != "positive":
            continue
        condition = conditions.get(clip.scenario_id)
        if condition is None:
            continue
        needed = min_clip_seconds(condition)
        if needed and clip.duration_s < needed:
            problems.append(
                f"{clip.clip_id}: {clip.duration_s:.0f}s clip claims a {clip.scenario_id} "
                f"positive, but that scenario cannot fire before {needed:.0f}s. Relabel it "
                f"negative with negative_kind: pre_dwell -- the engine staying silent here "
                f"is correct behaviour, and testing that is worth more than a clip that "
                f"asserts something impossible."
            )
    return problems


def build_report(
    clips: Iterable[ClipManifest],
    scenario_ids: Sequence[str],
    requirements: ClipRequirements | None = None,
    target_accepted: int = DEFAULT_TARGET_ACCEPTED,
    min_per_scenario: int = DEFAULT_MIN_CLIPS_PER_SCENARIO,
    conditions: dict[str, str] | None = None,
) -> DatasetReport:
    """Count a clip set against the collection targets.

    Args:
        clips:            Loaded manifests, in any state.
        scenario_ids:     Every *active* scenario id, so a scenario with zero
                          clips appears as a gap rather than being invisible.
        requirements:     Supplies ``min_negative_fraction``.
        target_accepted:  Accepted-clip target for the batch.
        min_per_scenario: Accepted clips below which a scenario is under-covered.
        conditions:       ``scenario_id -> compiled condition``. When given,
                          positives too short to have fired are reported.

    Returns:
        A :class:`DatasetReport`. ``problems`` is empty only when every gate
        holds; a report with problems has verdict FAIL.
    """
    requirements = requirements or ClipRequirements()
    clips = list(clips)
    accepted = [c for c in clips if c.is_accepted]

    accepted_positive = sum(1 for c in accepted if c.polarity == "positive")
    accepted_negative = sum(1 for c in accepted if c.polarity == "negative")
    negative_fraction = accepted_negative / len(accepted) if accepted else 0.0

    coverage: list[ScenarioCoverage] = []
    for scenario_id in sorted(set(scenario_ids) | {c.scenario_id for c in clips}):
        mine = [c for c in clips if c.scenario_id == scenario_id]
        mine_accepted = [c for c in mine if c.is_accepted]
        count = len(mine_accepted)
        coverage.append(
            ScenarioCoverage(
                scenario_id=scenario_id,
                accepted=count,
                accepted_positive=sum(1 for c in mine_accepted if c.polarity == "positive"),
                accepted_negative=sum(1 for c in mine_accepted if c.polarity == "negative"),
                pending=sum(1 for c in mine if c.review_status == "pending"),
                rejected=sum(1 for c in mine if c.review_status == "rejected"),
                sufficient=count >= min_per_scenario,
                shortfall=max(0, min_per_scenario - count),
            )
        )

    own = [c for c in clips if not c.provenance.is_external]
    external = [c for c in clips if c.provenance.is_external]

    problems: list[str] = []
    if len(accepted) < target_accepted:
        problems.append(
            f"{len(accepted)} accepted clip(s) against a target of {target_accepted}. "
            f"{len(clips) - len(accepted)} manifest(s) exist but are not accepted -- "
            f"a file on disk is not a dataset member."
        )
    if accepted and negative_fraction < requirements.min_negative_fraction:
        problems.append(
            f"negatives are {negative_fraction:.0%} of the accepted set, under the "
            f"{requirements.min_negative_fraction:.0%} floor -- the batch measures "
            f"sensitivity and is blind to the false-positive rate"
        )
    unstripped = [c for c in clips if not c.metadata_stripped]
    if unstripped:
        problems.append(
            f"{len(unstripped)} clip(s) are not marked metadata_stripped: "
            f"{sorted(c.clip_id for c in unstripped)[:5]}"
        )
    problems.extend(impossible_positives(clips, conditions or {}))
    uncovered = tuple(c.scenario_id for c in coverage if not c.sufficient)
    if uncovered:
        problems.append(
            f"{len(uncovered)} scenario(s) have fewer than {min_per_scenario} accepted "
            f"clips: {list(uncovered)}"
        )

    return DatasetReport(
        total_manifests=len(clips),
        accepted=len(accepted),
        pending=sum(1 for c in clips if c.review_status == "pending"),
        rejected=sum(1 for c in clips if c.review_status == "rejected"),
        accepted_positive=accepted_positive,
        accepted_negative=accepted_negative,
        negative_fraction=negative_fraction,
        min_negative_fraction=requirements.min_negative_fraction,
        negative_fraction_met=(
            bool(accepted) and negative_fraction >= requirements.min_negative_fraction
        ),
        target_accepted=target_accepted,
        target_met=len(accepted) >= target_accepted,
        indian_home=sum(1 for c in clips if c.indian_home),
        own_capture=len(own),
        external=len(external),
        privacy_stripped=sum(1 for c in clips if c.metadata_stripped),
        privacy_unstripped=len(unstripped),
        consent_complete=sum(1 for c in own if c.consent_reference),
        consent_missing=sum(1 for c in own if not c.consent_reference),
        licence_complete=sum(1 for c in external if c.provenance.license),
        licence_missing=sum(1 for c in external if not c.provenance.license),
        coverage=tuple(coverage),
        uncovered_scenarios=uncovered,
        rejections=tuple(
            {"clip_id": c.clip_id, "scenario_id": c.scenario_id, "reason": c.rejection_reason}
            for c in sorted(clips, key=lambda c: c.clip_id)
            if c.review_status == "rejected"
        ),
        problems=tuple(problems),
    )

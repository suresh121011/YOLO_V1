"""
Caregiver Notification Sink
============================
The consumer for ``Alert.caregiver_channel``, the fourth stage of the product
output (risk level -> patient prompt -> next best action -> caregiver alert).

ADR-P6-09 added the field with no consumer and recorded that M9 would specify
the sink. This is that sink, and it is deliberately **local only**: it appends
to ``logs/caregiver.jsonl``. No network, no push service, no telephony.

Why a local sink rather than none, and why not a remote one
-----------------------------------------------------------
A field that nothing reads is indistinguishable from a feature that works. Nine
scenarios in ``configs/scenarios/`` declare a channel today, and ``SC-BTH-002``
(no grab bar near the toilet) is *entirely* caregiver-facing — it deliberately
never speaks to the resident, so with no sink it produces nothing at all
anywhere. A local record makes the stage observable and testable now.

A remote sink is not written here because a caregiver transport is a product
with its own consent surface, retention policy, failure semantics and an
identity for the caregiver, none of which exist yet
(``configs/feature_flags.yaml`` ``caregiver_sync: false``). Writing a plausible
one would create the appearance of an escalation path that nobody receives.

``push_and_call`` is honest about this
--------------------------------------
No telephony exists, so a ``push_and_call`` alert is recorded with
``escalation_pending: true`` and logged at WARNING. The system's stated
non-negotiable is that escalation terminates at a human; when it cannot reach
one, it says so loudly instead of writing a line that looks delivered.

Privacy
-------
Records carry class *names*, never bounding boxes. ``StructuredLogger`` redacts
person/face geometry from frame logs, and a second alert-shaped log that leaked
it would reopen exactly that hole in a different file.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from . import Alert

logger = logging.getLogger(__name__)

#: Channels that go out the moment the alert fires.
IMMEDIATE_CHANNELS = frozenset({"push", "push_and_call"})

#: Channels that are batched until :meth:`LocalCaregiverSink.flush`.
BATCHED_CHANNELS = frozenset({"digest"})

#: The channel meaning "the caregiver is not told about this one".
SILENT_CHANNEL = "none"


class LocalCaregiverSink:
    """Append caregiver-channel alerts to a JSONL file. Satisfies ``BaseCaregiverSink``.

    Args:
        log_dir:    Directory for ``caregiver.jsonl``. Shares ``logs/`` with the
            structured logger but stays a separate file: these are records a
            human is meant to read, not frame telemetry.
        max_digest: Cap on buffered digest entries. Beyond it the oldest are
            dropped and the drop is counted in the flushed record, so a digest
            can never grow without bound on a device that runs for months.
    """

    def __init__(self, log_dir: str | Path = "logs", max_digest: int = 200) -> None:
        self._path = Path(log_dir) / "caregiver.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_digest = max_digest
        self._lock = threading.Lock()
        self._digest: list[dict[str, Any]] = []
        self._dropped = 0
        self._delivered = 0

    # ── BaseCaregiverSink ────────────────────────────────────────────────────

    def notify(self, alert: Alert) -> bool:
        """Route one alert. Returns False only when the channel is ``none``."""
        channel = (alert.caregiver_channel or SILENT_CHANNEL).strip().lower()
        if channel == SILENT_CHANNEL:
            return False

        record = self._record(alert, channel)

        if channel in BATCHED_CHANNELS:
            with self._lock:
                self._digest.append(record)
                if len(self._digest) > self._max_digest:
                    self._digest.pop(0)
                    self._dropped += 1
            return True

        if channel not in IMMEDIATE_CHANNELS:
            # An unrecognised channel is delivered rather than dropped. The
            # schema constrains the vocabulary at authoring time, so reaching
            # here means the two have drifted -- and over-notifying a caregiver
            # is recoverable in a way that a silently withheld alert is not.
            logger.error(
                f"{alert.rule_id}: unknown caregiver_channel {channel!r} — "
                f"delivering immediately rather than dropping it"
            )
            record["unknown_channel"] = True

        if channel == "push_and_call":
            record["escalation_pending"] = True
            logger.warning(
                f"{alert.rule_id} requested 'push_and_call' but no telephony channel is "
                f"connected. Recorded to {self._path.name} as escalation_pending; "
                f"NOBODY HAS BEEN CALLED."
            )

        self._write(record)
        return True

    def flush(self) -> None:
        """Write the accumulated digest, if any. Safe to call repeatedly."""
        with self._lock:
            if not self._digest:
                return
            entries, dropped = self._digest, self._dropped
            self._digest, self._dropped = [], 0

        self._write(
            {
                "ts": round(time.time(), 3),
                "type": "caregiver_digest",
                "channel": "digest",
                "count": len(entries),
                "dropped": dropped,
                "entries": entries,
            }
        )

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Where records are written."""
        return self._path

    @property
    def pending_digest(self) -> int:
        """How many digest entries are buffered but not yet written."""
        with self._lock:
            return len(self._digest)

    @property
    def delivered(self) -> int:
        """How many records have been written to the file."""
        return self._delivered

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _record(alert: Alert, channel: str) -> dict[str, Any]:
        """Build one record. Class names only — never bounding boxes."""
        return {
            "ts": round(time.time(), 3),
            "type": "caregiver_alert",
            "channel": channel,
            "rule_id": alert.rule_id,
            "scenario_id": alert.scenario_id,
            "severity": alert.severity.name,
            "message": alert.message,
            "next_best_action": alert.next_best_action,
            "capability_disclaimer": alert.capability_disclaimer,
            "patient_facing": alert.patient_facing,
            "frame_id": alert.frame_id,
            "observed_classes": sorted({d.class_name for d in alert.triggering_detections}),
        }

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        try:
            # newline="" keeps this LF-only on Windows, matching StructuredLogger.
            with open(self._path, "a", encoding="utf-8", newline="") as handle:
                handle.write(line + "\n")
            self._delivered += 1
        except OSError as exc:
            # A full or read-only disk must not take the safety pipeline down.
            logger.error(f"Could not write caregiver record: {exc}")

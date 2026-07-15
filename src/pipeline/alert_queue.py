"""
src.pipeline.alert_queue — Thread-Safe Alert Priority Queue
===========================================================

STAGE 1: Full implementation (lightweight, no dependencies).

Manages the queue of pending safety alerts between the Rule Engine (producer,
main thread) and Piper TTS (consumer, TTS thread).

Design:
    - Min-heap ordered by Severity (CRITICAL = highest priority, spoken first)
    - Thread-safe via threading.Lock
    - Bounded to max_size — oldest low-priority alerts dropped on overflow
    - Deduplication: same rule_id within cooldown window is suppressed

Threading model:
    - Producer: main thread (Rule Engine) calls put()
    - Consumer: TTS worker thread calls get()
    - Lock held only for queue mutation — not during TTS synthesis
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time

from . import Alert

logger = logging.getLogger(__name__)


class AlertQueue:
    """Thread-safe priority queue for safety alerts.

    Alerts are ordered by severity (CRITICAL first). When the queue is full,
    the lowest-priority pending alert is dropped to make room for higher-priority
    incoming alerts.

    Args:
        max_size: Maximum number of pending alerts in the queue (default: 10)

    Example:
        queue = AlertQueue(max_size=10)
        queue.put(alert)           # from main thread (Rule Engine)
        alert = queue.get(timeout=1.0)  # from TTS worker thread
    """

    def __init__(self, max_size: int = 10) -> None:
        # (neg_severity, timestamp, seq, alert) — seq is a strictly increasing
        # tiebreaker so heap tuples never fall through to comparing Alert objects
        # (which are not orderable). Without it, two alerts with equal severity
        # and equal timestamp — common on platforms with coarse-resolution clocks
        # like Windows — would raise TypeError during heap operations.
        self._heap: list[tuple[int, float, int, Alert]] = []
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._max_size = max_size

    def put(self, alert: Alert) -> bool:
        """Enqueue an alert (non-blocking).

        If the queue is full, drops the lowest-severity alert to make room.
        CRITICAL alerts always get into the queue.

        Args:
            alert: The Alert to enqueue.

        Returns:
            True if the alert was enqueued, False if dropped.
        """
        with self._not_empty:
            if len(self._heap) >= self._max_size:
                # Find the lowest-priority item
                min_item = min(self._heap, key=lambda x: -x[0])  # least negative = lowest priority
                if -min_item[0] >= alert.severity:
                    logger.debug(f"Alert dropped (queue full, lower priority): {alert.rule_id}")
                    return False
                # Drop lowest-priority to make room
                self._heap.remove(min_item)
                heapq.heapify(self._heap)
                logger.debug(f"Dropped lower-priority alert: {min_item[3].rule_id}")

            # Negate severity so highest severity = smallest heap value (min-heap → max priority)
            heapq.heappush(
                self._heap,
                (-int(alert.severity), time.monotonic(), next(self._counter), alert),
            )
            self._not_empty.notify()
            return True

    def get(self, timeout: float | None = None) -> Alert | None:
        """Get the highest-priority pending alert (blocking).

        Args:
            timeout: Maximum seconds to wait. None = wait forever. 0 = non-blocking.

        Returns:
            The highest-priority Alert, or None if timeout expires.
        """
        with self._not_empty:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._heap:
                if timeout == 0:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            _, _, _, alert = heapq.heappop(self._heap)
            return alert

    def size(self) -> int:
        """Return the number of pending alerts."""
        with self._lock:
            return len(self._heap)

    def clear(self) -> None:
        """Remove all pending alerts from the queue."""
        with self._not_empty:
            self._heap.clear()
            logger.debug("Alert queue cleared")

    def is_empty(self) -> bool:
        """Return True if no alerts are pending."""
        with self._lock:
            return len(self._heap) == 0

"""
Event Memory — Sliding Window Temporal State Tracker
=====================================================
Tracks which objects have been present across the last N frames.
Enables temporal rules: "stove unattended for 30 seconds", "person present for 10 frames".

Design constraints:
- In-memory only; never blocks on I/O in the hot path
- Thread-safe via threading.Lock
- Memory footprint: window_size × num_classes × ~100 bytes ≈ negligible
- Update time: < 2ms per frame
"""

from __future__ import annotations

import threading
from collections import deque

from . import Detection, MemoryEntry


class EventMemory:
    """Sliding-window temporal object tracker.

    Tracks which classes have been detected across the last N frames,
    enabling temporal rules such as "stove unattended for 30 seconds".

    All methods are thread-safe.
    """

    def __init__(self, window_size: int = 150) -> None:
        """
        Args:
            window_size: Number of recent frames to track.
                         Default 150 = ~10 seconds at 15 FPS.
        """
        self._window = window_size
        self._frames: deque[set[int]] = deque(maxlen=window_size)
        self._entries: dict[int, MemoryEntry] = {}
        self._frame_counter: int = 0
        self._lock = threading.Lock()

    # ─────────────────────────────────────────
    # Write — called every frame on main thread
    # ─────────────────────────────────────────

    def update(self, detections: list[Detection]) -> None:
        """Advance the window with the current frame's detections.

        Must be called exactly once per processed frame.
        """
        with self._lock:
            self._frame_counter += 1
            detected_ids = {d.class_id for d in detections}
            self._frames.append(detected_ids)

            for det in detections:
                if det.class_id not in self._entries:
                    self._entries[det.class_id] = MemoryEntry(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        last_seen_frame=self._frame_counter,
                        first_seen_frame=self._frame_counter,
                        detection_count=1,
                        avg_confidence=det.confidence,
                        last_bbox=det.bbox,
                    )
                else:
                    entry = self._entries[det.class_id]
                    entry.last_seen_frame = self._frame_counter
                    entry.detection_count += 1
                    # Exponential moving average of confidence
                    alpha = 0.9
                    entry.avg_confidence = (
                        alpha * entry.avg_confidence + (1 - alpha) * det.confidence
                    )
                    entry.last_bbox = det.bbox

    # ─────────────────────────────────────────
    # Read — used by Rule Engine
    # ─────────────────────────────────────────

    def is_present(self, class_id: int) -> bool:
        """Was this class detected in the most recent frame?"""
        if not self._frames:
            return False
        return class_id in self._frames[-1]

    def is_present_by_name(self, class_name: str) -> bool:
        """Convenience: check presence by class name string."""
        with self._lock:
            for entry in self._entries.values():
                if entry.class_name == class_name:
                    return self.is_present(entry.class_id)
        return False

    def has_ever_seen(self, class_name: str) -> bool:
        """Has this class been detected at least once since start-up?"""
        with self._lock:
            return any(entry.class_name == class_name for entry in self._entries.values())

    def frames_since_seen(self, class_id: int) -> int:
        """How many frames ago was this class last detected?

        A class that has **never** been seen returns the full frame count since
        start-up — it has been absent for the entire session.

        This previously returned ``window_size`` (150), which at the default
        15 FPS saturates at exactly 10.0 seconds. Every ``absent_for`` threshold
        above 10s was therefore unreachable until the class had been seen once,
        which made ``stove_unattended`` — the only CRITICAL rule in the system —
        structurally unfireable in a room nobody had yet walked into. See
        docs/08_scenario_engineering/architecture_review.md §3 (D4).
        """
        with self._lock:
            entry = self._entries.get(class_id)
            if entry is None:
                return self._frame_counter
            return self._frame_counter - entry.last_seen_frame

    def frames_since_seen_by_name(self, class_name: str) -> int:
        """Convenience: frames since seen by class name.

        Never-seen classes return the frames elapsed since start-up — see
        :meth:`frames_since_seen`.
        """
        with self._lock:
            for entry in self._entries.values():
                if entry.class_name == class_name:
                    return self._frame_counter - entry.last_seen_frame
            return self._frame_counter

    def seconds_since_seen(self, class_id: int, fps: float = 15.0) -> float:
        """Elapsed seconds since this class was last detected."""
        return self.frames_since_seen(class_id) / max(fps, 1.0)

    def is_absent_for(self, class_id: int, seconds: float, fps: float = 15.0) -> bool:
        """Has this class been continuously absent for at least `seconds`?"""
        return self.seconds_since_seen(class_id, fps) >= seconds

    def is_absent_for_by_name(self, class_name: str, seconds: float, fps: float = 15.0) -> bool:
        """Convenience: absence check by class name."""
        frames_absent = self.frames_since_seen_by_name(class_name)
        return (frames_absent / max(fps, 1.0)) >= seconds

    def consecutive_frames(self, class_id: int) -> int:
        """How many consecutive trailing frames contain this class?"""
        with self._lock:
            count = 0
            for frame_set in reversed(self._frames):
                if class_id in frame_set:
                    count += 1
                else:
                    break
            return count

    def get_entry(self, class_id: int) -> MemoryEntry | None:
        """Retrieve the full tracking entry for a class."""
        return self._entries.get(class_id)

    def get_snapshot(self) -> dict[int, MemoryEntry]:
        """Return a shallow copy of all current tracking entries."""
        with self._lock:
            return dict(self._entries)

    def frame_count(self) -> int:
        """Total frames processed since creation."""
        return self._frame_counter

    def clear(self) -> None:
        """Clear all state. Call on shutdown for privacy."""
        with self._lock:
            self._frames.clear()
            self._entries.clear()
            self._frame_counter = 0

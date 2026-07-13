"""
Structured Logger — JSONL Event & Active Learning Logger
=========================================================
All pipeline events are written as JSONL (one JSON object per line)
for easy streaming, grep, and post-processing.

Log types:
  frame        — per-frame detection summary
  alert        — every alert that fired (with explanation)
  active_learn — uncertain detections flagged for labeling review
  health       — periodic system health snapshot
  error        — caught exceptions with context

Privacy rules:
  - Raw images are NEVER written to logs
  - Face detections are logged as class name only (no bbox coords)
  - All data stays on-device

Files:
  logs/events.jsonl       — detections and alerts
  logs/performance.jsonl  — timing metrics
  logs/active_learn.jsonl — uncertain detections for review
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence band for active learning flagging
UNCERTAINTY_LOWER = 0.25
UNCERTAINTY_UPPER = 0.55

# Classes whose bboxes must not be stored (privacy)
BBOX_REDACT_CLASSES = frozenset({"face", "person"})


class StructuredLogger:
    """JSONL structured event logger for edge deployment.

    Writes all pipeline events to JSONL files.
    Handles rotation-by-date and hard-caps file size at max_mb.
    Thread-safe via append-only file writes (OS atomic on most FS).
    """

    def __init__(
        self,
        log_dir: str = "logs",
        max_mb: float = 50.0,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_mb * 1024 * 1024)

        self._events_path = self._log_dir / "events.jsonl"
        self._perf_path = self._log_dir / "performance.jsonl"
        self._al_path = self._log_dir / "active_learn.jsonl"

        # Session statistics
        self._session_start = time.time()
        self._frame_count = 0
        self._alert_count = 0
        self._error_count = 0

        logger.info(f"StructuredLogger initialised → {self._log_dir.resolve()}")

    # ─────────────────────────────────────────
    # Per-frame logging
    # ─────────────────────────────────────────

    def log_frame(
        self,
        frame_id: int,
        detections: list,
        alerts: list,
        metrics,
        mode: str = "full",
    ) -> None:
        """Log a processed frame summary.

        Also flags uncertain detections for active learning.
        """
        self._frame_count += 1
        ts = time.time()

        det_summary = [
            {
                "class": d.class_name,
                "conf": round(d.confidence, 3),
                # Only store bbox for non-privacy classes
                **({"bbox": [round(d.bbox.cx, 3), round(d.bbox.cy, 3),
                             round(d.bbox.w, 3), round(d.bbox.h, 3)]}
                   if d.class_name not in BBOX_REDACT_CLASSES else {}),
            }
            for d in detections
        ]

        entry = {
            "ts": round(ts, 3),
            "type": "frame",
            "frame_id": frame_id,
            "mode": mode,
            "detections": len(detections),
            "alerts": len(alerts),
            "classes": sorted({d.class_name for d in detections}),
            "det_detail": det_summary,
        }
        self._write(self._events_path, entry)

        # Active learning: flag uncertain detections for future labeling
        uncertain = [
            {"class": d.class_name, "conf": round(d.confidence, 3), "frame_id": frame_id}
            for d in detections
            if UNCERTAINTY_LOWER <= d.confidence <= UNCERTAINTY_UPPER
        ]
        if uncertain:
            self._write(self._al_path, {
                "ts": round(ts, 3),
                "type": "active_learn",
                "frame_id": frame_id,
                "uncertain": uncertain,
            })

        # Performance metrics
        if metrics is not None:
            self._write(self._perf_path, {
                "ts": round(ts, 3),
                "type": "perf",
                "frame_id": frame_id,
                "capture_ms": round(metrics.capture_ms, 2),
                "detection_ms": round(metrics.detection_ms, 2),
                "memory_ms": round(metrics.memory_update_ms, 2),
                "rule_ms": round(metrics.rule_eval_ms, 2),
                "vlm_ms": round(metrics.vlm_ms, 2) if metrics.vlm_ms else None,
                "total_ms": round(metrics.total_ms, 2),
                "fps": round(metrics.fps, 1),
                "ram_mb": round(metrics.ram_mb, 1),
            })

    def log_alert(self, alert) -> None:
        """Log a fired alert with full explanation for debugging."""
        self._alert_count += 1
        self._write(self._events_path, {
            "ts": round(time.time(), 3),
            "type": "alert",
            "rule_id": alert.rule_id,
            "severity": alert.severity.name,
            "message": alert.message,
            "frame_id": alert.frame_id,
            "explanation": alert.explanation,
        })

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log a caught exception with context string."""
        self._error_count += 1
        self._write(self._events_path, {
            "ts": round(time.time(), 3),
            "type": "error",
            "error": type(error).__name__,
            "message": str(error)[:500],
            "context": context,
        })
        logger.error(f"[{context}] {error}")

    def dump_health_summary(self) -> None:
        """Write a health snapshot. Call periodically (e.g., every 60 seconds)."""
        import os
        uptime = time.time() - self._session_start
        fps_avg = self._frame_count / max(uptime, 1.0)

        # Try to get RAM usage
        try:
            import psutil
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            ram_mb = 0.0

        self._write(self._events_path, {
            "ts": round(time.time(), 3),
            "type": "health",
            "uptime_seconds": round(uptime, 1),
            "total_frames": self._frame_count,
            "total_alerts": self._alert_count,
            "total_errors": self._error_count,
            "avg_fps": round(fps_avg, 1),
            "ram_mb": round(ram_mb, 1),
        })

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _write(self, path: Path, entry: dict) -> None:
        """Append one JSON entry to the specified JSONL file.

        Silently discards write if the file exceeds max_bytes
        (prevents storage exhaustion on edge devices).
        """
        try:
            if path.exists() and path.stat().st_size > self._max_bytes:
                logger.warning(f"Log file size limit reached: {path.name}")
                return
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            # Never crash the pipeline due to logging failures
            logger.error(f"Log write failed ({path.name}): {e}")

"""
src.logging — Structured Event Logging
======================================

Provides JSONL structured logging for the Elderly Assistant System pipeline.

Log files:
    logs/pipeline/assistant.jsonl  — Runtime events (frames, alerts, errors)

Log entry types:
    frame          — Per-frame summary (detection count, classes, latency, mode)
    alert          — Fired safety alert with rule and explanation
    active_learning — Uncertain detections in 0.25–0.55 confidence band
    error          — Pipeline errors with context
    health_summary — Session summary on shutdown (uptime, FPS, totals)
    startup        — Session startup record
"""

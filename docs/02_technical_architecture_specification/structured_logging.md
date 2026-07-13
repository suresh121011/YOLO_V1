# Structured Logging & Active Learning

## Purpose

Event logging schema, log targets, rotation policy, and active learning sample mining.

## Dependencies

Reads:
- interfaces.md

Used By:
- orchestrator.md

Related:
- ../03_engineering_appendix/sample_logs.md

---

**File:** `src/logging/event_logger.py`

## Event Log Schema (JSON)

See [../03_engineering_appendix/sample_logs.md](../03_engineering_appendix/sample_logs.md) for full JSON examples.

**Key Fields:** `event_id`, `frame_id`, `timestamp_iso`, `event_type`, `detections[]`, `alert`, `pipeline_metrics`, `device_info`

## Log Targets

- **SQLite** (`logs/events.db`) — primary, always written
- **JSON lines** (`logs/events.jsonl`) — optional, for external tooling
- **Stderr** — ERROR level only

## Privacy

**No PII in logs.** Raw frames are never logged. Bounding box coordinates and class labels only.

## Log Rotation

SQLite DB rotated at 50MB. Old DB renamed with timestamp. Only last 3 rotations kept.

---

## Active Learning Logging

**File:** `src/logging/active_learning_logger.py`

**Purpose:** Automatically identify frames that would most improve model performance if annotated.

### Mining Strategy

| Sample Type | Condition | Value |
|:-----------|:----------|:------|
| Low-confidence detection | `0.25 < confidence < 0.50` | Model uncertain → annotation improves decision boundary |
| User-dismissed alert | Alert fired, user dismissed it | Hard negative — model false positive |
| Long-absent safety class | Safety class not detected for 30+ minutes | Possible false negative in deployment |

### Retraining Trigger

When 500+ new active learning samples are collected and manually verified, a retraining job is initiated. A/B testing validates the new model against the current production model before any rollout.

---

Previous: [plugin_architecture.md](./plugin_architecture.md)

Next: [feature_flags.md](./feature_flags.md)

Related: [../03_engineering_appendix/sample_logs.md](../03_engineering_appendix/sample_logs.md)

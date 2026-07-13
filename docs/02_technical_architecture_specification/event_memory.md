# Event Memory

## Purpose

Sliding-window temporal memory for tracking object presence over time. Enables temporal rules like "stove unattended for 30 seconds."

## Dependencies

Reads:
- interfaces.md

Used By:
- rule_engine.md
- orchestrator.md

Related:
- confidence_fusion.md

---

## Design

Fixed-size sliding window (configurable depth, default = 150 frames at 15 FPS ≈ 10 seconds).

**File:** `src/pipeline/event_memory.py`

```
Frame Window (most recent N frames):
┌─────────────────────────────────────────────────────┐
│  Frame N-149  ...  Frame N-10  ...  Frame N-1  Frame N │
│  [knife ✓]         [stove ✓]        [stove ✓]  [stove ✓] │
└─────────────────────────────────────────────────────┘
→ stove present for 10+ frames
→ knife not present for 139 frames
```

## Key Methods

| Method | Signature | Description |
|:-------|:----------|:------------|
| `update` | `(detections: list[Detection]) → None` | Advance window with new frame |
| `is_present` | `(class_id: int) → bool` | Was class seen in last N frames? |
| `frames_since_seen` | `(class_id: int) → int` | How many frames ago was it last detected? |
| `consecutive_frames` | `(class_id: int) → int` | How many consecutive recent frames contains this class? |
| `get_snapshot` | `() → dict[int, MemoryEntry]` | Full state snapshot |

## Memory Budget

`window_size × num_classes × 4 bytes ≈ 150 × 23 × 4 = ~14KB` (negligible)

## Thread Safety

Uses `threading.Lock` for all read/write operations. Lock is held only during dictionary updates — never during I/O.

> For full implementation, see [../03_engineering_appendix/python_examples.md](../03_engineering_appendix/python_examples.md)

---

Previous: [interfaces.md](./interfaces.md)

Next: [rule_engine.md](./rule_engine.md)

Related: [orchestrator.md](./orchestrator.md), [confidence_fusion.md](./confidence_fusion.md)

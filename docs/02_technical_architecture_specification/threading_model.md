# Threading Model

## Purpose

Multi-thread architecture, synchronization primitives, and deadlock prevention strategy.

## Dependencies

Reads:
- orchestrator.md

Used By:
- error_handling.md

Related:
- performance_budget.md

---

## Architecture: Three Threads Minimum

```
Thread 1 (Main): Frame capture → YOLO → Memory → Rule evaluation → Queue management → Logger
Thread 2 (TTS):  Alert queue consumer → Piper TTS → Audio playback
Thread 3 (VLM):  VLM request queue consumer → SmolVLM2 inference → Result queue
```

## Synchronization

- `threading.Lock` on alert queue
- `threading.Lock` on event memory (reader-writer pattern: many readers, one writer)
- `queue.Queue` (thread-safe) for TTS and VLM inter-thread communication
- `threading.Event` for graceful shutdown signaling

## Deadlock Prevention

- All locks have explicit timeouts (500ms)
- Lock acquisition order is documented and enforced: **memory → queue → logger**
- No lock held during I/O (TTS file write, SQLite write)

## TTS Non-blocking

TTS runs in Thread 2. Main thread never blocks on speech output. Alert is enqueued and main thread continues processing frames immediately.

---

Previous: [performance_budget.md](./performance_budget.md)

Next: [error_handling.md](./error_handling.md)

Related: [orchestrator.md](./orchestrator.md)

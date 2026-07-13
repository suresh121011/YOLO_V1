# Performance Budget & Profiling

## Purpose

Latency budget per component, memory budget, and profiling instrumentation.

## Dependencies

Reads:
- orchestrator.md

Used By:
- error_handling.md

Related:
- threading_model.md
- ../01_executive_implementation_plan/architecture_overview.md

---

## Component Latency Budget

| Component | CPU Budget | GPU/NPU Budget | Measurement Method |
|:----------|:-----------|:---------------|:------------------|
| Frame capture | 5 ms | 5 ms | `time.perf_counter()` |
| Preprocessing | 3 ms | 2 ms | `time.perf_counter()` |
| YOLO11n inference | 120 ms | 20 ms | Ultralytics benchmark |
| Detection filtering | 1 ms | 1 ms | `time.perf_counter()` |
| Memory update | 2 ms | 2 ms | `time.perf_counter()` |
| Rule engine | 5 ms | 5 ms | `time.perf_counter()` |
| Queue management | 1 ms | 1 ms | `time.perf_counter()` |
| **Total (no VLM)** | **137 ms (~7 FPS)** | **36 ms (~28 FPS)** | |
| SmolVLM2 (every 5th) | 1,500 ms amortized | 400 ms amortized | Separate thread |
| Piper TTS | 400 ms | 400 ms | Separate thread |

> [!NOTE]
> On CPU-only Android (no NPU), effective pipeline FPS will be ~7-10 FPS. This is acceptable for V1 — alert response time is still within the 2-second budget because TTS runs in a separate thread.

## Memory Budget

| Component | RAM Budget |
|:----------|:----------|
| YOLO11n model (PT) | ~6 MB |
| YOLO11n (ONNX/TFLite) | ~3.5 MB |
| SmolVLM2-256M | ~500 MB |
| SmolVLM2-500M | ~1 GB |
| Event Memory | ~1 MB |
| OpenCV frame buffer | ~6 MB (640×640×3) |
| SQLite logger | ~10 MB |
| **Total (without VLM)** | **~30 MB** |
| **Total (with VLM-256M)** | **~550 MB** |

## Profiling Hooks

- `metrics_collector.py` records per-component timing for every frame
- Rolling 100-frame average FPS computed and logged every 10 seconds
- RAM usage sampled via `psutil.Process().memory_info().rss` every 30 seconds
- Results written to `logs/performance.jsonl`

---

Previous: [feature_flags.md](./feature_flags.md)

Next: [threading_model.md](./threading_model.md)

Related: [../01_executive_implementation_plan/architecture_overview.md](../01_executive_implementation_plan/architecture_overview.md)

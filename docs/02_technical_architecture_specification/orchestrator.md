# Orchestrator

## Purpose

Top-level pipeline coordinator. Drives the frame processing loop and coordinates all components.

## Dependencies

Reads:
- event_memory.md
- rule_engine.md
- confidence_fusion.md
- threading_model.md

Used By:
- plugin_architecture.md
- structured_logging.md

Related:
- error_handling.md
- performance_budget.md

---

**File:** `src/pipeline/orchestrator.py`

## Frame Processing Loop

```
Main Thread:                   TTS Thread:              VLM Thread (optional):
────────────────────────────   ─────────────────────   ─────────────────────────
while running:                 while running:           while running:
  frame = camera.read()          text = tts_queue.get()  request = vlm_queue.get()
  detections = detector(frame)   piper.speak(text)        result = vlm.analyze(...)
  memory.update(detections)      audio.play()             vlm_results.put(result)
  vlm_queue.put(frame)  ──────────────────────────────────────────────────────►
  context = vlm_results.get()  ◄──────────────────────────────────────────────
  alerts = rules(detections, memory, context)
  queue.push(alerts)
  tts_queue.put(best_alert.message)  ──────────────────────────────────────────►
  logger.log(frame_id, detections, alerts)
  metrics.record(timings)
```

## Graceful Shutdown

SIGTERM/SIGINT handler flushes the alert queue, finishes current TTS, and persists logs before exit.

## Frame Skip Logic

```
if thermal_mode == "throttled":
    process_every_n = 4    # 7.5 FPS effective
elif battery_mode == "low":
    process_every_n = 2    # 15 FPS effective
else:
    process_every_n = 1    # full FPS
```

> For full implementation, see [../03_engineering_appendix/python_examples.md](../03_engineering_appendix/python_examples.md)

---

Previous: [confidence_fusion.md](./confidence_fusion.md)

Next: [plugin_architecture.md](./plugin_architecture.md)

Related: [threading_model.md](./threading_model.md), [error_handling.md](./error_handling.md)

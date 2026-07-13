# Component Specifications: YOLO, VLM, TTS, Alert Queue

## Purpose

Detailed specifications for YOLO Detector, SmolVLM2 Analyzer, Piper TTS Engine, and Alert Queue.

## Dependencies

Reads:
- interfaces.md

Used By:
- api_contracts.md

Related:
- event_memory.md
- rule_engine.md
- ../03_engineering_appendix/python_examples.md

---

## YOLO Detector

**File:** `src/pipeline/detector.py`

**Responsibilities:** Load YOLO11n model (PT, ONNX, or TFLite), run inference, apply confidence threshold and NMS, return typed `Detection` list.

**Design Decisions:**
- Uses Ultralytics wrapper for PT format (training/development)
- Falls back to ONNX Runtime for production edge deployment
- TFLite delegate used for Android NPU acceleration
- Model hash verified on load to detect corruption

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `conf_threshold` | 0.25 | Minimum confidence to accept detection |
| `iou_threshold` | 0.45 | NMS IoU threshold |
| `imgsz` | 640 | Inference resolution |
| `device` | auto | cuda / cpu / mps / npu |
| `max_det` | 100 | Maximum detections per frame |

**Safety-class specific thresholds** (overrides default):

| Class | Threshold | Reason |
|:------|:----------|:-------|
| `wet_floor` | 0.20 | Prefer recall over precision |
| `knife` | 0.20 | Critical — lower threshold |
| `gas_cylinder` | 0.22 | Critical safety class |
| `wire` | 0.22 | Tripping hazard |
| `passport` | 0.30 | Avoid false positives on documents |

---

## SmolVLM2 Scene Analyzer

**File:** `src/pipeline/scene_analyzer.py`

| Variant | VRAM | Latency | Use Case |
|:--------|:-----|:--------|:---------|
| SmolVLM2-256M | ~1.5 GB | ~800ms CPU | Raspberry Pi / low-end Android |
| SmolVLM2-500M | ~2.5 GB | ~1,200ms CPU | Android flagship |
| SmolVLM2-2.2B | ~5.2 GB | ~2,500ms CPU | Development / evaluation only |

**Invocation Strategy:** Every 5th frame; skipped if `vlm_enabled = false`; 2,000ms timeout.

**Safety Constraint:** VLM response is advisory only. Rule Engine operates on YOLO detections first. VLM can upgrade alert severity but cannot cancel a rule-engine-triggered alert.

---

## Alert Queue

**File:** `src/pipeline/alert_queue.py`

- Min-heap ordered by `severity` (CRITICAL first)
- Singleton pattern — one queue for the pipeline lifetime
- Thread-safe (`threading.Lock`)
- Limits queue depth to 10 pending alerts (oldest LOW/INFO dropped first)

```
Queue State Example:
[CRITICAL: stove_unattended] ← highest priority, speaks first
[HIGH: knife_near_person]
[HIGH: wet_floor_hazard]
[INFO: medicine_reminder]    ← lowest priority
```

---

## Piper TTS Engine

**File:** `src/pipeline/tts_engine.py`

**Selection Rationale over pyttsx3:**
- Neural VITS architecture → natural, non-robotic speech (critical for elderly acceptance)
- 100% offline on CPU
- Indian English voice model available (`en_IN-medium`)
- Latency < 500ms for short sentences (< 20 words)

**Key Behaviors:** Non-blocking speak (runs in separate thread), queue incoming requests, health check on init, fallback beep via `sounddevice`.

**V2 Extension:** Hindi voice model (`hi_IN-medium`) addition requires only config change.

---

Previous: [error_handling.md](./error_handling.md)

Next: [api_contracts.md](./api_contracts.md)

Related: [../03_engineering_appendix/python_examples.md](../03_engineering_appendix/python_examples.md)

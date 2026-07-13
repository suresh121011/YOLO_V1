# API Reference

## Purpose

Pipeline module API table and SmolVLM2 prompt/response contract.

## Dependencies

Reads:
- python_examples.md

Used By:
- troubleshooting.md

Related:
- ../02_technical_architecture_specification/interfaces.md
- ../02_technical_architecture_specification/data_contracts.md

---

## 9.1 Pipeline Module API

| Module | Method | Input | Output |
|:-------|:-------|:------|:-------|
| `detector` | `detect(frame)` | `np.ndarray (H,W,3)` | `list[Detection]` |
| `detector` | `warmup()` | None | `None` |
| `event_memory` | `update(detections)` | `list[Detection]` | `None` |
| `event_memory` | `is_present(class_id)` | `int` | `bool` |
| `event_memory` | `seconds_since_seen(class_id, fps)` | `int, float` | `float` |
| `event_memory` | `is_absent_for(class_id, seconds, fps)` | `int, float, float` | `bool` |
| `scene_analyzer` | `analyze(frame, detections)` | `np.ndarray, list[Detection]` | `SceneContext` |
| `scene_analyzer` | `is_available()` | None | `bool` |
| `confidence_fusion` | `fuse(detections, context)` | `list[Detection], SceneContext?` | `list[Detection]` |
| `rule_engine` | `evaluate(detections, memory, context)` | `list[Detection], EventMemory, dict?` | `list[Alert]` |
| `rule_engine` | `reload_rules()` | None | `None` |
| `alert_queue` | `push(alert)` | `Alert` | `None` |
| `alert_queue` | `pop()` | None | `Optional[Alert]` |
| `tts_engine` | `speak(text, priority)` | `str, bool` | `None` |
| `tts_engine` | `is_speaking()` | None | `bool` |
| `tts_engine` | `health_check()` | None | `bool` |
| `tts_engine` | `shutdown()` | None | `None` |
| `event_logger` | `log(frame_id, detections, alerts)` | `int, list, list` | `None` |
| `event_logger` | `flush()` | None | `None` |
| `metrics_collector` | `record(**timings)` | keyword args | `None` |

---

## 9.2 SmolVLM2 Prompt/Response Contract

### System Prompt

```
You are an elderly safety assistant analyzing a home environment.
You must identify safety risks for an elderly person.
Always respond in the exact JSON format specified.
Never include personally identifiable information.
```

### User Prompt Template

```
Detected objects: [{"class": "knife", "conf": 0.87, "bbox": [0.45, 0.60, 0.12, 0.08]}, {"class": "person", "conf": 0.95, "bbox": [0.50, 0.50, 0.30, 0.70]}]

Answer in JSON:
{"activity": "string", "risks": [{"risk_type": "string", "severity": "low|medium|high|critical", "description": "string"}], "recommendations": ["string"]}
```

### Expected Response

```json
{
  "activity": "standing in kitchen near counter",
  "risks": [
    {
      "risk_type": "knife",
      "severity": "high",
      "description": "Sharp knife within arm's reach of elderly person"
    }
  ],
  "recommendations": [
    "Move knife to a secure drawer",
    "Keep sharp objects away from counter edge"
  ]
}
```

### Response Validation Rules

| Field | Validation |
|:------|:-----------|
| `activity` | Non-empty string |
| `risks` | Array (may be empty `[]` if no risk) |
| `risks[].risk_type` | Must match a known class_name |
| `risks[].severity` | Must be one of: `low`, `medium`, `high`, `critical` |
| `recommendations` | Array of strings (may be empty) |

If VLM response fails JSON validation, it is discarded and `context = None` is passed to Rule Engine.

---

Previous: [sample_logs.md](./sample_logs.md)

Next: [annotation_guide.md](./annotation_guide.md)

Related: [../02_technical_architecture_specification/interfaces.md](../02_technical_architecture_specification/interfaces.md)

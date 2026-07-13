# Sample JSON Logs

## Purpose

Reference JSON log examples for all event types emitted by the pipeline.

## Dependencies

Reads:
- python_examples.md

Used By:
- api_reference.md
- troubleshooting.md

Related:
- ../02_technical_architecture_specification/structured_logging.md

---

## 8.1 Detection Event Log

```json
{
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "frame_id": 12345,
  "timestamp_iso": "2026-07-12T10:30:00.123+05:30",
  "timestamp_ms": 1752320400123,
  "event_type": "DETECTION",
  "detections": [
    {
      "class_id": 5,
      "class_name": "knife",
      "confidence": 0.87,
      "bbox": {"cx": 0.45, "cy": 0.60, "w": 0.12, "h": 0.08}
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {"cx": 0.50, "cy": 0.50, "w": 0.30, "h": 0.70}
    }
  ],
  "alert": null,
  "pipeline_metrics": {
    "detection_ms": 22.4,
    "memory_ms": 0.8,
    "rule_eval_ms": 1.2,
    "total_ms": 24.4,
    "fps": 14.8,
    "ram_mb": 1240
  }
}
```

---

## 8.2 Alert Fired Event Log

```json
{
  "event_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "frame_id": 12348,
  "timestamp_iso": "2026-07-12T10:30:00.323+05:30",
  "timestamp_ms": 1752320400323,
  "event_type": "ALERT_FIRED",
  "detections": [
    {
      "class_id": 5,
      "class_name": "knife",
      "confidence": 0.87,
      "bbox": {"cx": 0.45, "cy": 0.60, "w": 0.12, "h": 0.08}
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {"cx": 0.50, "cy": 0.50, "w": 0.30, "h": 0.70}
    }
  ],
  "alert": {
    "rule_id": "knife_near_person",
    "severity": "HIGH",
    "message": "Please be careful, there is a knife nearby.",
    "cooldown_seconds": 60,
    "cooldown_expires_iso": "2026-07-12T10:31:00.323+05:30"
  },
  "pipeline_metrics": {
    "detection_ms": 23.1,
    "memory_ms": 0.9,
    "rule_eval_ms": 1.5,
    "tts_ms": 380.0,
    "total_ms": 405.5,
    "fps": 14.2,
    "ram_mb": 1245
  }
}
```

---

## 8.3 Active Learning Sample Log

```json
{
  "event_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "sample_type": "low_confidence",
  "frame_id": 45678,
  "timestamp_iso": "2026-07-12T11:15:30.456+05:30",
  "class_name": "wet_floor",
  "class_id": 20,
  "confidence": 0.31,
  "bbox": {"cx": 0.50, "cy": 0.72, "w": 0.40, "h": 0.15},
  "reason": "Confidence below mining threshold (0.50)",
  "frame_saved": false,
  "notes": "Candidate for manual review and annotation"
}
```

---

## 8.4 Performance Metrics Log

```json
{
  "timestamp_iso": "2026-07-12T10:30:10.000+05:30",
  "window_frames": 100,
  "avg_fps": 14.5,
  "avg_detection_ms": 23.2,
  "avg_total_ms": 26.8,
  "p95_total_ms": 31.4,
  "max_total_ms": 45.2,
  "ram_mb": 1240,
  "cpu_percent": 55.2,
  "alerts_fired": 2,
  "detections_total": 328,
  "device_temp_celsius": 38.5
}
```

---

## 8.5 SmolVLM2 Scene Analysis Log

```json
{
  "event_id": "d4e5f6a7-b8c9-0123-defa-234567890123",
  "frame_id": 12350,
  "timestamp_iso": "2026-07-12T10:30:00.500+05:30",
  "event_type": "VLM_ANALYSIS",
  "vlm_response": {
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
  },
  "inference_time_ms": 850.3
}
```

---

Previous: [dvc_pipeline.md](./dvc_pipeline.md)

Next: [api_reference.md](./api_reference.md)

Related: [../02_technical_architecture_specification/structured_logging.md](../02_technical_architecture_specification/structured_logging.md)

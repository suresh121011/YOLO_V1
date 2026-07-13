# Configuration Architecture

## Purpose

Config loading hierarchy, schema, and validation approach.

## Dependencies

Reads:
- data_contracts.md

Used By:
- deployment_architecture.md

Related:
- feature_flags.md
- ../03_engineering_appendix/yaml_examples.md

---

**File:** `src/config/loader.py`

## Config Hierarchy

```
system_defaults.yaml     ← baked-in defaults (never user-modified)
     ↓ overridden by
configs/deployment/*.yaml  ← device-specific overrides
     ↓ overridden by
~/.elderly_assistant/user.yaml  ← user/caregiver preferences (V2)
```

## Config Schema (Key Sections)

```yaml
pipeline:
  camera_fps: 30
  process_every_n_frames: 1
  confidence_threshold_default: 0.25
  vlm_enabled: false
  vlm_invoke_every_n_frames: 5
  max_alert_queue_depth: 10
  event_memory_window_frames: 150

tts:
  voice_model: en_IN-medium
  model_path: models/tts/en_IN-medium.onnx
  config_path: models/tts/en_IN-medium.json
  speech_rate: 0.9

logging:
  level: INFO
  output: sqlite
  db_path: logs/events.db
  active_learning_enabled: true
  low_confidence_threshold: 0.50

feature_flags:
  vlm_enabled: false
  hindi_tts: false
  caregiver_sync: false
  thermal_monitoring: true
```

## Config Validation

All configs validated against a Pydantic schema on load. Invalid configs raise `ConfigValidationError` with human-readable messages. Pipeline does not start with invalid config.

> For full config templates, see [../03_engineering_appendix/yaml_examples.md](../03_engineering_appendix/yaml_examples.md)

---

Previous: [data_contracts.md](./data_contracts.md)

Next: [deployment_architecture.md](./deployment_architecture.md)

Related: [feature_flags.md](./feature_flags.md)

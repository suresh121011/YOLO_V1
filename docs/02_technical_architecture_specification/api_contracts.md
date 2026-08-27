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

> **This section is a Phase-2 sketch, not the shipped file.** Corrected note added in Phase-6 M9 rather
> than rewriting the sketch, so the drift is visible instead of quietly patched. The real
> `configs/feature_flags.yaml` has four roots — `components:`, `classes:`, `rules:`, `runtime:` — and
> none of the roots below. `vlm_enabled` exists in no config file (the orchestrator read that key for
> the VLM until Phase-6 M1; the real one is `components.smolvlm_analysis`). For what is actually read,
> and which keys are inert, see [feature_flags.md](./feature_flags.md).

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

**Not as built.** There is no Pydantic schema and no `ConfigValidationError` — pydantic is not a
dependency of this project, and configs are hand-validated in the house style. `SystemConfig.load()`
falls back to the defaults in `src/config/config_loader.py` for missing keys and ignores unknown ones.
Validation that *is* enforced lives where the values are used: the scenario schema rejects an invalid
scenario at compile time, and the detector refuses weights that disagree with `configs/data.yaml`.

> For full config templates, see [../03_engineering_appendix/yaml_examples.md](../03_engineering_appendix/yaml_examples.md)

---

Previous: [data_contracts.md](./data_contracts.md)

Next: [deployment_architecture.md](./deployment_architecture.md)

Related: [feature_flags.md](./feature_flags.md)

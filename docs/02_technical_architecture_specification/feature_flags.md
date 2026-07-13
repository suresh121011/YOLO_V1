# Feature Flags

## Purpose

Runtime feature toggles that enable/disable pipeline capabilities without code changes.

## Dependencies

Reads:
- orchestrator.md

Used By: None (consumed at runtime)

Related:
- plugin_architecture.md
- ../03_engineering_appendix/yaml_examples.md

---

**File:** `configs/feature_flags.yaml`

## Flag Definitions

| Flag | Type | Default | Effect |
|:-----|:-----|:--------|:-------|
| `vlm_enabled` | bool | false | Enable/disable SmolVLM2 inference |
| `hindi_tts` | bool | false | Use Hindi TTS voice (V2) |
| `caregiver_sync` | bool | false | Enable WiFi log sync (V2) |
| `thermal_monitoring` | bool | true | Monitor device temperature |
| `active_learning` | bool | true | Log low-confidence detections |
| `rule_hot_reload` | bool | false | Enable live rule updates from YAML |
| `debug_overlay` | bool | false | Show detection overlay on screen |
| `performance_logging` | bool | true | Log per-frame timing metrics |

## Loading Behavior

Flags are read at startup and can be hot-reloaded if `rule_hot_reload=true`. All flags are type-validated.

> For full YAML template, see [../03_engineering_appendix/yaml_examples.md](../03_engineering_appendix/yaml_examples.md)

---

Previous: [structured_logging.md](./structured_logging.md)

Next: [performance_budget.md](./performance_budget.md)

Related: [plugin_architecture.md](./plugin_architecture.md)

# Rule Engine

## Purpose

YAML-driven safety rule evaluation engine. Evaluates configurable rules against detections and event memory, generating typed Alert objects.

## Dependencies

Reads:
- interfaces.md
- event_memory.md

Used By:
- orchestrator.md

Related:
- confidence_fusion.md
- ../03_engineering_appendix/yaml_examples.md

---

**File:** `src/pipeline/rule_engine.py`

## Rule Anatomy

```
Rule ID → Condition (detection + memory state) → Severity → Message → Cooldown
```

## Condition Types (V1)

- `detected(class_name)` — class present in current frame
- `absent_for(class_name, seconds)` — class not seen for N seconds
- `near(class_a, class_b, pixel_dist)` — two detections within pixel distance
- `any_of(class_list)` — any class from list detected

## Rule Evaluation Order

1. CRITICAL rules
2. HIGH rules
3. MEDIUM rules
4. LOW rules
5. INFO rules

## Cooldown Mechanism

Each rule maintains a `last_fired_timestamp`. Alert suppressed if `now - last_fired < cooldown_seconds`.

## Rule Re-loading

Rules can be hot-reloaded from YAML without restart via `reload_rules()` — protected by a read-write lock.

## Built-in Rules (V1)

| Rule ID | Condition | Severity | Cooldown |
|:--------|:----------|:---------|:---------|
| `knife_near_person` | `detected(knife)` AND `detected(person)` | HIGH | 60s |
| `stove_unattended` | `detected(stove)` AND `absent_for(person, 30s)` | CRITICAL | 30s |
| `wet_floor_hazard` | `detected(wet_floor)` | HIGH | 120s |
| `wire_tripping_hazard` | `detected(wire)` AND `detected(person)` | HIGH | 120s |
| `gas_cylinder_check` | `detected(gas_cylinder)` AND NOT `detected(stove)` | INFO | 600s |
| `medicine_reminder` | `any_of([medicine_strip, medicine_bottle])` | INFO | 300s |

> For full YAML rule definitions, see [../03_engineering_appendix/yaml_examples.md](../03_engineering_appendix/yaml_examples.md)
> For full Python implementation, see [../03_engineering_appendix/python_examples.md](../03_engineering_appendix/python_examples.md)

---

Previous: [event_memory.md](./event_memory.md)

Next: [confidence_fusion.md](./confidence_fusion.md)

Related: [orchestrator.md](./orchestrator.md)

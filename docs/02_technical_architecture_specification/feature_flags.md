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

## Flag Definitions — `components:`

Corrected in Phase-6 M9 against the code. This table previously named `vlm_enabled`, which exists in no
config file (it was the key the orchestrator read for the VLM — defect 3), and `active_learning`, whose
real name is `active_learning_logging`. **Wired** means some code in `src/` actually reads it;
`tests/unit/test_feature_flags_are_live.py` fails if this column drifts again.

| Flag | Default | Wired | Effect |
|:-----|:--------|:-----:|:-------|
| `yolo_detection` | true | — | Structural. Without the detector there is no pipeline to run |
| `smolvlm_analysis` | false | yes | Enable SmolVLM2 inference |
| `tts_output` | true | yes | When false the device is genuinely silent (wired in M9) |
| `performance_logging` | true | yes | Log per-frame timing metrics |
| `active_learning_logging` | true | — | Uncertain-band capture is V2 |
| `thermal_monitoring` | false | — | Phase 7 field testing |
| `rule_hot_reload` | false | — | `reload_rules()` works; nothing calls it on a timer |
| `caregiver_sync` | false | — | Governs *remote* delivery. The local caregiver sink is always on |
| `hindi_tts` | false | — | Superseded by `runtime.tts_language` + per-scenario `messages` |
| `debug_overlay` | false | — | No render surface in the headless pipeline |

Per-class toggles (`classes:`) are enforced in the detector: a disabled class never becomes a
`Detection`, so it reaches no rule, no log and no alert. Per-scenario toggles (`rules:`) name scenario
ids and are applied when the engine loads; they are kill switches, not the clinical review gate.

## Loading Behavior

Flags are read at startup by `SystemConfig.load()`. Missing keys fall back to the defaults in
`src/config/config_loader.py`; unknown keys are ignored.

There is **no type validation and no hot reload**, contrary to what this section claimed before M9.
`ScenarioRuleEngine.reload_rules()` exists and works, but nothing calls it on a timer or a file watch —
which is why `rule_hot_reload` is listed as not wired above.

> For full YAML template, see [../03_engineering_appendix/yaml_examples.md](../03_engineering_appendix/yaml_examples.md)

---

Previous: [structured_logging.md](./structured_logging.md)

Next: [performance_budget.md](./performance_budget.md)

Related: [plugin_architecture.md](./plugin_architecture.md)

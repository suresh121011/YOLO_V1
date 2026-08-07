# Architecture Review — Runtime As Built vs. Runtime As Documented

## Purpose

Phase 1 deliverable. Establishes what `src/pipeline/` actually does today, where a scenario knowledge
layer plugs in, what must be repaired before that layer can function, and the layering rules the new
package must obey.

## Dependencies

Reads:
- `../02_technical_architecture_specification/rule_engine.md`
- `../02_technical_architecture_specification/interfaces.md`
- `../02_technical_architecture_specification/performance_budget.md`

Used By:
- `adr/ADR-P6-03-structured-ast-over-string-dsl.md`
- `adr/ADR-P6-04-rule-engine-injection.md`
- Milestone M1

---

## 1. What exists

| Component | File | Role |
|:---|:---|:---|
| Data contracts | `src/pipeline/__init__.py` | `Severity`, `BoundingBox`, `Detection`, `SceneContext`, `Alert`, `MemoryEntry`; abstract `BaseRuleEngine` at `:357-371`. Declared **LOCKED** at `:8` |
| Detector | `src/pipeline/detector.py` | `YOLODetector.detect(frame, frame_id) -> list[Detection]` |
| Temporal state | `src/pipeline/event_memory.py` | Sliding window of `set[class_id]` + per-class `MemoryEntry` |
| Rule evaluator | `src/pipeline/rule_engine.py` | Regex DSL over a condition **string**, per-rule cooldowns, `RLock`, `reload_rules()` |
| Alert arbitration | `src/pipeline/alert_queue.py` | Severity-ordered bounded queue with low-priority eviction. **Fully built, unit-tested, and never imported by the orchestrator** |
| Speech sink | `src/pipeline/tts_engine.py` | Piper TTS, `PriorityQueue(maxsize=5)` |
| VLM | `src/pipeline/scene_analyzer.py` | SmolVLM2 → `SceneContext`; advisory only, cannot cancel an alert |
| Fusion | `src/pipeline/confidence_fusion.py` | `SAFETY_CLASSES` hardcoded at `:30-40` (7 of 23) |
| Plugin base | `src/pipeline/plugin_base.py` | `AnalysisPlugin` ABC. See defect 10 |
| Coordinator | `src/pipeline/orchestrator.py` | Frame loop: detect → memory → VLM → fuse → rules → plugins → speak → log |
| Rules data | `configs/risk_rules.yaml` | 6 hand-written rules |

`src/pipeline/__init__.py` imports nothing from `src` and pulls no ML dependency — heavy imports are
lazy inside submodules. It is a genuine leaf, which is what makes the layering in §4 cycle-free.

## 2. Spec drift

The specification documents describe a system that was never built:

| Documented | Reality |
|:---|:---|
| `near(class_a, class_b, pixel_dist)` is a V1 condition type (`rule_engine.md:34`) | Never implemented. Falls through to the unrecognised-condition branch and returns `False` |
| Rules evaluate CRITICAL → HIGH → MEDIUM → LOW → INFO (`rule_engine.md:37-43`) | `rule_engine.py:95` iterates YAML file order |
| "All runtime behavior is controlled here. No code changes required." (`configs/feature_flags.yaml:3`) | Every flag is inert. See defect 1 |
| "Disable individual safety rules without touching code" (`feature_flags.yaml:52`) | No rule-enable map is ever read |
| `passport: false  # Disabled by default — privacy` (`feature_flags.yaml:35`) | Not enforced anywhere |
| `max_alerts_per_minute: 6  # Hard cap — prevents alert fatigue` (`feature_flags.yaml:77`) | Referenced by no code |
| Safety classes prefer recall via low thresholds (`class_thresholds.yaml:8-10`) | Structurally unreachable. See defect 5 |

This matters more than the individual bugs: a config comment that lies is worse than a missing feature,
because it stops anyone from looking.

## 3. The ten defects

Each was verified directly against the code, not inferred. **D1–D9 were fixed in milestone M1**, each
with the regression test that pins it; D10 is documented and deliberately left as found (see §4).

**D1 — Every feature flag is dead.** `orchestrator.py:49` does
`yaml.safe_load(...).get("feature_flags", {})`, but `configs/feature_flags.yaml` has no
`feature_flags:` root — its roots are `components:`, `classes:`, `rules:`, `runtime:`. `self._flags` is
therefore permanently `{}`. `src/config/config_loader.py` implements all of this correctly
(`is_class_enabled`, `is_rule_enabled`, `get_class_threshold`) and is instantiated only by tests; the
production path bypasses it entirely.

**D2 — The privacy flag is a false claim.** Consequence of D1, and independently true: `detector.py`
never consults a class-enable map either. Passport detections are produced and logged.

**D3 — The VLM always loads.** `orchestrator.py:97` reads `self._flags.get("vlm_enabled", True)` — a key
that appears in no configuration file — so `components.smolvlm_analysis: false` has no effect.

**D4 — The only CRITICAL rule cannot fire cold.** `event_memory.py:108-114`
(`frames_since_seen_by_name`) returns `self._window` (150) when a class has never been seen. At
`target_fps: 15` that is exactly 10.0 s. `stove_unattended` requires `absent_for(person, 30)`, and
`10.0 >= 30` is false — permanently. The rule only ever works if a person was detected earlier in the
session (entries are never pruned). Point the device at a kitchen before anyone walks in, and the one
CRITICAL rule in the system is structurally unfireable.

**D5 — Recall-tuned thresholds are unreachable.** `detector.py:131-136` passes `conf=self.conf_threshold`
(global `0.25`) into `model.predict`, so Ultralytics filters *before* the per-class check at `:146`.
`knife: 0.20` and `wet_floor: 0.20` cannot lower anything; per-class thresholds can only ever be
stricter. Separately `orchestrator.py:87` constructs `YOLODetector(model_path=...)` without
`class_thresholds`, so `configs/class_thresholds.yaml` is never loaded at runtime and the hardcoded
`DEFAULT_CLASS_THRESHOLDS` at `detector.py:27-38` wins — and disagrees with the YAML.

**D6 — The condition parser is wrong in four ways.** `rule_engine.py:139-201`:
- `:164` `re.search`es for `any_of([...])` across the **whole** string and returns at `:167`, *before*
  the AND/OR split at `:170`/`:177`. So `any_of([a, b]) AND detected(person)` silently discards the
  conjunct.
- AND is split before OR, giving AND *lower* precedence than OR — inverted from standard semantics.
- No parentheses. A leading `(` defeats every `re.match`-anchored predicate, falling through to `:200`
  → warn → `False`.
- `NOT` composes only with `detected` (`:184`). `NOT any_of(...)` and every future predicate return
  `False`.

**D7 — One bad severity string kills all rules, silently.** `rule_engine.py:110` `Severity[...]` raises
`KeyError` on a malformed value; `orchestrator.py:188-192` catches `Exception`, logs, and sets
`alerts = []`. A single typo turns the device into a camera that never speaks, with no user-visible
signal. **There are zero tests exercising the rule engine** — `tests/integration/conftest.py:21` defines
a `risk_rules_path` fixture that nothing uses.

**D8 — `save_csv_report` writes CRLF.** `report_utils.py:122` opens with `newline=""` and omits
`lineterminator="\n"`, so `csv.DictWriter` emits CRLF. Its JSON sibling at `:88` documents at length why
LF is mandatory (DVC hashes on-disk bytes; `.gitattributes` normalises to LF). Any committed CSV would
be permanently dirty in `git status`, failing release gate RG5 and diverging between the CI matrix legs.

**D9 — Temporal predicates are capped at 10 s.** `memory_window_frames: 150` ÷ `target_fps: 15`.
`consecutive_frames` (`event_memory.py:129-138`) is exactly what a dwell predicate needs and no rule
calls it. Also: `orchestrator.py:189` passes `self._target_fps` — the *nominal* 15 — not the measured
rate, so thermal throttling silently rescales every temporal threshold.

**D10 — The plugin seam is broken three ways.** `orchestrator.py:198` calls `plugin.on_frame(frame)`;
`plugin_base.py:70-87` defines `analyze(frame, detections, memory)`;
`../02_technical_architecture_specification/plugin_architecture.md:24-35` defines a third contract.
No `register_plugin()` exists anywhere in `src/`, and `orchestrator.py:128` declares
`self._plugins: list = []` which nothing populates. Any plugin written today is never called.

## 4. Integration seam and layering

**The seam is `orchestrator.py:105`** — `self._rule_engine = RuleEngine(rules_path=..., fps=...)`.
`BaseRuleEngine` (`src/pipeline/__init__.py:357-371`) already declares exactly
`evaluate(detections, memory, context) -> list[Alert]` plus `reload_rules()`, and the incumbent
`RuleEngine` already satisfies it structurally. Adding one `rule_engine: BaseRuleEngine | None = None`
constructor parameter is the entire integration. This mirrors the trainer-injection pattern ratified in
Phase 4.

The plugin seam is rejected — see [adr/ADR-P6-04-rule-engine-injection.md](adr/ADR-P6-04-rule-engine-injection.md).

```
scripts/scenarios/30_compile_scenarios.py        build-time CLI
        ↓
src/scenario_engine/                             LEAF PACKAGE
  schema.py · compile.py · ast.py · predicates/ · runtime.py
        ↓ may import
src/pipeline/__init__.py  ·  src/pipeline/event_memory.py  ·  src/utils/
```

Hard rules, enforced by an import-direction test (mirroring the `src/dataset` ⊥ `src/training` rule
stated at `src/dataset/release/gates.py:19`):

- `src/scenario_engine` **may** import `src.pipeline` contracts, `event_memory`, and `src.utils`.
- `src/scenario_engine` **must never** import `orchestrator`, `detector`, `scene_analyzer`,
  `tts_engine`, `src.dataset`, or `src.training`.
- `src/pipeline/rule_engine.py` **must never** import `src.scenario_engine`. The orchestrator — a higher
  layer — chooses which engine to inject, so the dependency arrow stays one-way.
- The compiler CLI lives in `scripts/`, so the runtime never imports the compiler.

## 5. Two contract problems the schema will hit

**`Alert` has nowhere to put the product's own output.** `src/pipeline/__init__.py:186` documents
`message` as the only user-facing field and `:193` documents `explanation` as "never shown to user".
There is no field for `next_best_action` or `caregiver_alert`. Either `Alert` grows — breaking a
contract marked LOCKED, which needs an ADR and a CHANGELOG entry — or a `ScenarioOutcome` is returned
alongside with a real sink. Resolved in
[adr/ADR-P6-09-alert-contract-extension.md](adr/ADR-P6-09-alert-contract-extension.md).

**`explanation` is a privacy hole for spatial predicates.** `src/logging/structured_logger.py:40`
defines `BBOX_REDACT_CLASSES = frozenset({"face", "person"})` and redacts those bboxes from frame logs.
But `log_alert` at `:160-174` writes `alert.explanation` **verbatim, unredacted**. Spatial predicates
naturally want `{"person_center": [0.41, 0.62], "dist_to_stove": 0.08}` for debuggability — which would
reintroduce per-alert person coordinates into `logs/events.jsonl` on a device in an elderly person's
home. Explanation dicts must route through the same redaction allowlist, with a test asserting no
`person`/`face` geometry survives.

## 6. Alert arbitration — was missing and inverted; repaired in M1

As found: the orchestrator spoke `max(alerts, key=severity)` and logged the rest, so a backlog could
not survive a frame. Underneath, `tts_engine.py` did `put_nowait` on a `PriorityQueue(maxsize=5)` and,
on `queue.Full`, **discarded the incoming message** — so a CRITICAL arriving behind five queued INFO
prompts was dropped with a `logger.warning`. `alert_queue.py:76-85` was written precisely to evict the
lowest-priority item instead, and was imported by nothing.

Repaired in M1, because a scenario layer produces *more* concurrent alerts, not fewer:

- `AlertQueue` sits between the rule engine and TTS. Alerts accumulate across frames and are drained
  in severity order.
- `PiperTTS.speak` evicts the lowest-priority **queued** message on overflow. A message is dropped
  only when nothing queued outranks it.
- `runtime.max_alerts_per_minute` is enforced (it was referenced by no code). **CRITICAL bypasses the
  cap** — a limit that can silence an emergency is a worse failure than the fatigue it prevents, and
  the alarm-fatigue evidence in `domain_research_report.md` §3 concerns routine chatter. Suppression
  affects speech only; every alert is still queued, logged, and counted.

Still outstanding for the scenario layer, and *not* solved by this: dwell, hysteresis, an event state
machine, escalation-instead-of-repetition, per-scenario daily budgets, and quiet hours. Those are
schema-level concerns (M2–M5), not queue-level ones.

## 7. CI reality

- `mypy src/pipeline` measures **clean** (`Success: no issues found in 10 source files`). The comment at
  `.github/workflows/ci.yml:37-40` claiming 17 errors is stale, and `Makefile:78` already runs
  `mypy src/` — local dev is stricter than CI. The exclusion is deleted in M1, not worked around.
- `ci.yml:5-7` previously triggered only on `main`, `develop`, and
  `phase-5-production-dataset-engineering`. A `phase-6-*` branch received **no CI on push**, making any
  "enforced in CI" claim vapour. Fixed in M0.
- `pyproject.toml` sets `fail_under = 50` on `--cov=src`, so a large new package landing without tests
  fails CI on coverage alone. Tests ship in the same commit as the code they cover.
- ruff's `S` (bandit) ruleset is enabled, so an `eval()`-based condition evaluator is blocked by S307.
  Correctly.

## 8. Performance

`../02_technical_architecture_specification/performance_budget.md:30` already allocates **5 ms** to the
rule engine, and `PipelineMetrics.rule_eval_ms` (`src/pipeline/__init__.py:256`) already measures it —
so the budget test is nearly free and follows the established `tests/performance/test_*_budget.py`
pattern. Spatial predicates are O(n²) over detections, so the compiled artifact carries an inverted
index (class → scenario ids) bounding how many scenarios are evaluated per frame.

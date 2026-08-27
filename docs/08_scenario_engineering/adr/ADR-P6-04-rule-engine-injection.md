# ADR-P6-04 — Integrate by `BaseRuleEngine` Constructor Injection, Not the Plugin Seam

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

The scenario engine must reach the running pipeline with minimal change to existing code. Two candidate
seams exist: the `AnalysisPlugin` ABC in `src/pipeline/plugin_base.py`, and the `RuleEngine` construction
site at `src/pipeline/orchestrator.py:105`.

The plugin seam was inspected and is **broken three ways**:

- `orchestrator.py:198` calls `plugin.on_frame(frame)`.
- `plugin_base.py:70-87` defines `analyze(frame, detections, memory)`.
- `docs/02_technical_architecture_specification/plugin_architecture.md:24-35` defines a third contract
  (`BasePlugin` with `on_startup`/`on_detection`/`on_alert`/`on_frame(frame, frame_id)`/`on_shutdown`).

No `register_plugin()` exists anywhere in `src/`, despite being documented in both `plugin_base.py:16-17`
and `src/plugins/__init__.py:16-17`. `orchestrator.py:128` declares `self._plugins: list = []` and nothing
populates it. Any plugin written today is never called.

## Decision

Add one constructor parameter to `ElderlyAssistantPipeline`:
`rule_engine: BaseRuleEngine | None = None`, defaulting to the incumbent `RuleEngine`. The scenario
engine ships as `ScenarioRuleEngine(BaseRuleEngine)` and is injected.

`BaseRuleEngine` (`src/pipeline/__init__.py:357-371`) already declares exactly
`evaluate(detections, memory, context) -> list[Alert]` plus `reload_rules()`, and the incumbent
`RuleEngine` already satisfies it structurally. The call sites at `orchestrator.py:189` and `:252`
therefore need no change at all. This is the same trainer-injection pattern ratified in Phase 4.

## Alternatives considered

1. **Ship as an `AnalysisPlugin`.** Rejected on four grounds. (a) Adopting it means first repairing a
   three-way contract split and building the missing registration mechanism — pure scope creep on a
   subsystem we are not otherwise touching. (b) `analyze(frame: np.ndarray, ...)` forces a numpy
   dependency on a package that will never read a pixel, and silently ignoring a required positional
   parameter is a smell that would propagate to every future plugin. (c) Plugin alerts are appended at
   `orchestrator.py:195-201` *outside* the `RuleEngine` cooldown map, so a plugin-based engine would need
   a duplicate cooldown implementation, and the two alert lists are never deduplicated before `:205`
   picks a single winner. (d) Most decisively, the supersede decision means the scenario dataset *is* the
   single source of truth for risk — so the scenario engine **is** the rule engine, not something running
   alongside it. Modelling it as a plugin guarantees two competing alert sources.
2. **Edit `rule_engine.py` in place to load the compiled artifact.** Rejected: it creates a genuine
   package-level cycle (`src.pipeline → src.scenario_engine → src.pipeline`). It happens to work today
   only because `src/pipeline/__init__.py` imports no submodules, and would detonate the first time
   anyone re-exports one.
3. **A new orchestrator.** Rejected: duplicates a working frame loop, metrics, and logging.

## Consequences

- Positive: the integration is one constructor parameter and zero changes to the frame loop.
- Positive: the dependency arrow stays one-way. `src/scenario_engine` is a leaf that may import
  `src.pipeline` contracts and `event_memory`; `src/pipeline/rule_engine.py` must never import
  `src.scenario_engine`; the orchestrator, a higher layer, chooses the implementation. Enforced by an
  import-direction test.
- Positive: both engines can run against the same recorded detection trace, which is what makes the M8
  golden alert-trace test possible.
- Constraint: the broken plugin seam is left as found. It is documented in `../architecture_review.md`
  §3 (D10) and is not this phase's scope.
- Constraint: `AlertQueue` must be wired in before M8, since a scenario layer produces more concurrent
  alerts and `orchestrator.py:205-206` currently speaks one and discards the rest.

Related: [ADR-P6-03](ADR-P6-03-structured-ast-over-string-dsl.md), [ADR-P6-09](ADR-P6-09-alert-contract-extension.md)

# 08 — Scenario & Rule Engine Engineering (Phase 6)

## Purpose

The knowledge layer that turns YOLO detections over the frozen 23-class taxonomy into elderly-care
scenarios: risk level, patient prompt, next best action, caregiver alert. This tree holds the design
record; the code lives in `src/scenario_engine/`, the authored knowledge in `configs/scenarios/`, and
the compiled artifacts and clips in `data/scenario_engine/`.

## Dependencies

Reads:
- `../02_technical_architecture_specification/rule_engine.md`
- `../02_technical_architecture_specification/interfaces.md`
- `../02_technical_architecture_specification/performance_budget.md`
- `../04_dataset_engineering/capture_annotation_runbook.md`
- `../01_executive_implementation_plan/validation_strategy.md`

Used By:
- `src/scenario_engine/`
- `configs/scenario_engine.yaml`
- `dvc.yaml` (`compile_scenarios`, `validate_scenarios`, `ingest_scenario_clips`)

Related:
- `../07_dataset_production/` (Phase 5 — the dataset this layer reasons over)

---

## Why this phase exists

The brief that opened Phase 6 assumed a greenfield build. It is not. `src/pipeline/` already ships a
rule evaluator, event memory, an alert queue, a TTS sink, a VLM analyser, and six live rules in
`configs/risk_rules.yaml`. `near(class_a, class_b, pixel_dist)` is already documented as a V1 condition
type at `../02_technical_architecture_specification/rule_engine.md:34` and was never implemented;
severity-ordered rule evaluation is documented at the same file's lines 37-43 and
`src/pipeline/rule_engine.py:95` iterates YAML order instead.

So Phase 6 has two jobs, in this order:

1. **Reconcile spec drift** — make the runtime actually do what its own configs and docs claim.
2. **Build the governed knowledge dataset** the runtime was always designed to consume.

A three-lens cold-read council reviewed the design before any code was written and found that the
runtime could not execute the schema we were about to specify. Ten defects were verified against the
code; they are enumerated with file:line evidence in
[architecture_review.md](architecture_review.md) and are the content of milestone M1.

## Documents

| Document | What it settles |
|:---|:---|
| [architecture_review.md](architecture_review.md) | What exists today, the ten verified defects, the integration seam, and the layering rules |
| [requirements_specification.md](requirements_specification.md) | What a Scenario and a Rule are; mandatory vs optional fields; what must never be assumed |
| [domain_research_report.md](domain_research_report.md) | ADL/IADL framing, CDC STEADI and WHO guidance, alarm-fatigue evidence, and the engineering requirements they imply |
| [negative_register.md](negative_register.md) | Scenarios permanently rejected as undeliverable with these 23 classes, and why |
| [scenario_taxonomy.md](scenario_taxonomy.md) | Category scheme, id grammar, the three shapes a scenario takes, and the deliberately empty categories |
| [clip_capture_protocol.md](clip_capture_protocol.md) | How scenario clips are consented, shot, sanitised, and ingested — and what differs from the photograph workflow |
| [clip_collection_plan.md](clip_collection_plan.md) | The first 100-clip batch: M7 acceptance results, the dwell-vs-clip-length arithmetic, the collection matrix, and the five-member allocation |
| [adr/](adr/) | The ten load-bearing decisions, with rejected alternatives |

## Status

Phase 6 is in progress. Milestone status is tracked in `../../CHANGELOG.md` under `[Unreleased]`.

| M | Deliverable | State |
|:--|:---|:---|
| M0 | Design record: this tree + ADR-P6-01…09; CI branch triggers | **done** |
| M1 | `EvalContext`; the runtime defects; RuleEngine tests; alert arbitration | **done** |
| M2 | Schema, frozen dataclasses, predicate registry | **done** |
| M3 | Compiler, capability map, inverted index, CSV view | **done** |
| M4 | Validator suite + alert-volume simulation gate | **done** |
| M5 | Scenario taxonomy + first ~20 scenarios | **done** (draft; clinical review pending) |
| M6 | **Validation gate — M7+ blocked until this passes** | **PASS** (8/8 gates) |
| M7 | Capture protocol, clip ingest, MP4 metadata stripper | **done** |
| M7-RW | Real-world ingest acceptance + collection design | **PASS** (10/10); collection not yet started |
| M8 | Runtime integration; `configs/risk_rules.yaml` retired | **done** (inert until clinical review — see below) |
| M9 | Integration strategy: YOLO · tracking · VLM · voice · caregiver | pending |

### How the runtime is wired (M8)

```
src/app/factory.py            composition root — the only module importing both
        │
        ├──> src/pipeline/orchestrator.py       ElderlyAssistantPipeline(rule_engine=…)
        └──> src/scenario_engine/runtime.py     ScenarioRuleEngine
```

`build_pipeline(room="kitchen")` is the entry point. The orchestrator does **not**
construct the engine: that would make `src.pipeline` import `src.scenario_engine`,
which already imports `src.pipeline`. `configs/risk_rules.yaml` is retired; the frozen copy
lives at `tests/fixtures/legacy_risk_rules.yaml` so the migration comparison stays runnable.

**The engine refuses to start while every scenario is `draft`.** That is the current state and
it is correct — a safety engine that quietly loads zero rules is indistinguishable from one
working perfectly and seeing nothing.

### Two things engineering cannot clear on its own

- **Clinical review.** Every scenario is `status: draft`. Promotion to `active` requires
  `reviewed_by`/`reviewed_on` from a qualified human, not engineering sign-off. The seven
  `safety-class-uncovered` warnings in the validation report persist for that reason and are correct.
- **Field collection.** ffmpeg 9.0 is now installed and the ingest path is proven end-to-end against a
  real MP4 carrying real GPS (`data/qa_reports/m7_acceptance_report.json`, 10/10). What remains is
  human work: consent, filming, and review. **Zero scenario clips have been collected** —
  `scripts/scenarios/33_clip_dataset_report.py` reports FAIL against the 100-clip target and will keep
  doing so until real clips are accepted. See [clip_collection_plan.md](clip_collection_plan.md).

## Non-negotiables

- **This is not a medical device.** It does not detect falls, medical emergencies, gas leaks, or
  whether medication was taken. See [negative_register.md](negative_register.md).
- **Nothing asserts state it cannot observe.** The taxonomy is static objects plus `person`/`face`.
  There is no pose, no on/off state, no tracking.
- **Escalation terminates at a human.** No automated emergency dispatch, ever.

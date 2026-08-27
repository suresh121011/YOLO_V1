# ADR-P6-05 — The Compiled Artifact Carries a Class Capability Map

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

Phase 5 established `taxonomy_fingerprint(nc, names)` (`src/dataset/completeness.py:67-85`), a sha256 over
the canonical `{nc, [[id, name], ...]}` structure that changes **iff `nc` or any (id, name) pair
changes**. It is cross-checked in `coverage.py`, `ledger.py`, `candidates.py`, and recorded in every
release manifest. The obvious move was to carry it in the scenario artifact and rely on it.

It does not work for this purpose, and the counterexample is already scheduled.

The documented R24 decision (`docs/04_dataset_engineering/capture_annotation_runbook.md` §8, risk
register R24, gate `check_wet_floor_pilot_decision` in `src/dataset/release/gates.py:187-208`) demotes
`wet_floor` to scene-level if dual-annotator IAA falls below 0.60 — with **class ID 20 explicitly
reserved, no taxonomy renumber**. So `nc` stays 23, `names[20]` stays `wet_floor`, and the fingerprint is
byte-identical. Every scenario requiring `wet_floor` keeps compiling green while the detector no longer
emits a `wet_floor` box.

The same bug is **already live**: `configs/feature_flags.yaml:35` sets `passport: false` with the comment
"Disabled by default — privacy". Any scenario requiring `passport` would be dead at runtime, and no
fingerprint would ever notice.

A fingerprint answers "is the taxonomy the same?". Scenarios need to know "is this class actually
detectable and enabled?" — a different question.

## Decision

`scenarios.compiled.json` carries a **class capability map** alongside the taxonomy fingerprint:

```json
"classes": {
  "wet_floor": {"id": 20, "detection_level": "bbox",  "enabled": true},
  "passport":  {"id":  8, "detection_level": "bbox",  "enabled": false}
}
```

derived at compile time from `configs/data.yaml` + `configs/feature_flags.yaml` `classes:` + the R24
decision artifact. Validator V3 hard-fails any scenario whose `required_objects` contains a class that is
not `bbox`-detectable or not enabled.

Class names are resolved exclusively through `load_data_config()` and
`get_class_names_from_data_yaml()` (`src/utils/config_helpers.py:90-111, 150-167`). The repository
already contains eight partial or complete duplicates of the 23-class list; this adds no ninth.

Supporting process change: the R24 pilot decision is persisted to a small git-tracked artifact
(`data/qa_reports/wet_floor_decision.json`) rather than being derived on the fly by globbing `iaa_*.json`.
The scenario compiler reads that artifact and must not re-implement `read_wet_floor_pilot_decision`.

The **category vocabulary is not promoted into `configs/data.yaml`.** Its five semantic groupings are
comments at `:47-52`, and although adding real keys would be inert to Ultralytics, `configs/data.yaml` is
a declared dep of seven DVC stages — the promotion would force a full re-run of a 24k-image pipeline to
change a comment into a key. The scenario category vocabulary lives in `configs/scenario_engine.yaml`
and reuses the room vocabulary from `configs/capture_config.yaml:32-33`.

## Alternatives considered

1. **Rely on the taxonomy fingerprint alone.** Rejected: provably blind to the exact demotion event that
   is already scheduled, and to the `passport` flag that is already set.
2. **A per-row `taxonomy_fingerprint`.** Rejected: 300 identical strings and 300 places to go stale. The
   fingerprint belongs once, in the artifact header, beside `schema_version`, `git_commit`, and
   `created_at` — the existing `release_manifest.json` header shape.
3. **Runtime detection of a missing class.** Rejected: a scenario that never fires is indistinguishable
   from a scenario whose condition is never met. The failure must be caught at compile time, loudly.

## Consequences

- Positive: a class demotion or a privacy toggle becomes a **build failure** naming the affected
  scenarios, rather than silent permanent non-firing.
- Positive: the artifact becomes self-describing about what the deployed detector can actually see.
- Constraint: the compiler now depends on `configs/feature_flags.yaml` and the R24 decision artifact, both
  of which must be declared DVC deps of `compile_scenarios`.
- Constraint: the capability map is only as honest as its inputs, and `configs/feature_flags.yaml` is
  currently not enforced at runtime at all (see `../architecture_review.md` §3, D1/D2). M1 fixes that
  first; until it does, the map describes intent rather than behaviour.

Related: [ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md), [ADR-P6-08](ADR-P6-08-rejected-scenario-negative-register.md)

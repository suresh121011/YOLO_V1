# ADR-P6-12 — Weights Must Carry the Declared Taxonomy, Verified at Load

**Status:** Accepted (Phase 6, M9, 2026-08)
**Deciders:** Phase-6 engineering

## Context

The whole knowledge layer was built with **no trained model in the repository**. Everything above the
detector addresses classes by name:

- the scenario engine's inverted index (`class name -> scenario ids`),
- per-scenario `min_confidence` floors, applied per required class,
- the class capability map ([ADR-P6-05](ADR-P6-05-class-capability-map.md)),
- `disabled_classes`, which is what makes `passport: false  # privacy` a guarantee rather than a comment.

`YOLODetector` took whatever `model.names` the weights happened to carry and used it directly. Weights
trained on a different class list — a different ordering, a renamed class, a 20-class subset — would
load without complaint, run at full speed, and produce `Detection` objects whose `class_name` addresses
the wrong concept everywhere above. Nothing in the system would notice; the alert-volume gate, the
determinism validator and the performance budget would all stay green.

Ids matter as much as names. The documented R24 decision demotes `wet_floor` to scene level **with class
id 20 reserved and no renumber**, so ids are load-bearing across `configs/data.yaml`, the capability map
and the taxonomy fingerprint. A model that shuffles ids while keeping every name is precisely the case a
name-set comparison would wave through.

## Decision

**`YOLODetector` verifies the loaded weights against an expected `id -> name` mapping at construction and
raises `TaxonomyMismatchError` on any difference** — missing ids, extra ids, or an id whose name differs.
The error names the specific differences rather than reporting a boolean.

The expectation is supplied by the caller. `SystemConfig` gains `class_names`, read from
`configs/data.yaml`, and `src/app/factory.build_pipeline` passes it. The detector therefore still reads
no YAML of its own, which is `SystemConfig`'s documented contract.

`expected_classes` is **optional**, so tests and experiments can load arbitrary weights — but omitting it
logs a warning containing `UNVERIFIED`. The guard is permissive by default and never silent.

`scripts/qa/model_landing_check.py` (check L3) makes this the first thing run when weights arrive, with
`tests/unit/test_model_landing_check.py` proving the check can fail.

## Alternatives considered

1. **Compare only the taxonomy fingerprint.** Rejected: the fingerprint is computed from
   `configs/data.yaml`, not from the weights, so it says nothing about what the model was trained on.
   That is the same blindness ADR-P6-05 was written about, one layer down.
2. **Map class names at inference (a translation table).** Rejected: a rename map is a second taxonomy.
   The repo already carries eight partial duplicates of the 23-class list, and ADR-P6-05 exists because
   of that. A mismatch is a training or export defect and belongs fixed at its source.
3. **Warn instead of raising.** Rejected: a warning at startup on a headless device in someone's home is
   not read by anyone. The consequence of continuing is a safety pipeline confidently reasoning about the
   wrong classes, which is worse than a device that refuses to start and says why.
4. **Check on the first frame rather than at construction.** Rejected: it moves a configuration error
   into the hot path, and `detect()` has no good way to fail loudly without either dropping frames
   silently or crashing the camera loop.

## Consequences

- Positive: the single most likely model-landing failure is caught before the first frame, with the
  specific difference named.
- Positive: `SystemConfig.class_names` gives one place for the runtime taxonomy, rather than a ninth
  copy of the class list.
- Constraint: a deliberately reduced model (say, a 5-class kitchen-only build) cannot be run through
  `build_pipeline` without editing `configs/data.yaml` — which is correct, since the scenarios,
  thresholds and capability map are all derived from that file.
- Constraint: `expected_classes=None` remains a legitimate path for experiments, so the guard depends on
  the composition root being used. That is why L3 exists as an explicit check rather than relying on the
  constructor alone.

Related: [ADR-P6-05](ADR-P6-05-class-capability-map.md) · [../integration_strategy.md](../integration_strategy.md)

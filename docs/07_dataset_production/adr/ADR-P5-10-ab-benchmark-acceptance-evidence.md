# ADR-P5-10 — Full-Scale A/B = v1.0 Acceptance Evidence (One Run per Arm, Fixed Config); Tuning → Phase 6

**Status:** Accepted (decision); A/B **execution pending** at M10/M11 (real GPU run)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17 (A/B-evidence finding accepted)

## Context

Dataset v1.0 must show that the missing-annotation mitigation actually helps, not
just assert it. But Phase 5 is a dataset-engineering phase — model tuning is
Phase 6, and compute is finite.

## Decision

The v1.0 acceptance evidence is exactly **two** training runs — mitigation off vs
on — at a fixed `yolo11n_config.yaml` (100 epochs, imgsz 640, fixed seed),
evaluated on the frozen test split AND the locked `eval-indian-home-v0` set via
`scripts/training/evaluate_model.py`. RG10 requires both an `eval_report.json`
and an `ab_benchmark/` directory before v1.0.0 can be cut. This is acceptance
**evidence**, not a statistical claim (one run per arm) — the limitation is
stated in the report banner. Hyperparameter tuning and `export_model.py` are
re-deferred to Phase 6.

## Alternatives considered

1. **Multi-seed statistical A/B.** Rejected for Phase 5: compute-prohibitive; the
   goal is directional evidence that mitigation helps, not a p-value.
2. **Skip A/B, assert mitigation value from Phase-4 unit tests.** Rejected:
   unit-level bit-identity proves correctness, not end-to-end benefit.

## Consequences

- Positive: v1.0 ships with real, reproducible end-to-end evidence at bounded
  cost; the honest one-run-per-arm caveat is documented.
- Constraint: the actual A/B run is a multi-hour GPU operation (M10) that must
  complete before RG10 passes — this ADR records the decision; execution is a
  pending operational milestone.

- **Amended 2026-07-29 (pre-v0.7.0 review) — "a fixed `yolo11n_config.yaml`" was
  not realisable as written, and taken literally would have produced a
  confounded experiment.** Preflight gate **G8**
  (`src/training/preflight.py:197`, `mixing_augmentation_policy: forbid`)
  requires `mosaic/mixup/copy_paste` to be `0` whenever mitigation is enabled,
  because composited samples expose only the primary image and make per-image
  masks unsound. The committed config sets `mosaic: 0.8, mixup: 0.1,
  copy_paste: 0.1`. So the mitigation-**on** arm cannot use that config, and the
  obvious repair — zeroing the mixing augmentations only for the on-arm — makes
  the two arms differ in **two** variables at once. Any measured delta would then
  be unattributable between "mitigation helped" and "mixing augmentation hurt".

  **Decision: both arms train with `mosaic: 0.0, mixup: 0.0, copy_paste: 0.0`,
  from one shared config; `missing_annotation_mitigation.enabled` is the only
  difference between them.** The off-arm therefore is *not* the stock Ultralytics
  recipe, and the A/B report banner must say so — it measures the value of
  mitigation, holding augmentation fixed, which is the question RG10 asks.

  Rejected: a third arm (stock recipe, mitigation off) to also measure the
  augmentation cost. It answers a Phase-6 tuning question at v1.0's compute
  budget, and ADR-P5-10's own scope is directional evidence, not a full factorial.

  This becomes far more consequential after
  [ADR-P5-15](ADR-P5-15-per-slug-trust-resolution-and-targeting.md): with
  per-slug trust the masked-cell fraction rises from 0.186 to **0.846**, so the
  mitigation-on arm masks most of the supervision signal. That is the honest
  baseline, and this A/B is the instrument that measures what it costs.

Related: [ADR-P5-07](ADR-P5-07-releases-as-code.md),
[ADR-P5-15](ADR-P5-15-per-slug-trust-resolution-and-targeting.md)

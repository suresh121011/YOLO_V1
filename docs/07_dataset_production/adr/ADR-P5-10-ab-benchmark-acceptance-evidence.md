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

Related: [ADR-P5-07](ADR-P5-07-releases-as-code.md)

# ADR-P5-06 — L4 Coverage Derives Exclusively from Pinned Candidates; No Inference at Report Time

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

The L4 coverage report quantifies residual missing-annotation risk per class. If
it re-ran a model, its output would drift with GPU nondeterminism and couple
reporting to the annotation stack.

## Decision

The coverage stage performs **zero inference** — it is pure arithmetic over the
pinned candidates artifact + processed labels + the ledger. "Unknown" objects =
candidates in untrusted+unverified cells → `potential_missing`, discounted by a
per-class estimator precision **calibrated from verified batches**
(verified_absent cells expose estimator false positives — free calibration
data), falling back to `coverage.estimation_conf` when uncalibrated. L5
(`17_dataset_quality_report`) is a pure aggregation over completeness + coverage
+ merged manifest + ledger — it deliberately recomputes nothing.

## Alternatives considered

1. **Re-run the detector at report time for fresh estimates.** Rejected:
   nondeterministic, slow, and couples the report to the model; the artifact it
   would report on already exists and is pinned.
2. **Report raw candidate counts without calibration.** Rejected: uncalibrated
   counts overstate risk; the ledger's own verified cells are free calibration.

## Consequences

- Positive: deterministic, fast (≤5 min at 30k), and cannot drift from the
  artifacts it reports on; honest ("deferred until calibration data exists")
  when the ledger is empty.
- Constraint: coverage quality improves only as verification produces calibration
  cells — by design.

Related: [ADR-P5-02](ADR-P5-02-yolo-world-primary-backend.md),
[ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md)

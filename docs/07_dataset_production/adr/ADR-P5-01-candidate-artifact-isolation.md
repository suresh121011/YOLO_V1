# ADR-P5-01 — Candidate-Artifact Isolation: Auto-Generated Labels Never Write `labels/`

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Phase 5 adds model-assisted (L2) and cross-dataset (L3) annotation to fill the
missing-annotation gap. Auto-generated boxes are advisory — an open-vocabulary
model hallucinates, and unverified boxes written straight into `labels/` would
silently corrupt the trusted training labels the whole pipeline (masking,
completeness, QA) depends on.

## Decision

Auto-annotation output lands ONLY in a separate candidate artifact
(`data/annotation/candidates/<backend>/candidates.json`) — never in any
`labels/` directory. Candidates cross a mandatory human CVAT round-trip
([ADR-P5-03](ADR-P5-03-cvat-round-trip.md)), are recorded in the verification
ledger ([ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md)), and only
then are overlaid onto merged labels by a deterministic stage
([ADR-P5-05](ADR-P5-05-labels-overlay-no-mutation.md)).

## Alternatives considered

1. **Write pseudo-labels directly, flag them as low-confidence.** Rejected: any
   path that mixes model output into `labels/` risks it being trained on or
   masked incorrectly; the "flag" is one bug away from being ignored.
2. **Auto-accept high-confidence detections, human-review only the rest.**
   Rejected for v1: precision on the priority classes is unknown until the
   ledger calibrates it; auto-accepting before that is exactly the R30
   hallucination-flood risk.

## Consequences

- Positive: trusted labels are structurally protected; candidates are fully
  regenerable (a DVC out) and never load-bearing for training until verified.
- Positive: the empty-ledger passthrough is byte-identical to the pre-Phase-5
  build (golden regression), so M1–M6 tooling lands with zero behavior change.
- Constraint: every consumer of candidates (coverage L4, batches) treats them
  as advisory and pins the `candidates_sha256` it consumed.

Related: [ADR-P5-03](ADR-P5-03-cvat-round-trip.md),
[ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md),
[ADR-P5-05](ADR-P5-05-labels-overlay-no-mutation.md)

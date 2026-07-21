# ADR-P5-05 — Labels-Only Overlay (`data/merged_verified`), No Mutation, No Image Duplication

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Verified delta labels must reach the training split without mutating the
immutable `data/merged` output or duplicating tens of thousands of images at
full scale.

## Decision

`15_apply_verified_labels` writes `data/merged_verified/labels` = base labels ∪
verified deltas. `data/merged` stays immutable. The split stage reads labels via
`split.source_labels_dir` while taking images from `data/merged` — no image
duplication. An empty ledger yields a byte-identical labels passthrough
(golden-tested).

## Alternatives considered

1. **Mutate `data/merged/labels` in place.** Rejected: destroys the immutable
   merge output and its DVC lineage; not reversible.
2. **Materialize a full `data/merged_verified` with copied images.** Rejected:
   duplicates ~30k images on disk for a labels-only change.

## Consequences

- Positive: overlay is deterministic and reversible; empty-ledger byte-identity
  keeps backward compatibility; no storage blow-up.
- Constraint: the split stage must be pointed at the overlay via one config key.

Related: [ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md),
[ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md)

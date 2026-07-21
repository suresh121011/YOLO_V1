# ADR-P5-12 — Vectorized-Hamming Dedup with Decision-Equivalence Guarantee, over a BK-Tree

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Perceptual dedup is O(n²) worst case (each kept image compared against the
growing kept-set). At the ~30k full-mode scale the naive Python scan risks the
≤10-minute budget, but the exact-bucket aHash semantics must not change.

## Decision

Keep the exact-bucket aHash **semantics**; add a numpy-vectorized Hamming path
inside `DedupIndex` (packed `uint64` + XOR/popcount) activated above 2,000 kept
images. A property test asserts **decision-for-decision equivalence** with the
naive path over 500 random hash sets plus adversarial cases (the final audit
caught, and this test now pins, a match-attribution ordering bug where the
vectorized path could attribute a duplicate to a different kept image than the
naive path — same keep/drop decision, different `duplicate_of`). A perf test
proves 30k images within budget.

## Alternatives considered

1. **BK-tree metric index.** Rejected: more code and no better worst case at
   n≈30k with a small Hamming threshold; the vectorized scan is simpler and
   provably equivalent.
2. **Approximate/LSH dedup.** Rejected: changes semantics (false drops/keeps);
   the requirement is exact equivalence with the established behavior.

## Consequences

- Positive: meets the 30k budget with provably identical decisions; the naive
  path stays as the reference the property test checks against.
- Constraint: the equivalence guarantee must be re-verified whenever the
  vectorized path changes (the property + adversarial tests enforce this).

Related: [ADR-P5-08](ADR-P5-08-cross-dataset-label-salvage.md)

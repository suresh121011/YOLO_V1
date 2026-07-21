# ADR-P5-08 — L3 = Exact-SHA256 Label Salvage + Opportunistic Near-Dup Candidates; No Image Registration

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Perceptual dedup drops one twin of a duplicate image pair — and, before Phase 5,
discarded the dropped twin's labels too. When the twins come from different
sources with different trusted classes (real overlap is Roboflow↔COCO
derivatives), that silently throws away exhaustively-labeled boxes the surviving
image could carry.

## Decision

L3 label salvage runs at merge, in a new module `src/dataset/cross_dataset_salvage.py`
(deliberately NOT under `src/dataset/annotation/` — `merge.py` must not depend on
the annotation package). Exact-SHA256 duplicates: transplant the dropped source's
trusted-class boxes onto the survivor, provenance-tagged, suppressing any
transplant whose class overlaps an existing box at IoU ≥ 0.9 (both sources
labeling the same object must not double-box). Near-duplicate (aHash/flip)
matches have unknown geometry → unsafe to transfer → emitted as `cross_dataset`
**candidates** through the normal human loop. No image-registration machinery.

## Alternatives considered

1. **Transfer labels for near-duplicates too.** Rejected: aHash near-dups can
   differ in crop/scale; transferred boxes would be geometrically wrong.
2. **Full image registration to align near-dups.** Rejected: heavy machinery for
   marginal gain; the human loop already handles the uncertain cases safely.
3. **Keep discarding the dropped twin's labels.** Rejected: this was the actual
   silent data-loss bug the final audit caught (`merge.py` dropped all duplicate
   labels unconditionally, contradicting the "L1–L5 mechanized" claim).

## Consequences

- Positive: recovers exhaustively-labeled boxes for byte-identical duplicates
  with a safe IoU guard; uncertain cases stay in the verified loop.
- Constraint: gains are limited to true cross-source duplicates — deliberately
  thin, honest about what exact-hash salvage can and cannot recover.

Related: [ADR-P5-12](ADR-P5-12-vectorized-dedup.md),
[ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md)

# ADR-P4-01 — Missing-Annotation Mitigation at the Loss Level (Masked BCE)

**Status:** Accepted (Phase 4, 2026-07)
**Deciders:** Phase-4 engineering; pre-approved direction in the Pre-Phase-4 audit §8

## Context

Public datasets label only a subset of the 23-class taxonomy (COCO labels
`person` but not `face`; WIDER FACE labels only `face`). YOLO's classification
BCE treats every unlabeled class as a negative at every anchor, so
present-but-unlabeled objects receive systematic "push to background"
supervision. On the smoke dataset the mean image trusts only 6.25/23 classes —
~73 % of (image, class) supervision cells were false negatives.

## Decision

Zero the classification BCE for (image, class) cells whose source does not
annotate that class exhaustively ("trusted classes"), at the loss level, driven
by the per-image completeness artifact (`data/processed/completeness.json`).
Box and DFL losses are untouched (they only apply to matched foreground
anchors of labeled ground truth).

## Alternatives considered

1. **Sampling-level filtering** — drop images whose sources are incomplete for
   the classes they contain. Rejected: with per-source trust lists this
   discards *most* public data (every COCO image is incomplete for 13
   classes); it also cannot express "this image supervises person but not
   face" — the actual situation.
2. **Label imputation (pseudo-labeling)** — run a teacher model to fill in
   missing boxes. Rejected for Phase 4: introduces a second model, confidence
   thresholds, and error feedback loops; far weaker reproducibility. May
   complement masking in a later phase once a trustworthy teacher exists.
3. **Per-class loss weighting** — down-weight untrusted classes globally.
   Rejected: trust is per-*source*, not global; weighting still leaks false
   negative supervision (just less of it) and adds a tuning knob.

## Consequences

- Positive: exact removal of false supervision; provably identical to stock
  loss when every class is trusted (bit-identity unit test); zero data loss.
- Negative: untrusted classes receive *no* negative supervision from those
  images — background discrimination for a class is learned only from sources
  that trust it (incl. `negatives`, which trusts all 23). This is the accepted
  trade-off and the reason verified-negative images stay valuable.
- Constraint: mixing augmentations conflict with per-image masks →
  [ADR-P4-04](ADR-P4-04-strict-mixing-aug-policy.md).

Related: [ADR-P4-02](ADR-P4-02-trainer-injection.md),
[ADR-P4-03](ADR-P4-03-completeness-artifact.md),
[masked_loss_architecture.md](../masked_loss_architecture.md)

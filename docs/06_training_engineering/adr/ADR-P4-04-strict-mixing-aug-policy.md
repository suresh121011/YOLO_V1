# ADR-P4-04 — Forbid Mixing Augmentations Under Mitigation (Gate G8)

**Status:** Accepted (Phase 4, 2026-07) — explicit product decision

## Context

Mosaic, MixUp, and Copy-Paste composite several source images into one
training sample, but Ultralytics exposes only the **primary** image's path in
`batch["im_file"]`. A per-image completeness mask therefore cannot be correct
for composited samples: (i) a trusted positive pasted from another image can
be masked out (lost signal, mild), and (ii) an unlabeled-but-present object
from a pasted tile under an all-ones primary (e.g. negatives) still receives
false background push — the exact error this phase eliminates.

## Decision

Preflight gate G8 **fails** training when mitigation is enabled and any of
`mosaic`/`mixup`/`copy_paste` is > 0 (`mixing_augmentation_policy: forbid`,
the default — confirmed with the product owner on 2026-07-16). `warn` and
`ignore` remain as documented escape hatches for experiments. An absent
`augmentation:` section resolves to Ultralytics' own defaults (mosaic = 1.0!)
and is treated as active. Benchmarks zero these augmentations in **both** arms
so comparisons stay apples-to-apples.

## Alternatives considered

1. **Warn and mask by the primary image** — rejected as the default: silently
   reintroduces a fraction of the false supervision the feature exists to
   remove; kept as the `warn` escape hatch with the limitation documented.
2. **Intersect the masks of all constituent images** — strictly correct, but
   the constituent list is not exposed at the loss boundary; obtaining it
   means deep dataset surgery (out of "no Ultralytics modification" scope).
3. **Mask-aware custom mosaic implementation** — Phase-5+ candidate if the
   accuracy cost of disabling mosaic proves material at full scale.

## Consequences

- Positive: mask semantics are always exact; the correctness proof
  (bit-identity, zero gradients) covers every training sample.
- Negative: mitigation runs forgo mosaic/mixup regularization. At full scale
  this may cost some augmentation benefit — measured explicitly by the
  benchmark and revisited in Phase 5 (risk R28); `close_mosaic` already
  disables mosaic for final epochs in stock training anyway.

Related: gate G8 in src/training/preflight.py, risk R28.

# ADR-P5-15 — Per-Slug Trust Resolves From Ingest Manifests, and Targeting Must Use It Too

**Status:** Accepted 2026-07-29 — **supersedes [ADR-P5-14](ADR-P5-14-per-slug-completeness-policy.md)**
**Deciders:** Phase-5 engineering (design review before v0.7.0; user decision recorded)

## Context

ADR-P5-14 (accepted the same day, before the `dataset-v0.6.0` ceremony) recorded
that `merge_datasets` publishes one source-level `label_completeness` entry for
`local_captures` holding the **union** of all 23 slugs' trusted classes, so
16,926 of 24,352 images — **69.5% of the release** — inherit ~21.7 falsely-trusted
classes each. That finding stands. Two of its conclusions do not.

The pre-v0.7.0 review measured what P5-14 assumed. Both assumptions were wrong,
in opposite directions: the fix is **cheaper** than P5-14 priced it, and its
**blast radius is wider** than P5-14 scoped it.

### Correction 1 — slug is already recorded provenance

P5-14 chose "record the slug in `image_provenance`" (requiring a
`merge_datasets` rebuild) and rejected "parse the slug from the filename" as
implicit coupling on a naming convention. Both options missed a third: ingest
already writes the mapping.

`data/raw/local_captures/<slug>/manifest.json` carries `image_hashes` keyed by
`<slug>__<original>`, with a sha1 per image. The merged filename is exactly
`local_captures_` + that key. Slug is therefore recoverable from **recorded
provenance**, and the recovery is *verifiable* against the stored hash — strictly
stronger than an `image_provenance` string would have been.

Measured against the `dataset-v0.6.0` build:

| | |
|---|---|
| Slug manifest keys | **26,339** |
| Cross-slug key collisions | **0** |
| Merged `local_captures` images resolved | **16,926 / 16,926 (100.00%)** |
| Distinct slugs represented | **23 / 23** |

### Correction 2 — the defect is not confined to `completeness.json`

P5-14 framed this as a completeness-artifact defect whose blast radius "bites
only at train time". That is false.

`src/dataset/annotation/targeting.py` selects untrusted cells for
auto-annotation and reads trust from **`merged_manifest.label_completeness`** —
the same source-level union — not from `completeness.json`. With `local_captures`
declaring all 23 classes trusted, every local-capture cell looked supervised and
was skipped.

The evidence is the candidate inventory itself: **all 6,773 candidates come from
public sources**, confined to `charger` (1,719), `cupboard` (3,976), `face`
(915), `medicine_bottle` (163). **The auto-annotation pipeline has never run
against 69.5% of the dataset.**

This matters for sequencing, not just correctness: `dataset-v0.7.0` requires
`min_verified_cells: 3000`, and verifying cells drawn from a candidate set
selected under a known-wrong trust model would spend the scarcest resource in the
project — human verification time — on the wrong cells.

### What the corrected artifact looks like

| Metric | `dataset-v0.6.0` as shipped | Corrected | Δ |
|---|---|---|---|
| `mean_trusted_classes_per_image` | 18.724 | **3.542** | −81% |
| `masked_cell_fraction` | 0.1859 | **0.8460** | 4.55× |
| Trusted cells | 455,966 | **86,250** | −369,716 |

Per source after correction: `local_captures` **1.16**, `coco` 10.00,
`openimages` 3.00, `negatives` 23 (`verified_absence_all`), `wider_face` 1.00.

## Decision

1. **Resolve slug from the ingest manifests**, not from `image_provenance` and
   not from the filename convention. `generate_completeness` and `auto_annotate`
   take `data/raw/local_captures` as an explicit DVC dep. `merge_datasets` is
   **not** re-run, so `data/merged` and `data/processed/*` stay byte-stable and
   the Phase-F2/F3 clean-container evidence carries over.
2. **Fix both consumers of trust** — `src/dataset/completeness_policies.py`
   (policy keys `local_captures/<slug>`) **and**
   `src/dataset/annotation/targeting.py`. Fixing one is not a partial fix; it is
   a fix that leaves the more damaging half in place.
3. **Regenerate candidates** (`auto_annotate`, GPU) after (2), before any human
   verification begins.
4. **Resolution must fail loudly.** An image that does not resolve to a slug is
   an error, never a fallback to the source-level set. A silent default here
   would reproduce the original defect while appearing fixed.
5. **No release threshold moves.** `configs/release.yaml` is untouched;
   `dataset-v0.7.0` keeps `min_verified_cells: 3000` and its coverage floors.

## Alternatives considered

1. **P5-14's approach — record slug in `image_provenance`.** Rejected: requires
   a `merge_datasets` rebuild (48,706 files), which changes `data/merged`'s hash
   and cascades into `split_train_val_test` and the **GPU** `auto_annotate`
   stage, invalidating reproducibility evidence — to obtain a mapping that is
   already recorded and hash-verifiable.
2. **Parse the slug from the merged filename.** Still rejected, for P5-14's
   reason: it would make masking semantics depend on a naming convention. The
   adopted approach uses the filename only to *index into recorded data*, and can
   verify the result against the stored sha1.
3. **Fix `completeness_policies.py` only, defer `targeting.py`.** Rejected —
   this was effectively P5-14's scope. It would leave auto-annotation blind to
   69.5% of the dataset and send the verification campaign at the wrong cells.
4. **Re-baseline `dataset-v0.7.0`'s thresholds** once the candidate pool grows.
   Rejected: moving a bar a release is currently failing is exactly what
   ADR-P5-13 refused to do for RG9. If the thresholds later prove wrong, that is
   its own ADR, not an edit.

## Consequences

- Positive: the correctness fix costs no dataset rebuild and no retraining, and
  the `dataset-v0.6.0` reproducibility evidence remains valid.
- Positive: auto-annotation begins covering the 69.5% of the dataset it has
  never examined, which is the only route to a candidate pool that makes
  `min_verified_cells: 3000` a meaningful measure.
- **Constraint:** after the fix, **84.6% of (image, class) cells are masked.**
  Training with `missing_annotation_mitigation` enabled will have ~81% less
  trusted supervision than the `dataset-v0.6.0` artifact implied. This is the
  honest baseline, and it makes each verified ledger cell materially more
  valuable — verification moves from bookkeeping to the primary lever on dataset
  value.
- **Constraint:** the candidate pool will grow substantially, changing
  `coverage_report` denominators and the verification economics. Snapshot the
  pre-change coverage/quality reports as release evidence (the Phase-E
  `data/qa_reports/_snapshots/` pattern) before regenerating.
- Negative, accepted: `dataset-v0.6.0` remains published with the over-claiming
  artifact. Its changelog, ADR-P5-14 and the pinned test all say so, and its tag
  message carries the instruction not to train on it with mitigation enabled.

Related: [ADR-P5-14](ADR-P5-14-per-slug-completeness-policy.md) (superseded),
[ADR-P5-13](ADR-P5-13-v060-local-capture-release-track.md),
[ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md),
[ADR-P5-07](ADR-P5-07-releases-as-code.md)

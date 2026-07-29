# ADR-P5-14 — Completeness Policy Granularity for `local_captures`: Defer Per-Slug Keys to `dataset-v0.7.0`

**Status:** Accepted 2026-07-29
**Deciders:** Phase-5 engineering (design review before the `dataset-v0.6.0` ceremony; user decision recorded)

## Context

`local_captures` is ingested from 23 Drive ZIPs by
`scripts/dataset/20_ingest_local_zips.py`, one sub-directory per slug. It
contributes **16,926 of the 24,352 images** in the `dataset-v0.6.0` build —
**69.5% of the release**.

Completeness policies (ADR-P5-03/-04) declare which classes a source annotates
**exhaustively**. For a trusted class, an absent label means "verified absent"
and is supervised as background; for an untrusted class, the cell is masked out
of the loss. Over-declaring trust therefore produces **false-negative
supervision** — the model is taught that a real object is background.

The per-slug ground truth already exists on disk. Ingest writes it:

```
data/raw/local_captures/bed/manifest.json
  "trusted_classes": ["bed", "chair", "person", "sink"]
  source_classes.json: {"0": "person", "16": "chair", "17": "bed", "19": "sink"}
```

But `TrustedListPolicy.resolve_policies`
(`src/dataset/completeness_policies.py:232`) returns `{ctx.source: ids}` — **one
policy key per source, not per slug** — and `merge_datasets` records a single
`label_completeness["local_captures"]` entry holding the **union** of all 23
slugs' trusted classes. `completeness.json`'s `stats.by_policy` confirms one
`local_captures` policy covering all 16,926 images.

### The over-claim, measured

Every slug's own `trusted_classes` compared against the merged union:

| | |
|---|---|
| Images on the union policy | **16,926 / 24,352 = 69.5%** |
| Union size | **23** classes |
| Slugs whose own trusted set is 1 class | **20 of 23** (`bed` and `medicine_bottle` = 4, `cupboard` = 3) |
| Mean falsely-trusted classes per image | **21.7 of 23** |
| `mean_trusted_classes_per_image` as shipped | **18.724** — per-slug evidence supports ≈ **3.6** |
| `masked_cell_fraction` as shipped | **0.1859** — per-slug evidence supports ≈ **0.84** |

The shipped artifact claims roughly **five times** more trustworthy supervision
than the slug manifests justify.

The harm is not hypothetical. Filenames in the `bed` slug read
`local_captures_bed__000000034480_jpg.rf.*` — COCO image IDs re-exported through
Roboflow — so those frames contain unlabelled people, chairs and books that the
union policy marks verified-absent.

### Correcting the record

The backlog item that raised this (task #22, 2026-07-27) stated that the slugs
are single-class and that the union "affects all 23 classes equally". Both are
wrong: `bed`, `medicine_bottle` and `cupboard` are multi-class, and the
over-claim is per-slug, not uniform. The defect is **larger** than recorded, and
the fix has a data source the original note did not know existed.

## Decision

**Defer the per-slug policy fix to `dataset-v0.7.0`. Ship `dataset-v0.6.0` with
the limitation disclosed in prose *and* pinned in code.**

Concretely, for v0.6.0:

1. No change to `completeness_policies.py`, `merge_datasets`, or any config.
2. `data/DATASET_CHANGELOG.md` states the over-claim with these numbers under
   Known Limitations.
3. `tests/unit/test_completeness_policy_granularity.py` pins the measured state,
   so it cannot silently worsen and cannot be silently fixed without revisiting
   this ADR.

And for v0.7.0, the fix is: policy keys of the form `local_captures/<slug>`,
sourced from the per-slug `manifest.json` files, plus a release gate that fails
when a source's declared trust exceeds what its constituent manifests support.

### Why this is not a release blocker

- It changes **no image, no label, and no split**. Only `completeness.json`.
- It is consumed **at train time**, under `missing_annotation_mitigation`.
  `dataset-v0.6.0` carries no training evidence by design (ADR-P5-13) — RG10 is
  out of scope, and `src/training/preflight.py` gates training separately.
- **No v0.6.0 gate covers it**, and none pretends to: RG2 checks completeness
  *self-consistency and input-hash freshness* (`gates.py:248`), never the
  semantic correctness of a trusted set.

## Alternatives considered

1. **Fix before tagging v0.6.0.** Rejected on sequencing, not principle. The
   fix requires re-running `merge_datasets` (48,706 files) to record slug
   provenance, which changes `data/merged`'s hash. That would invalidate the
   Phase F2/F3 clean-container evidence obtained the same day — the gate that
   had been open since 2026-07-14 — in order to correct an artifact that affects
   no released label. Paying a full re-validation cycle for that trade is worse
   engineering than sequencing it.
2. **Derive the slug from the filename prefix** (`local_captures_<slug>__…`) in
   the policy provider, leaving `merge_datasets` untouched. Rejected as a design
   choice even though it is the cheapest path: it would make masking semantics
   depend on a filename convention rather than on recorded provenance, which is
   the kind of implicit coupling that breaks silently later. The slug belongs in
   `image_provenance`, which today records only the source string.
3. **Disclose in the changelog and stop there.** Rejected. This project's
   recurring failure mode is a check that reports something other than what it
   claims — `dvc status -c` judged by stdout, a preflight that read identically
   before and after its own remediation, RG6 unfalsifiable, a lock entry no
   checkout could match. Prose does not fail a build. Hence the pinned test.
4. **Add a release gate for trusted-set correctness now.** Rejected for this
   release only: such a gate would fail today and block a ceremony over a defect
   every reviewer agrees is out of v0.6.0's scope. It ships with the v0.7.0 fix.

## Consequences

- **Constraint — `dataset-v0.6.0` must not be used for training with
  `missing_annotation_mitigation` enabled.** With ~69.5% of images on an
  over-broad trusted set, mitigation would mask far less than it should and
  supervise unlabelled objects as background. This is stated in the changelog.
- Positive: the Phase F2/F3 reproducibility evidence stays valid, and the RC
  machinery is exercised on a real release before v1.0 depends on it.
- Positive: the fix arrives with a gate, so the class of defect — a source
  declaring more trust than its constituent manifests support — becomes
  detectable rather than merely documented.
- Negative, accepted: `completeness.json` as published under `dataset-v0.6.0` is
  known to be wrong for 69.5% of images. Anyone consuming that tag inherits it,
  which is why the limitation is recorded in three places (this ADR, the
  changelog, and the test).

Related: [ADR-P5-13](ADR-P5-13-v060-local-capture-release-track.md),
[ADR-P5-07](ADR-P5-07-releases-as-code.md)

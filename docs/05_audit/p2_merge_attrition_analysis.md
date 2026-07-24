# P2 — Merge Attrition Analysis (local_captures)

**Generated:** 2026-07-24 · **Method:** read-only analysis of `merged_manifest.json`,
per-slug `_ingest_done.json` / `manifest.json` (SHA-256 hashes), flat `labels/`, and a
faithful replay of `src/dataset/dedup.py` aHash on the `wire` slug. No data mutated.

## Headline

Of **25,838** ingested local captures, only **13,820 (53.5%)** reached `data/merged`:

| Drop reason | Count | Root cause |
| --- | --- | --- |
| Empty label after remap | **5,650** | Every annotation on the image was dropped at ingest (see §1) |
| Perceptual near-duplicate | **6,368** | aHash Hamming<5 (flip-aware) vs a kept image (see §2) |
| **Merged (kept)** | **13,820** | — |

## §1 — Empty-label drops (5,650): mostly *out-of-taxonomy annotation loss*, not backgrounds

Empty labels concentrate in single-class slugs — toilet 67%, bed 76%, monitor 78%,
water_bottle 62%, stove 56%. They are driven by **22,053 annotations dropped as
`unmapped`** at ingest (vs 97,164 kept). Top droppers:

| slug | imgs | kept ann | dropped_unmapped |
| --- | --- | --- | --- |
| wet_floor | 2043 | 18223 | 10812 |
| walking_stick | 2598 | 4057 | 4294 |
| support_handle | 2250 | 1676 | 1667 |
| cupboard | 1043 | 3748 | 1427 |
| bed | 1316 | 419 | 1307 |
| door | 1130 | 1275 | 953 |
| monitor | 538 | 121 | 549 |
| sink | 798 | 507 | 532 |

**Cause:** the per-slug tables in `20_ingest_local_zips.py:_ARCHIVE_CLASS_REMAPS`
map only each slug's target classes and drop everything else — **including
in-taxonomy objects that appear as secondary classes**:
- `cupboard` drops `Chair` (cls 16)
- `sink` drops `Chair` (16) and `Stove` (6)
- `walking_stick` drops `knife` (5) and `person` (0) — 4,294 annotations
- `bed` drops `bottle` (~water_bottle 4)

⚠️ `wet_floor` dropped **10,812** with only a "separator string" documented as
dropped — this needs a source-class-name re-inspection (possible name-match loss of
real wet_floor boxes, or a genuinely multi-class source). **Open item for P7.**

**Recoverability:** partial. In-taxonomy secondary objects (Chair, Stove, knife,
person, water_bottle) are recoverable by extending the remap tables — high value for
under-represented classes. Truly out-of-taxonomy drops (fire, gun, sofa, table…) are
correctly dropped. Pure-background frames could be routed to negatives, but only under
the completeness policy's per-source trusted mask (an image labeled for one class may
still contain other unlabeled in-taxonomy objects → unsafe as a global negative).

## §2 — Near-duplicate drops (6,368): 96% are perceptual-only, not exact

| Class of drop | Count | % |
| --- | --- | --- |
| Exact SHA-256 twin among kept | 259 | 4.1% |
| **Perceptual-only (aHash Hamming<5)** | **6,109** | **95.9%** |

Perceptual drops concentrate in scarce single-object slugs with **zero** exact dups:
support_handle 1,084 · knife 875 · walking_stick 829 · wire 740 · person 526.

**Hamming-distance measurement (`wire`, 200 sampled drops vs 284 kept):**
Hamming 0:11 · 1:51 · 2:61 · 3:33 · 4:26 · ≥5:18. **~30% sit at Hamming 3–4** —
distinct frames the 8×8 (64-bit) aHash cannot separate. For `wire` only **284 of
1,024** frames survived (72% dropped).

**Assessment:** the 4.1% exact dups are correct drops. The 96% perceptual-only drops
at 8×8-aHash/Hamming<5 are **over-aggressive for continuous capture sessions**,
discarding genuine viewpoint/lighting diversity and worsening imbalance for the
scarcest classes.

## Recommendations (feed P7 / dedup tuning)

1. **Tighten local_captures dedup**: higher-resolution hash (pHash/dHash 16×16 = 256
   bits) and/or lower Hamming threshold (e.g. <2), or drop only exact+flip-exact for
   local_captures while keeping perceptual dedup for augmented public sources.
   Estimated recovery: ~2,000–3,500 frames, weighted to scarce classes.
2. **Extend remap tables** to keep in-taxonomy secondary classes (Chair, Stove,
   knife, person, water_bottle) — recovers real annotations for scarce classes.
3. **Re-inspect wet_floor source class names** (10,812 unmapped) for name-match loss.
4. Re-run leakage QA after any dedup change (train/test leakage check must stay green).

# ADR-P4-03 — Policy-Indirection Completeness Artifact

**Status:** Accepted (Phase 4, 2026-07)

## Context

The masked loss needs, per processed image, the set of taxonomy classes its
source annotates exhaustively. Per-source `trusted_classes` already existed in
`configs/dataset_sources.yaml` and propagate to
`MergedManifest.label_completeness`; per-image granularity, validation, and a
train-time contract did not. Critically, `trusted_classes: []` is ambiguous:
for `negatives` it means "verified absence of ALL classes" (all 23 classes
trusted-absent → all-ones mask), for `custom_captures` it means "declared per
capture session".

## Decision

A DVC stage (`generate_completeness`, between split and the frozen train
stage) compiles `data/processed/completeness.json`:

- **`policies`**: few named entries, each an explicit `mode` +
  materialized `trusted_class_ids`;
- **`images`**: every processed image basename → policy key + split;
- **`taxonomy`**: nc + names + SHA-256 fingerprint;
- **`inputs`**: SHA-256 of merged manifest and split summary (freshness gate).

Semantics are declared, never inferred: a new **top-level** `completeness:`
section in `configs/dataset_sources.yaml` assigns each source a policy mode
(`trusted_list` | `verified_absence_all` | `per_session`). Top-level placement
is deliberate — keys under `sources.*` would change the DVC params-hashes of
the acquisition stages and force re-downloads. The generator hard-fails on
ANY ambiguity (unknown image, unmapped source, config/manifest drift, unknown
class name, duplicate filenames/policy keys, unfinalized capture sessions);
unused policies are warnings.

## Alternatives considered

1. **Per-image trusted-class lists** — redundant (30k × 23 at full scale),
   hides the per-source semantics, and bloats diffs. Policy indirection keeps
   the artifact reviewable.
2. **Train-time reads of the merged manifest** (no artifact) — rejected: no
   materialized, hashable, DVC-tracked contract; no place for validation to
   fail early; provenance recovery via string prefixes would run inside the
   training loop.
3. **Hardcoded source-name special cases** (e.g. `if source == "negatives"`)
   — rejected: silently wrong the day a second verified-negative source
   appears; semantics belong in config, resolution in a registry
   ([ADR-P4-05](ADR-P4-05-policy-provider-registry.md)).

## Consequences

- Positive: single validated source of truth; taxonomy/input fingerprints
  make staleness detectable (preflight G2/G7); additive schema (consumers
  ignore unknown keys, matching the manifest convention).
- Negative: one more DVC stage and artifact to regenerate after dataset
  rebuilds — mitigated by gate G7's explicit "re-run
  `dvc repro generate_completeness`" diagnostics.

Related: [ADR-P4-05](ADR-P4-05-policy-provider-registry.md), risks R26/R27.

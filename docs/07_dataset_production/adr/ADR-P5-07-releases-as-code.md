# ADR-P5-07 — Releases as Code: Gate-Driven Manifests, Frozen `record_release`, v0.5→v1.0 Ladder

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Dataset releases must be reproducible, auditable, and impossible to cut when
quality/leakage/license criteria are unmet — not a manual "tag and hope".

## Decision

`18_make_release.py` (`check`/`make`/`verify`) drives release gates RG1–RG10
plus MODE/WETFLOOR prerequisites, implemented in `src/dataset/release/gates.py`
as **pure functions over already-loaded data** (no gate recomputes what an
earlier stage established). Per-track thresholds live in `configs/release.yaml`;
`evaluate_release` computes only the gates a track opts into. `make` writes a
`release_manifest.json` (counts, gate results, artifact hashes, licenses,
reproducibility block) snapshotted by the **frozen** `record_release` stage.
Ladder: **v0.5.0** full public build + first push · **v0.7.0** + ≥3,000 verified
cells · **v0.9.0** + custom captures, locked eval, house split, wet_floor
decision · **v1.0.0** all targets + 0 criticals + A/B evidence. The
`GateResult`/report shape duplicates `src/training/preflight.py`'s rather than
importing it — `src/dataset` must never depend on `src/training`.

## Alternatives considered

1. **Manual release checklist in a runbook.** Rejected: non-reproducible, easy
   to skip a check; a release with a leaked eval set could ship.
2. **Import the preflight `GateResult` from `src/training`.** Rejected: wrong
   layering direction; the shape is cheap to duplicate.

## Consequences

- Positive: a release is a gated, hash-pinned, immutable snapshot; a negative
  check (e.g. smoke build fails MODE for a full track) is itself an acceptance
  test.
- Constraint: cutting a real release is a semi-irreversible human action (tag +
  `dvc push` + `dvc commit -f record_release`) performed only when gates pass.

Related: [ADR-P5-08](ADR-P5-08-cross-dataset-label-salvage.md),
[ADR-P5-09](ADR-P5-09-local-dvc-remote.md),
[ADR-P5-10](ADR-P5-10-ab-benchmark-acceptance-evidence.md)

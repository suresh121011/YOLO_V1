# ADR-P5-13 — A `dataset-v0.6.0` Track for the Local-Capture Build; RG9/RG10 Stay Mandatory for v1.0.0

**Status:** Accepted 2026-07-27
**Deciders:** Phase-5 engineering (user decision recorded in the Phase-F plan)

## Context

The build now carries 17,888 images: the public sources plus 23 local-capture
slugs ingested by `scripts/dataset/20_ingest_local_zips.py`. It is QA-green,
reproducible byte-for-byte across platforms, and mirrored to three storage
locations. It has never been validated by any machine other than the one that
built it.

Phase F (RC validation) needs a release track to validate *against*, and the
audit that preceded it established that `dataset-v1.0.0` cannot be that track.
Two of its gates fail structurally rather than numerically:

- **RG9** reads capture-session manifests from `data/raw/custom_captures` via
  `load_session_manifests()`. That directory is empty — `capture_progress.json`
  reports `total_images: 0`, `houses: []`, and all eight custom classes at `0`
  against required 2000 / 3 houses / 200-per-class. The local-capture images
  live in `data/raw/local_captures`, a different provenance channel that RG9
  never reads, and archives sourced this way are not "Indian-home captures
  across three houses" in any case.
- **RG10** requires `eval_report.json` and an `ab_benchmark/` directory.
  `data/eval/indian_home_v0/images` is empty, so `evaluate_model.py --split
  eval` cannot run, and the two-arm A/B of ADR-P5-10 has not been executed.

The existing lower tracks do not fit either. `dataset-v0.5.0` is reachable but
its own comment describes a "full-mode **public** build", which under-describes
what this is. `dataset-v0.7.0` requires `min_verified_cells: 3000` (the ledger
holds 0) and `dataset-v0.9.0` requires the wet_floor IAA pilot decision (no
`iaa_*.json` exists), so both genuinely fail.

## Decision

Add one track, `dataset-v0.6.0`: `mode: full`, gates `RG1`–`RG8`,
`min_verified_cells: 0`.

**RG9 and RG10 are not relaxed, re-pointed, or re-defined.** They remain
mandatory for `dataset-v1.0.0` exactly as written. The new track simply does
not claim the evidence it does not have.

The track is strictly stronger than `dataset-v0.5.0`: it adds RG8, so split
leakage must be zero. With no locked eval set, RG8's eval-overlap and
house-exclusivity arms report `available: false` and are skipped
(`src/dataset/release/gates.py:460`) — that is the gate's designed behaviour for
absent evidence, not a bypass, and the leakage arm it does evaluate is real.

`allow_noncommercial: true` is already set, so this ships as a **research-scope**
release with WIDER FACE enumerated — the same caveat `dataset-v1.0.0` carries
(docs/04 §2). Any commercial build still requires the documented v1.1 path.

## Alternatives considered

1. **Re-point RG9 at `local_captures` and lower its thresholds to what was
   built.** Rejected. It would make the gate pass by redefining what it
   measures, and RG9's purpose is specifically Indian-home capture diversity
   across households — a property scraped archives do not have at any count.
   `configs/release.yaml` already requires a recorded decision to move
   thresholds; this would have been a decision to stop measuring the thing.
2. **Tag `dataset-v0.5.0` instead.** Rejected as merely inaccurate: the track's
   own comment says "public build", and a consumer reading the manifest would
   not learn that 14k local-capture images are present.
3. **Wait for the real captures and validate Phase F directly against v1.0.0.**
   Rejected for sequencing, not principle. The captures are weeks of human work;
   deferring means the RC-validation machinery gets exercised for the first time
   during the v1.0 cut, which is precisely when a novel failure is most
   expensive.

## Consequences

- Positive: the RC-validation harness is proven end-to-end on a real release
  before v1.0 depends on it, and the release ladder gains a track that describes
  the build honestly.
- Positive: `dataset-v1.0.0`'s gate list is untouched, so the v1.0 bar does not
  quietly move.
- Constraint: `dataset-v0.6.0` carries **no training evidence**. It certifies the
  data, not that the data trains a good model. The changelog entry and release
  notes must say so explicitly.
- Constraint: `record_release.cmd` in `dvc.yaml` is version-pinned and must be
  updated per release; it was still pointing at `dataset-v0.5.0`.
- **Observation from the first baseline check (2026-07-27).** With
  `min_verified_cells: 0` and no `min_coverage_score`, RG3 on this track reduces
  to a file-*existence* check: it reported `[PASS] coverage-quality` against
  coverage/quality reports that still describe the 188-image smoke build. RG1
  caught the same staleness (`l4_l5_report_warnings: 2`, "image count 188 != live
  dataset 17888"), so the release is still blocked and no false green reaches a
  tag. But RG3's name overstates what it verifies, and a track that required RG3
  without RG1 would not be protected. Recorded, not fixed — changing RG3's
  contract is a separate decision, and Phase C resolves the underlying staleness.

Related: [ADR-P5-07](ADR-P5-07-releases-as-code.md),
[ADR-P5-10](ADR-P5-10-ab-benchmark-acceptance-evidence.md)

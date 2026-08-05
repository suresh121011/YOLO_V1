# Phase-5 Milestone Status

Status tracker for Phase 5 (**Production Dataset Engineering** — this is the
current phase's own name, not a future one; see "Phase naming" below).
Reflects the repository as verified on **2026-08-06** at commit `fea0d2a`
(local `main`, 4 commits ahead of `origin/main`, not yet pushed).

**Bottom line:** all milestone tooling (M0–M11) remains implemented and
tested (suite **1207** passing, 1 skipped, 0 failed — 3 pre-existing
test/prod drifts found and fixed this pass). The CVAT verification campaign
(H-C) is now genuinely underway — 2 of 44 batches imported — but that is
11% of the batch count and the ledger sits at 341/3000 cells (11.4%) against
the `dataset-v0.7.0` gate. **Phase 5 is not complete.** See the full
release-readiness assessment for the session that produced this revision.

## Phase naming (correction)

"Production Dataset Engineering" is Phase 5 **itself** — see
`docs/01_executive_implementation_plan/implementation_phases.md`'s
executed-phase table. It is not a phase that follows the current one. The
next phase is **Phase 6 — Model Training & Integration**, which three
independent docs (`implementation_phases.md`, ADR-P5-10,
`docs/07_dataset_production/README.md`) agree does not start until
**Dataset v1.0.0** ships (RG1–RG10 all pass). That is a materially higher
bar than `v0.7.0` — see the release ladder below.

Legend: ✅ done · ⏳ tooling done / execution pending (operational) · 👤 human track

## Milestones

| # | Scope | Tooling | Remaining (operational/human) |
|:--|:--|:--|:--|
| M0 | Scaffolding, DVC remote, first push | ✅ | — |
| M1 | Auto-annotation core (L2) | ✅ | ⏳ this session's `.venv` has no `ultralytics` — a fresh GPU env is needed before `auto_annotate` can re-run against the current (post-vb001/vb002) ledger |
| M2 | CVAT verification round-trip | ✅ | 👤 continue batches (H-C) — 2/44 imported |
| M3 | Completeness expansion + label overlay | ✅ | ✅ activated — `completeness.json` regenerated against the 87-image ledger 2026-08-06 |
| M4 | Coverage (L4) + quality (L5) reports | ✅ | ⏳ **stale** (2026-07-29 numbers) — blocked on the `auto_annotate` GPU re-run above; `completeness.json` itself is current but `coverage_report`/`dataset_quality_report` haven't re-run since |
| M5 | Release automation (gates RG1–RG10) | ✅ | ✅ `dataset-v0.6.0` remains the last cut release; ladder tracks v0.7.0 → v1.0.0 |
| M6 | Correctness-validation gate | ✅ | — (PASS committed) |
| M7 | Full-mode transition + Dataset v0.5.0 | ✅ | 👤 Roboflow slugs (H-B) still open — see note below, not currently gate-blocking |
| M8 | Verification at scale + v0.7.0 | ✅ | 👤 **341/3000 verified cells (11.4%)**; per-class `coverage_score`: charger 0.12 (fails 0.5 floor — binding constraint), wire 0.99, medicine_bottle 0.78, cupboard 0.63 (all three already pass — **note: coverage numbers are the stale 07-29 snapshot**, re-verify once `coverage_report` regenerates) |
| M9 | Custom capture integration + eval lock + v0.9.0 | ✅ | 👤 captures (H-A) — **0 images, 0 houses**; wet_floor pilot; eval lock — `data/eval/indian_home_v0` is empty |
| M10 | Evaluation + full-scale A/B evidence | ✅ | ⏳ real GPU A/B run — **two** full training runs per ADR-P5-10 (mitigation on/off), not one; blocked on locked eval set (M9) |
| M11 | Dataset v1.0.0 + Phase-6 readiness | ✅ | ⏳ full release ladder + unfreeze `train_yolo11n`/`evaluate_yolo11n` |

## Human tracks

- **H-A — Custom capture campaign** 👤 — signed consent per household,
  capture toward ≥3 houses / ≥2,000 images / ≥200 instances per custom class,
  ingest → dual-annotator CVAT → IAA → finalize, then enable `custom_captures`.
  Status: **0 images, 0 houses** (confirmed by direct file count 2026-08-06).
  Blocks `v0.9.0` (≥1,000 images/≥2 houses) and `v1.0.0` (≥2,000/≥3) via RG9.
  **Does not block `v0.7.0`** — RG9 is not in that track's gate list.
  Runbook: `docs/04_dataset_engineering/capture_annotation_runbook.md`.
- **H-B — Roboflow licensing** 👤 — search Roboflow Universe for
  `medicine_bottle`/`charger`/`wire`/`gas_cylinder`, record slug + version +
  license + class mapping, populate `sources.roboflow.datasets`, set
  `ROBOFLOW_API_KEY`. Status: `datasets: []` (empty).
  **Correction from the prior revision of this doc:** RG7 (`rg7_license_gate`
  in `src/dataset/release/gates.py`) passes vacuously when no Roboflow data
  has been ingested — it does not currently fail any release. The real value
  of H-B right now is lifting `charger`'s coverage_score (currently 0.12,
  well under the 0.5 floor `v0.7.0` requires) — Roboflow charger images would
  help this directly. Framed as an RG3 accelerant, not an RG7 blocker.
- **H-C — CVAT verification campaign** 👤 — create tasks from `cvat_labels.json`
  (now includes `"type": "rectangle"` per entry — the earlier CVAT
  label-type bug is fixed at the source, commit `38c4b7c`), verify candidate
  boxes, dual-annotate the 10% IAA sample where present, export → import →
  `dvc commit -f`.
  **Status: 2 of 44 batches imported** (`vb002_cross_dataset`: 7 images, 21
  cells, IAA 1.00; `vb001_yolo_world`: 80 images, 320 cells, no IAA sample
  for this batch, imported with `--allow-missing-base` since it predates the
  merge rebuild — see `feat(annotation)` commit `aa1ed07`).
  **Ledger: 87 images, 341 cells verified** — 11.4% of the 3,000-cell
  `v0.7.0` floor. **42 batches / ~8,137 images remain.**
  Runbook: `verification_runbook.md`.

## Release ladder (`configs/release.yaml`)

| Track | Gates | Binding blockers today |
|:--|:--|:--|
| `dataset-v0.5.0` | RG1–RG7 | shipped |
| `dataset-v0.6.0` | RG1–RG8 | **shipped** (current HEAD tag) |
| `dataset-v0.7.0` | RG1–RG7 | RG3: 341/3,000 cells; charger coverage_score 0.12 < 0.5 (stale number — coverage_report needs the blocked `auto_annotate` re-run to confirm current value) |
| `dataset-v0.9.0` | RG1–RG8 | RG9: 0 custom images/houses (H-A untouched); RG3 as above at higher bar implicitly via continued verification |
| `dataset-v1.0.0` | RG1–RG10 | RG9: ≥2,000 images/≥3 houses; RG10: locked eval set (0 images) + **two** full GPU training runs (ADR-P5-10) + `ab_benchmark/` + `eval_report.json`; RG3 at full scale |

## Known gaps found and fixed this session (2026-08-06)

- `src/dataset/completeness.py`: `generate_completeness` hard-failed on any
  ledger entry whose image predates the current merge snapshot (vb001's 80
  images, none of which are in `data/merged/merged_manifest.json`'s
  `image_provenance` post mode:full rebuild). Now skipped with a warning,
  mirroring `--allow-missing-base`. Commit `adf6106`.
- Three pre-existing test/prod drifts (unrelated to this session's direct
  work, found by a full-suite run): `test_cvat_package.py` and
  `test_candidate_artifact.py` had stale expectations; `test_dedup_budget.py`'s
  mock was missing a `hash_size` kwarg added during earlier P2 dedup tuning.
  All three fixed and verified passing. Commit `55062b7`.
- A failed `auto_annotate` re-run attempt (missing `ultralytics` in this
  `.venv`) deleted `data/annotation/candidates` before crashing. Recovered
  via `dvc checkout` from cache — confirmed no data loss.

## Current shared-state facts

*Re-verified 2026-08-06.*

- Build mode: **`full`** — 24,352 images / re-split 2026-08-06 (group-aware,
  seed 42): 20,588 train / 1,882 val / 1,882 test. Leakage verification: PASS.
- `dataset-v0.6.0` remains the last cut release. `v0.7.0`/`v0.9.0`/`v1.0.0`
  do not exist yet.
- Ledger: **87 images, 341 cells verified** (see H-C above).
- `custom_captures/` and `eval/indian_home_v0/` are still empty (0 files each).
- `main` is **4 commits ahead of `origin/main`**, not pushed (awaiting
  explicit confirmation — standing project rule).
- DVC object-integrity sweep (`scripts/qa/verify_lock_objects.py`): **PASS**,
  67,810/67,810 hashes resolve, 6 correctly skipped (`cache: false`).
- `dvc status -c` unchecked this pass (remote push out of scope without
  confirmation); local cache/workspace consistency confirmed via the sweep
  above and targeted `dvc status`/`dvc checkout`.
- Full test suite: **1207 passed, 1 skipped, 0 failed** (after this
  session's 3 fixes).
- Two pre-existing frozen-stage defects remain **unresolved and blocked on
  human/GPU work**, not on anything fixable in-repo right now:
  `train_yolo11n`'s `dvc.lock` hashes are unreproducible (fix: real re-run,
  premature before the dataset nears v1.0 readiness); `evaluate_yolo11n` is
  declared in `dvc.yaml` but absent from `dvc.lock` and depends on the
  currently-empty locked eval set.

See [`../../CHANGELOG.md`](../../CHANGELOG.md) `[Unreleased]` for the full
Phase-5 change list and [`adr/`](adr/README.md) for the design decisions.

# Phase-5 Milestone Status

Status tracker for Phase 5 (**Production Dataset Engineering** — this is the
current phase's own name, not a future one; see "Phase naming" below).
Reflects the repository as verified on **2026-08-06** at commit `4b5b898`
(local `main`, in sync with `origin/main`, working tree clean except two
intentionally untracked scratch items).

**Bottom line:** the engineering/tooling side of Phase 5 has no known
blockers. All milestone tooling (M0–M11) is implemented and tested (suite
**1207** passing, 1 skipped, 0 failed). H-A, H-B, and H-C are **ongoing
operational tracks that continue in parallel** and do not block further
engineering work — but they, plus RG9/RG10, remain the reason
`dataset-v1.0.0` (and therefore Phase 6) is not yet reachable. This is not a
phase transition: we remain in Phase 5.

## Phase naming (no change — explicitly not redefined)

"Production Dataset Engineering" is Phase 5 **itself** — see
`docs/01_executive_implementation_plan/implementation_phases.md`'s
executed-phase table. It is not a phase that follows the current one, and
this doc does not redefine that model. The next phase is **Phase 6 — Model
Training & Integration**, which three independent docs
(`implementation_phases.md`, ADR-P5-10, `docs/07_dataset_production/README.md`)
agree does not start until **Dataset v1.0.0** ships (RG1–RG10 all pass).

Legend: ✅ done · ⏳ tooling done / execution pending (operational) · 👤 ongoing operational track (non-blocking to engineering)

## Milestones

| # | Scope | Tooling | Remaining (operational, non-blocking) |
|:--|:--|:--|:--|
| M0 | Scaffolding, DVC remote, first push | ✅ | — |
| M1 | Auto-annotation core (L2) | ✅ | ⏳ this `.venv` has no `ultralytics` — a GPU-provisioned env is needed before `auto_annotate` re-runs against the current ledger |
| M2 | CVAT verification round-trip | ✅ engineering complete | 👤 H-C batches continue (2/44 imported) — operational, not an engineering task |
| M3 | Completeness expansion + label overlay | ✅ | ✅ activated — `completeness.json` current as of 2026-08-06 |
| M4 | Coverage (L4) + quality (L5) reports | ✅ | ⏳ stale (2026-07-29 numbers) — blocked on the `auto_annotate` GPU re-run above |
| M5 | Release automation (gates RG1–RG10) | ✅ | ✅ `dataset-v0.6.0` remains the last cut release |
| M6 | Correctness-validation gate | ✅ | — (PASS committed) |
| M7 | Full-mode transition + Dataset v0.5.0 | ✅ | 👤 H-B — see status below |
| M8 | Verification at scale + v0.7.0 | ✅ | 👤 341/3,000 verified cells (11.4%) — H-C, ongoing |
| M9 | Custom capture integration + eval lock + v0.9.0 | ✅ | 👤 H-A, ongoing — 0 images, 0 houses |
| M10 | Evaluation + full-scale A/B evidence | ✅ | ⏳ reserved for the official v1.0 A/B run (see DVC lock defects below) |
| M11 | Dataset v1.0.0 + Phase-6 readiness | ✅ | ⏳ full release ladder — gated on M8/M9/M10 above |

## Operational tracks (ongoing, non-blocking to engineering)

Per explicit direction (2026-08-06): H-A, H-B, and H-C are tracked as
ongoing operational activities. They are not gates on further engineering
progress in this repository, but they remain the actual gates on
`dataset-v0.7.0`/`v0.9.0`/`v1.0.0` and thus on Phase 6, exactly as
`configs/release.yaml` already documents — nothing about the release gate
semantics has changed, only how this doc frames the work tracks.

- **H-A — Custom capture campaign** 👤 — signed consent per household,
  capture toward ≥3 houses / ≥2,000 images / ≥200 instances per custom class,
  ingest → dual-annotator CVAT → IAA → finalize, then enable `custom_captures`.
  Status: **0 images, 0 houses**. Blocks `v0.9.0`/`v1.0.0` via RG9; does not
  block `v0.7.0` or any engineering work.
  Runbook: `docs/04_dataset_engineering/capture_annotation_runbook.md`.
- **H-B — Roboflow licensing** 👤 — search Roboflow Universe for
  `medicine_bottle`/`charger`/`wire`/`gas_cylinder`, record slug + version +
  license + class mapping, populate `sources.roboflow.datasets`, set
  `ROBOFLOW_API_KEY`. Status: `datasets: []` (empty).
  **Attempted 2026-08-06, not completable with available tooling**: web
  search surfaced unverified leads (e.g. an India-specific gas-cylinder
  dataset, several charger/cable/pill datasets of varying relevance) but two
  independent verification paths both failed — `WebFetch` against
  `universe.roboflow.com` returns HTTP 403 (blocks non-browser requests),
  and the Chrome browser extension was not connected for an interactive
  session. Unverified search-summary text is not a sound basis for a
  licensing decision baked into training data, so no dataset was selected
  and `sources.roboflow.datasets` was not populated.
  **Does not block RG7** — `rg7_license_gate` passes vacuously while no
  Roboflow data is ingested. Its value is lifting `charger`'s coverage_score
  (0.12 vs the 0.5 floor `v0.7.0` needs), not unblocking a currently-failing
  gate.
  **Next step (needs a human):** browse Roboflow Universe directly (its
  filterable search/license badges are far more reliable than text search),
  pick 1 slug + version per class, confirm the license is compatible with
  this project's noncommercial posture, and hand back the 4 slugs — I can
  then wire them into `configs/dataset_sources.yaml` and record the ADR.
- **H-C — CVAT verification campaign** 👤 — engineering/tooling complete
  (create tasks from `cvat_labels.json`, verify boxes, dual-annotate the IAA
  sample, export → import → `dvc commit -f`, all proven end-to-end on real
  batches). **Status: 2 of 44 batches imported**, 87 images, 341 of 3,000
  ledger cells (11.4%). 42 batches / ~8,137 images remain — pure human
  annotation work from here.
  Runbook: `verification_runbook.md`.

## DVC lock defects — correctly blocked, not engineering bugs

Both are declared+deps'd but frozen (`dvc.yaml`), by design (see the header
note there on human-loop stages). Neither is a code defect; both are
intentionally reserved for real evidence that doesn't exist yet:

- **`evaluate_yolo11n`**: depends on `data/eval/indian_home_v0`, the locked
  evaluation dataset — currently **0 images**. This is H-A's eval-lock
  deliverable, not something this environment can produce.
- **`train_yolo11n`**: its `dvc.lock` dependency hashes are unreproducible
  against current HEAD. The correct fix is a real training run — but that
  run is reserved for the **official Dataset v1.0 A/B benchmark** defined in
  ADR-P5-10 (two full runs, mitigation on/off, evaluated against the locked
  eval set). Running it now, before the eval set exists and while the
  dataset is at 11.4% verification, would produce non-representative
  evidence that has to be redone anyway. This session's `.venv` also has no
  `ultralytics` installed, so it isn't currently equipped to run it even as
  a placeholder — and no placeholder/smoke artifacts were created, per
  explicit direction, to avoid a false-green `dvc.lock` entry.

Both remain deferred to when Dataset v1.0's other gates (RG9 custom
captures, locked eval set) are actually satisfied.

## Release ladder (`configs/release.yaml`) — unchanged, for reference

| Track | Gates | Binding blockers today |
|:--|:--|:--|
| `dataset-v0.5.0` / `v0.6.0` | RG1–RG7 / RG1–RG8 | shipped |
| `dataset-v0.7.0` | RG1–RG7 | RG3: 341/3,000 cells; charger coverage_score 0.12 < 0.5 (stale — `coverage_report` blocked on the `auto_annotate` GPU re-run) |
| `dataset-v0.9.0` | RG1–RG8 | RG9: 0/1,000 custom images, 0/2 houses (H-A) |
| `dataset-v1.0.0` | RG1–RG10 | RG9: 2,000/3 houses; RG10: locked eval set + two full GPU A/B runs (ADR-P5-10) |

## Known gaps found and fixed this session (2026-08-06)

- `src/dataset/completeness.py`: `generate_completeness` hard-failed on any
  ledger entry predating the current merge snapshot (vb001's 80 images).
  Fixed to skip + warn (naming the affected batch ids), verified against
  the real 24,352-image dataset with unchanged metrics. Commits `adf6106`,
  `4b5b898` (test coverage added: `test_completeness_generation.py`,
  `test_verification_ledger.py`).
- Three pre-existing test/prod drifts unrelated to this session's direct
  work, found by a full-suite run, fixed and verified. Commit `55062b7`.
- A failed `auto_annotate` re-run attempt (missing `ultralytics`) deleted
  `data/annotation/candidates` before crashing; recovered via `dvc checkout`
  from cache, confirmed no data loss.
- H-B attempted and found genuinely blocked on tooling access (see above) —
  reported rather than guessed.

## Current shared-state facts

*Re-verified 2026-08-06.*

- Build mode: **`full`** — 24,352 images, re-split (group-aware, seed 42):
  20,588 train / 1,882 val / 1,882 test. Leakage verification: PASS.
- `dataset-v0.6.0` remains the last cut release.
- Ledger: **87 images, 341 cells verified**.
- `custom_captures/` and `eval/indian_home_v0/` are still empty (0 files each).
- `main` is **in sync with `origin/main`** (pushed), working tree clean
  except `data/annotation/batches - Shortcut.lnk` (untracked, looks
  accidental) and `exports/` (untracked scratch — not meant to be
  git-tracked per the runbook).
- DVC object-integrity sweep: **PASS**, 67,810/67,810 hashes resolve.
- Full test suite: **1207 passed, 1 skipped, 0 failed**.
- No known engineering blockers remain. Everything left in Phase 5 is
  operational (H-A/H-B/H-C) or reserved for a real GPU/eval-set milestone
  (M1 re-run, M10 A/B evidence) — not fixable by further code changes.

See [`../../CHANGELOG.md`](../../CHANGELOG.md) `[Unreleased]` for the full
Phase-5 change list and [`adr/`](adr/README.md) for the design decisions.

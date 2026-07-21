# Phase-5 Milestone Status

Status tracker for Phase 5 (Production Dataset Engineering / Dataset v1.0),
branch `phase-5-production-dataset-engineering`. Reflects the repository as
verified on **2026-07-21** (three-agent cold audit, not prior summaries).

**Bottom line:** all milestone *tooling* (M0–M11) is implemented, tested, and
validated (M6 correctness gate PASS; suite 1003 passing). Everything that
remains is **operational or human-only** — real captures, human verification,
Roboflow licensing, the full-mode download, the GPU A/B run, and cutting real
releases. No Phase-5 code gaps remain.

Legend: ✅ done · ⏳ tooling done / execution pending (operational) · 👤 human track

## Milestones

| # | Scope | Tooling | Remaining (operational/human) |
|:--|:--|:--|:--|
| M0 | Scaffolding, DVC remote, first push | ✅ | — |
| M1 | Auto-annotation core (L2) | ✅ | ⏳ real GPU candidate generation at scale |
| M2 | CVAT verification round-trip | ✅ | 👤 stand up self-hosted CVAT + real batches (H-C) |
| M3 | Completeness expansion + label overlay | ✅ | — (activates as the ledger fills) |
| M4 | Coverage (L4) + quality (L5) reports | ✅ | — |
| M5 | Release automation (gates RG1–RG10) | ✅ | ⏳ cut real releases |
| M6 | Correctness-validation gate | ✅ | — (PASS committed) |
| M7 | Full-mode transition + Dataset v0.5.0 | ✅ | 👤 Roboflow slugs (H-B); ⏳ flip `mode: full` + real download + v0.5.0 |
| M8 | Verification at scale + v0.7.0 | ✅ | 👤 ≥3,000 verified cells (H-C); ⏳ v0.7.0 |
| M9 | Custom capture integration + eval lock + v0.9.0 | ✅ | 👤 captures (H-A), wet_floor pilot, eval lock; ⏳ v0.9.0 |
| M10 | Evaluation + full-scale A/B evidence | ✅ | ⏳ real GPU A/B run (2 arms) |
| M11 | Dataset v1.0.0 + Phase-6 readiness | ✅ | ⏳ full release ladder + unfreeze `train_yolo11n` |

## Human tracks

- **H-A — Custom capture campaign** 👤 — signed consent per household,
  capture toward ≥3 houses / ≥2,000 images / ≥200 instances per custom class,
  ingest → dual-annotator CVAT → IAA → finalize, then enable `custom_captures`.
  Status: 0 images captured. Runbook: `docs/04_dataset_engineering/capture_annotation_runbook.md`.
- **H-B — Roboflow licensing** 👤 — search Roboflow Universe for
  `medicine_bottle`/`charger`/`wire`/`gas_cylinder`, record slug + version +
  license + class mapping, populate `sources.roboflow.datasets`, set
  `ROBOFLOW_API_KEY`. Status: `datasets: []` (empty) — blocks RG7.
- **H-C — CVAT verification campaign** 👤 — stand up self-hosted CVAT (Docker)
  + pin version, create tasks from `cvat_labels.json`, verify candidate boxes,
  dual-annotate the 10 % IAA sample, export → import → `dvc commit -f`. Status:
  ledger empty (0 cells). Runbook: `verification_runbook.md`.

## Group C — ready, waiting only on execution (no code needed)

All five download stages (full mode) · `auto_annotate` (GPU) ·
`train_yolo11n`/`evaluate_yolo11n` (frozen, GPU) ·
`ingest_custom_captures`/`ingest_eval_set` (frozen, human data) · the
verification-loop stages (human CVAT) · `record_release` (needs a passing full
build). Each is fully implemented and gated only on real data / real compute /
a passing full build.

## Current shared-state facts

- Build mode: `smoke` (`configs/dataset_sources.yaml`); no full-mode download run.
- No release tags cut (`dataset-v0.5.0…v1.0.0` do not exist); `data/releases/`
  is empty.
- Ledger, `verified_labels/`, `custom_captures/`, and `eval/indian_home_v0/` are
  empty (drill artifacts only).
- The branch has **not** been pushed to `origin` — every commit exists locally
  until the first push.
- DVC remote: `localstore` → `C:\dvc_remote` (default, off OneDrive);
  `storage` → S3 (secondary, credentials not configured in-tree).

See [`../../CHANGELOG.md`](../../CHANGELOG.md) `[Unreleased]` for the full
Phase-5 change list and [`adr/`](adr/README.md) for the design decisions.

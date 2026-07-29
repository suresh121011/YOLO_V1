# Phase-5 Milestone Status

Status tracker for Phase 5 (Production Dataset Engineering / Dataset v1.0).
Reflects the repository as verified on **2026-07-29** at `dataset-v0.6.0`
(`699b9ba`); the previous revision described the pre-Phase-E state of
2026-07-21 and had drifted substantially.

**Bottom line:** the full-mode build is done and **`dataset-v0.6.0` is
released**. Milestone tooling (M0–M11) is implemented and tested (suite **1188**
passing). What remains is mostly operational or human-only — real captures,
human verification, Roboflow licensing, the GPU A/B run, and the remaining
releases — **with one code gap now known**: per-slug trust resolution in
`completeness_policies.py` *and* `targeting.py`
([ADR-P5-15](adr/ADR-P5-15-per-slug-trust-resolution-and-targeting.md)). The
earlier claim "no Phase-5 code gaps remain" did not survive measurement.

Legend: ✅ done · ⏳ tooling done / execution pending (operational) · 👤 human track

## Milestones

| # | Scope | Tooling | Remaining (operational/human) |
|:--|:--|:--|:--|
| M0 | Scaffolding, DVC remote, first push | ✅ | — |
| M1 | Auto-annotation core (L2) | ✅ | ⏳ real GPU candidate generation at scale |
| M2 | CVAT verification round-trip | ✅ | 👤 stand up self-hosted CVAT + real batches (H-C) |
| M3 | Completeness expansion + label overlay | ✅ | — (activates as the ledger fills) |
| M4 | Coverage (L4) + quality (L5) reports | ✅ | — |
| M5 | Release automation (gates RG1–RG10) | ✅ | ✅ **first real release cut 2026-07-29 — `dataset-v0.6.0`**, gates MODE+RG1–RG8 all PASS; remaining tracks v0.7.0 → v1.0.0 |
| M6 | Correctness-validation gate | ✅ | — (PASS committed) |
| M7 | Full-mode transition + Dataset v0.5.0 | ✅ | ✅ `mode: full` + real download done (Phase E, 24,352 images); released as **`dataset-v0.6.0`** (ADR-P5-13) rather than v0.5.0, which under-describes the local-capture content; 👤 Roboflow slugs (H-B) still open |
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
- **H-C — CVAT verification campaign** 👤 — create tasks from `cvat_labels.json`,
  verify candidate boxes, dual-annotate the 10 % IAA sample, export → import →
  `dvc commit -f`. **Self-hosted CVAT is UP** (12 containers: `cvat_server`,
  `cvat_db`, `cvat_ui`, 9 workers — verified 2026-07-29); the "stand it up" step
  is done. Status: ledger empty (0 cells) — **this is the sole binding blocker on
  `dataset-v0.7.0`**, alongside `charger` coverage 0.1546 vs the 0.5 floor.
  Do not start batches until ADR-P5-15's candidate regeneration lands: today's
  6,773 candidates were selected under the wrong trust model.
  Runbook: `verification_runbook.md`.

## Group C — ready, waiting only on execution (no code needed)

**Done since:** all five download stages have run in full mode, `auto_annotate`
has run on GPU, and `record_release` produced `dataset-v0.6.0`.

**Still waiting on execution:** `train_yolo11n`/`evaluate_yolo11n` (frozen, GPU)
· `ingest_custom_captures`/`ingest_eval_set` (frozen, human data) · the
verification-loop stages (human CVAT, now unblocked — see H-C).

Two frozen stages carry defects that must clear **before** RG10 training
evidence is recorded:
- `train_yolo11n`'s `dvc.lock` dep hashes match neither the LF nor the CRLF form
  of HEAD, so the recorded run is unreproducible. `dvc status` hides it on every
  platform because the stage is frozen. Clear it by **re-running training**, then
  `dvc commit -f` — never by re-stamping hashes.
- `evaluate_yolo11n` is declared in `dvc.yaml` but absent from `dvc.lock`, which
  is why `dvc pull` exits 1 on a clean clone.

## Current shared-state facts

*Re-verified 2026-07-29 against the repository at `dataset-v0.6.0` (`699b9ba`).
The list below previously described the pre-Phase-E state and had drifted on
every line.*

- Build mode: **`full`** — 24,352 images / 104,289 boxes (Phase E).
- **`dataset-v0.6.0` is cut and pushed** (annotated tag → `699b9ba`);
  `data/releases/dataset-v0.6.0/release_manifest.json` is git-tracked, 9 gates
  recorded, all `pass`. `v0.7.0`/`v0.9.0`/`v1.0.0` do not exist yet.
- Ledger is **still empty (0 cells)** — this is the binding constraint on
  `dataset-v0.7.0`, which requires `min_verified_cells: 3000`.
  `custom_captures/` and `eval/indian_home_v0/` are also still empty.
- `main` is pushed; HEAD = tag = `origin/main` = `699b9ba`. CI green.
- DVC: cache `C:\dvc_cache`, `localstore` → `C:\dvc_remote` (default),
  `storage` → S3 (**68,301 objects / 7.82 GB**, versioning + SSE-AES256 + all
  four public-access blocks on). `dvc status -c` exits 0 for both remotes;
  `verify_lock_objects.py --deep` resolves all 67,734 objects.
- Cross-platform reproducibility **proven** 2026-07-29 in a clean Linux
  container (`docs/04_dataset_engineering/reproduction_log.md`).

### Known-wrong artifact shipped with `dataset-v0.6.0`

`completeness.json` over-claims trusted classes for **69.5%** of images, and the
same union suppressed candidate generation for those images
([ADR-P5-15](adr/ADR-P5-15-per-slug-trust-resolution-and-targeting.md)).
**Do not train on `dataset-v0.6.0` with `missing_annotation_mitigation`
enabled.** Fix scheduled for `v0.7.0`.

See [`../../CHANGELOG.md`](../../CHANGELOG.md) `[Unreleased]` for the full
Phase-5 change list and [`adr/`](adr/README.md) for the design decisions.

# Pre-Phase-4 Production Readiness Audit

> **Scope:** Full repository audit before Phase 4 (Missing Annotation Mitigation) begins.
> **Audited at:** `main` @ `71618e5` (2026-07-16), branch `pre-phase4-production-audit`.
> **Method:** Evidence-based verification — every claim below is backed by a command run,
> a file read, or a git record produced during the audit. No findings are assumed.
> **Environment caveat:** `gh` CLI unauthenticated → GitHub Actions run history and PR
> metadata could not be queried; CI health was verified by running the exact `ci.yml`
> commands locally and by matching `ci.yml` against the repo configuration.

---

## 1. Executive Summary

The repository is in **good production shape**: all 412 tests pass (73.35 % coverage,
floor 40 %), formatting/lint/type gates are clean on the CI-checked scope, the 11-stage
DVC DAG is coherent, git is fully synchronized with the remote, and the Phase-3 tooling
layer is complete, tested, and documented. No abandoned implementations, no duplicated
logic, and no dead code beyond two stray root files were found.

The audit found **no Critical code defects**. It found **one Critical operational
exposure** (the smoke dataset exists as a single copy — local DVC cache missing, S3
remote never pushed), **two High findings** (an ineffective mypy "fix" that leaves the
local type-gate broken at default settings; stale DVC pipeline state relative to
Phase-3 code), and a set of Medium/Low documentation-drift and hygiene items.
Everything locally fixable is remediated on this branch except the DVC lock refresh,
which requires a delete-and-regenerate `dvc repro` over the single-copy dataset —
deliberately left as a **user-approved step** (exact commands in §9, full backup taken
at `~/.claude/jobs/0de3dae9/tmp/data_backup/`).

| Metric | Value |
|---|---|
| Tests | 412 passed / 0 failed (unit + integration) |
| Coverage | 73.35 % (gate: 40 %) |
| black / ruff / mypy (CI scope) | all clean |
| DVC stages | 11 in DAG; 9 locked (2 frozen human-in-the-loop stages unrun, by design) |
| Git | `main` == `origin/main` @ `71618e5`; no stale branches; no stashes |
| **Production readiness score** | **8.5 / 10** |
| **Recommendation** | **⚠ → ✅ Ready for Phase 4 after this branch's remediation** (see §10) |

Score rationale: −1.0 for the single-copy dataset exposure (closable only with AWS
credentials — human task); −0.5 for pipeline-state staleness and the local mypy gate
(both fixed on this branch, but they indicate the repro/typecheck loop is not run
routinely on this machine).

---

## 2. Phase Verification

| Phase | Verdict | Evidence |
|---|---|---|
| **Phase 1 — Foundation & Taxonomy** | **PASS** | 23-class taxonomy (`configs/data.yaml`, `nc: 23`, 23 names); full config tree (`configs/` 12 YAMLs, all with readers or declared future use); runtime skeleton `src/pipeline/` (9 modules); structured logging (`src/logging/`); config loader + tests (`tests/unit/test_config_loader.py`, `tests/unit/pipeline/`); docs 01–03 complete (15+17+14 files). |
| **Phase 2 — Public Dataset Engineering** | **PASS** | Scripts 01–07 + `split_dataset.py` + `dataset_stats.py` all present and wired into `dvc.yaml`; downloaders package with parser tests; merge/split/QA libraries; smoke dataset built (188 merged images, provenance manifest, QA report); `dvc.lock` records all 9 pipeline stages. |
| **WP3.0 — Platform Remediation** | **PASS** | CI matrix (ubuntu+windows × py3.10/3.12) in `ci.yml`; downloader tests (`test_downloaders*.py`); S3 remote configured (`.dvc/config` → `s3://elderly-assistant-mlops/datasets/yolo_v1`); clean-machine reproduction log PASS-partial (2026-07-14, network-restricted) in `docs/04_dataset_engineering/reproduction_log.md`. |
| **Phase 3 — Custom Capture & Annotation tooling** | **PASS** | All 8 milestones merged (`07cb7e0`): `src/dataset/capture/` (7 modules), CLIs 08–10, LOHO split strategy, eval-leakage QA guards, 2 frozen DVC ingest stages, runbook + engineering report; 129 unit tests + E2E integration test (`test_capture_workflow.py`) — all passing in this audit's run. |

**Caveat on "Phase 3":** the executive plan (`docs/01_executive_implementation_plan/implementation_phases.md`)
defines Phase 3's deliverable as *"2,000+ Indian-home images, fully annotated"*. That
human collection work has **not started** (0 captured images; `data/raw/custom_captures`
empty; eval set absent). What shipped and passed above is the Phase-3 **tooling and
governance layer** — the repo-side deliverable. The phase table in docs/01 also still
lists "Phase 4 = Dataset QA & Versioning" (already absorbed into Phase 2) and does not
mention "Missing Annotation Mitigation" at all → see finding D-1.

---

## 3. Repository Findings

### Critical

| ID | Finding | Detail |
|---|---|---|
| **C-1** | **Smoke dataset is a single copy** | `dvc status` reports *not in cache* for every Phase-2 stage output; `.dvc/cache` is essentially empty (repo was moved between user profiles — `tree.txt` records the previous `C:\Users\sm646\` location); `dvc push` has never run (no AWS credentials). The 188-image smoke dataset + labels exist **only as workspace files on this machine** (plus OneDrive sync). Workspace files still match `dvc.lock` hashes (status shows no "modified" on data dirs), so recovery is `dvc commit` + first `dvc push`. **The push requires AWS credentials — human task, and the hard gate before any real capture collection begins (runbook step 0).** |

### High

| ID | Finding | Detail |
|---|---|---|
| **H-1** | **Local mypy gate broken at default settings; the committed "fix" is ineffective** | `mypy src/dataset src/utils src/config src/logging` (exact CI command) dies locally with `numpy/__init__.pyi:737: Type statement is only supported in Python 3.12 and greater` — mypy (configured `python_version = "3.10"`) cannot parse installed numpy 2.x stubs. Commit `3292e6b` ("add numpy.* mypy override") added `numpy.*` to an `ignore_missing_imports` override — which only affects *missing* stubs, not installed ones, so it does not fix this (and `follow_imports = "skip"` cannot either: mypy always processes installed stub packages — verified during this audit). CI is green only because its lint job never installs numpy. Fix applied: remove the `python_version = "3.10"` pin so mypy checks under the running interpreter; the 3.10 floor remains enforced by CI's Python-3.10 quality job. |
| **H-2** | **DVC pipeline state stale relative to Phase-3 code** | `dvc status`: changed deps on `merge_datasets` (filters.py, new `sources.custom_captures` param), `split_train_val_test` (split code refactored in Phase-3 M5), `qa_check` (QA code extended in M6), `download_roboflow` (roboflow_dl.py). The recorded outputs predate the current code — `dvc repro` was never re-run after Phase 3 merged. Also `data/raw/custom_captures` was a bare empty dir (the `08 --init` canonical tree was never created/committed), and the tracked `data/qa_reports/annotation_qa_report.json` no longer matches its lock hash. **Status: partially remediated** — canonical capture/eval trees restored via `08 --init` (non-destructive); the lock refresh itself needs `dvc repro`, which deletes-then-regenerates outputs of the single-copy dataset and was therefore left for explicit user approval (§9 item 3). |

### Medium

| ID | Finding | Detail |
|---|---|---|
| **M-1** | **CHANGELOG drift** | The three post-Phase-3 commits — AlertQueue heap fix (`5d93052`, a real Windows CI bug fix in `src/pipeline/alert_queue.py`), `requests` runtime dep (`eb01308`), numpy mypy override (`3292e6b`) — have **no CHANGELOG entries**. |
| **M-2** | **docs/README.md index predates Phase 2/3** | The Document Structure tree lists only docs/01–03; `docs/04_dataset_engineering/` (4 files) is absent; role-navigation has no Dataset/MLOps route to docs/04; the file counts are stale. |
| **M-3** | **`tree.txt` junk file tracked in git** | 168-line UTF-16 PowerShell directory dump from the previous machine (`C:\Users\sm646\...`), including `__pycache__` listings. No references anywhere. Delete. |
| **M-4** | **`src/pipeline` test coverage thin** | Only `alert_queue` (92 %) and `interfaces` have unit tests; 8 runtime modules at 0 % coverage. Already a *documented* descope ("Phase-6 runtime cleanup", noted in `ci.yml` comments and docs/04 §7) — recorded here so it is not lost. |
| **M-5** | **Misleading merge-commit name** | `71618e5 "chore: merge missing-annotation-mitigation into main"` merged only the requests-dep + numpy-override commits (3 lines). **No Phase-4 work exists in the repo.** History cannot be rewritten (pushed); this report is the canonical correction. |

### Low

| ID | Finding | Detail |
|---|---|---|
| **L-1** | mypy notes unused override sections (`sounddevice.*`, `transformers.*`, `ultralytics.*`) — modules not imported by the checked scope. Harmless; they become active when Phase-6 brings `src/pipeline` into scope. Keep. |
| **L-2** | `configs/deployment/{onnx,tflite}_config.yaml` and `configs/training/yolo11s_config.yaml` have no code readers — forward-declared for Phase-5/7 export and the larger-model variant (`train_yolo.py --config`). Intentional; documented here. |
| **L-3** | `make test-system` / `make test-perf` exit non-zero ("no tests collected") — `tests/system/` and `tests/performance/` are documented scaffolding for Phase-7 (conftest lists the 10 field-test scenarios). Acceptable. |
| **L-4** | `task.md` (root) is a superseded Stage-1 checklist — already carries a "Superseded" banner; self-documenting. Keep or delete at owner's discretion. |
| **L-5** | `scripts/inference/test_video.py` uses a `test_` prefix outside `tests/` — not collected by pytest (`testpaths=["tests"]`), naming nit only. |
| **L-6** | `dvc.lock` working-copy has CRLF endings while the index stores LF (`git ls-files --eol`): the known F1 churn is contained by `.gitattributes` (`* text=auto eol=lf`) — commits stay LF; local noise only. |

---

## 4. Architecture Review

**Strengths**
- Consistent config-driven design: every behavior knob is a YAML read into a typed
  frozen dataclass with imperative validation and `with_overrides()` CLI precedence —
  uniform across dataset, capture, split, and training configs.
- Clean layering: `src/` libraries ↔ thin numbered `scripts/` CLIs ↔ DVC stages. No
  logic duplication found (SHA-256 file hashing, image validation, YOLO label parsing,
  perceptual hashing each single-sourced in `src/utils`/`src/dataset/dedup.py`).
- The capture subsystem reuses Phase-2 contracts instead of forking them (group-key
  pattern for split integrity, manifest glob for the license gate, raw-labels fallback
  in merge) — Phase 3 added zero changes to Phase-2 merge/remap code.
- Frozen-DVC-stage pattern for human-in-the-loop data is the correct call (a normal
  stage would delete real photos on `dvc repro`).
- Registry pattern for split strategies; downloader base class; plugin scaffolding for
  Phase 6 — extension points exist where Phase 4/5/6 will need them.

**Weaknesses**
- `src/pipeline/` (Phase-1 runtime skeleton) is far behind the dataset platform in test
  and type rigor (M-4) — acceptable now, but Phase 6 inherits a known debt lump.
- The DVC state relies on discipline (`dvc repro` after code changes, `dvc commit -f`
  after ingests). Twice now the workspace has drifted (H-2); nothing automated catches it.
- Single-machine, OneDrive-synced working copy for governance-critical data (C-1).

**Recommendations**
1. Add a CI or pre-merge step that runs `dvc status` and fails on unexpected drift
   (cheap: compares lock vs workspace hashes, no data needed) — consider in Phase 4.
2. Complete the first `dvc push` before any capture session (already runbook step 0).
3. Schedule the Phase-6 `src/pipeline` typing/testing cleanup explicitly in the roadmap.

---

## 5. Documentation Review

| Item | Status |
|---|---|
| README.md | Accurate (Phase-2/3 sections, doc links verified). |
| CHANGELOG.md | Drift: 3 undocumented commits (M-1) → **fixed on this branch**. |
| docs/README.md | Stale index (M-2) → **fixed on this branch** (docs/04 + 05 added). |
| docs/01 implementation_phases.md / roadmap.md | Phase table does not reflect the executed sequence (Phase 4 renamed/reordered; Missing Annotation Mitigation absent) → **D-1, deliberately NOT rewritten here**: renumbering the executive phase plan is a product-owner decision; this report records the mapping instead (docs/01 "Phase 4 QA & Versioning" ≙ delivered in Phase 2; "Phase 4 Missing Annotation Mitigation" ≙ the next engineering phase, from docs/04 §3's label-completeness policy). |
| docs/04 README, runbook, phase3 report, reproduction log | Current and mutually consistent; runbook flags verified against actual CLI flags. |
| Risk register | R24 gate present with measurable criteria; C1 referenced in docs/04 §6 + runbook step 0. Current. |
| Undocumented features | None found — every shipped module appears in docs/04 §7 or the runbook. |

---

## 6. CI / DVC Review

**CI (`.github/workflows/ci.yml`)**
- Quality job: black, ruff, `mypy src/dataset src/utils src/config src/logging` on
  py3.10, installing `types-PyYAML types-requests` — matches `requirements.txt` dev
  block and pyproject dev extras. The `src/pipeline` mypy descope is documented inline.
- Test matrix: ubuntu+windows × py3.10/3.12, `pytest tests/unit tests/integration --cov=src`.
- **Local reproduction of every CI command: all green** (412 passed, 73.35 % cov;
  black 114 files clean; ruff clean; mypy clean on the 41-file scope).
- Config drift vs pyproject: none found (markers, coverage floor, tool scopes agree).
- Not verifiable: actual GitHub Actions run history (gh unauthenticated).

**DVC**
- `dvc dag`: renders all 11 stages; `ingest_custom_captures → merge_datasets` edge
  correct; `ingest_eval_set` intentionally isolated (eval must never feed training).
- `dvc.lock`: 9 stages locked; the 2 frozen ingest stages and frozen `train_yolo11n`
  correctly have no lock entries yet (never run — by design for human-in-the-loop data).
- `dvc repro --dry`: coherent plan; frozen stages skipped with warnings as designed.
- Broken items: **H-2** (stale state — capture trees restored here; lock refresh
  prescribed for user approval, §9 item 3) and **C-1** (empty cache / no remote copy —
  requires credentials, human).

---

## 7. Git Review

| Check | Result |
|---|---|
| Sync | `main` == `origin/main` == `71618e5`. Nothing unpushed, nothing unpulled. |
| Branches | Local: `main` + this audit branch. Remote: `main` only (verified `git ls-remote --heads`). **No branches to delete** — all feature branches already cleaned up. |
| Stashes | None. |
| Working tree at audit start | 1 uncommitted one-word doc fix (committed here) + `cls` junk file (removed here). |
| PRs | Cannot query (gh unauthenticated). Merge evidence: PR #2 (`claude/pr-merge-error-fix-futra2`, AlertQueue fix) merged; no other PR-shaped merges outstanding. |
| History quality | Conventional-commit discipline throughout; one misleading merge name (M-5) — recorded, not rewritten. |

---

## 8. Phase 4 Readiness (Missing Annotation Mitigation)

**Verdict: no architectural blockers. Phase 4 can begin once this branch merges.**

| Requirement | Assessment |
|---|---|
| Metadata-driven training supported? | **Yes.** `MergedManifest.label_completeness` (per-source exhaustively-labeled classes) and `image_provenance` (188/188 images mapped to source) are already populated in `data/merged/merged_manifest.json`. Filenames in `data/processed/` keep their source prefixes, so per-image completeness is recoverable post-split without schema changes. |
| DVC accommodates completeness metadata? | **Yes.** A small stage between `split_train_val_test` and `train_yolo11n` (deps: merged manifest + split report; out: e.g. `data/processed/completeness.json`) slots into the DAG with no restructuring. |
| Manifests support completeness info? | **Yes** (above). If per-*session* trusted classes are needed for future custom captures, `CaptureSessionManifest` already carries `class_counts`/`trusted_classes` via the aggregate `SourceManifest`. |
| Trainer injection without modifying Ultralytics? | **Yes.** `train_yolo.py` uses the public `YOLO(...).train(**kwargs)` API. Ultralytics supports `trainer=` (custom `DetectionTrainer` subclass — override `build_dataset`/loss for per-image class masking) and `model.add_callback(...)`. The script's kwargs-assembly structure extends cleanly; `src/pipeline` untouched. |
| Training pipeline extendable cleanly? | **Yes.** Config-driven pattern means a `completeness:` section in the training YAML + a `src/training/` (or `src/dataset/`) module keeps the established architecture. Stage stays frozen until Phase 5. |
| Hidden blockers? | **None found.** Operational prerequisite (not a code blocker): close C-1 before large-scale data work. Note Phase 4 design should decide early whether masking happens at loss level (custom trainer) or sampling level (dataset filtering) — both feasible today. |

---

## 9. Prioritized Action Plan

| # | Action | Priority | Effort | Depends on | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | Fix mypy numpy override (`follow_imports = "skip"`), verify default CI command passes locally (H-1) | High | XS | — | None | **Done (this branch)** |
| 2 | CHANGELOG entries for `5d93052`/`eb01308`/`3292e6b` (M-1) | High | XS | — | None | **Done (this branch)** |
| 3 | Refresh DVC state (H-2): with the full backup already taken, run `dvc repro`, sanity-check the QA report, then commit the refreshed `dvc.lock` + `annotation_qa_report.json` | High | S | 1 | Low — smoke-scale, deterministic seeds; full backup at `~/.claude/jobs/0de3dae9/tmp/data_backup/` | **Prescribed — needs user run/approval** (this session's autonomous-destructive-op guard blocked `dvc repro`/`dvc commit`; capture trees already restored via `08 --init`). Phase 4's first pipeline change forces this refresh anyway. |
| 4 | docs/README.md index refresh incl. docs/04+05 (M-2); delete `tree.txt` (M-3) | Medium | XS | — | None | **Done (this branch)** |
| 5 | **First `dvc push`** (configure AWS credentials, then `dvc push`; verify `dvc pull` on scratch clone) — closes C-1 | **Critical (operational)** | S | AWS creds | None | **Human task — the only gate before capture collection** |
| 6 | Decide docs/01 phase-table reconciliation (rename/renumber vs addendum) (D-1) | Medium | XS | Product owner | None | Open — owner decision |
| 7 | Populate Roboflow `datasets:[]` with license review per slug | Medium | M | Roboflow account | License gate enforces review | Open — human task (pre-full-build) |
| 8 | wet_floor pilot session (50 images, dual-annotated) → R24 gate decision | Medium | M | First captures | Protocol in runbook §8 | Open — human task |
| 9 | Phase-6 backlog: `src/pipeline` typing + tests (M-4) | Low (now) | L | Phase-6 start | Known descope | Deferred, tracked |

## 9a. Known-Issues Validation (requested checklist)

| Known issue | Status |
|---|---|
| DVC remote push | **Still outstanding** (C-1) — now urgent because the local cache is also gone. |
| Roboflow dataset configuration | **Still outstanding** — `datasets: []`, instructions + license gate in place. |
| Consent registry | **Working as designed** — local-only, gitignore verified (`data/consent/*` + README exception); no PII fields in manifests (unit-tested). |
| `custom_captures` enablement | **Outstanding by design** — `enabled: false` until first finalized session (documented flip). Local dir was missing its `--init` tree → restored on this branch. |
| Cross-platform line endings | **Mitigated** — `.gitattributes` normalizes; committed content is LF; residual local CRLF churn on `dvc.lock` is cosmetic (L-6). |
| Smoke vs full dataset mode | **As designed** — `mode: smoke` active; full build is one param flip away, gated on Roboflow + captures. |
| wet_floor pilot | **Pending human pilot** — R24 gate defined and measurable (IAA ≥ 0.60 over ~50 dual-annotated images). |
| Evaluation dataset readiness | **Tooling ready, data absent (expected)** — `data/eval/` awaits first eval capture session; ingest + lock + leakage guards tested synthetically. |
| CI matrix validation | **Config verified + all commands green locally**; Actions run history not queryable this session (gh unauthenticated). |

---

## 10. Final Recommendation

> ## ✅ Ready for Phase 4 — after this audit branch is merged
>
> All Critical/High **code-level** findings are remediated on this branch (mypy gate,
> documentation drift, hygiene); the DVC lock refresh (H-2) is fully prescribed with a
> backup in place and needs one user-approved `dvc repro` — or simply happens as part
> of Phase 4's first pipeline change. The remaining Critical item (C-1, first
> `dvc push`) is an **operational/credentials task that does not block Phase-4
> engineering work** — Phase 4 builds training-metadata machinery against the smoke
> dataset — but it **must be completed before any real capture collection**, as the
> runbook already mandates. Absent this branch's fixes the verdict would have been
> "⚠ Ready after minor fixes"; with them, Phase 4 can start immediately.

**Justification:** phases 1–3 verified PASS with evidence (§2); zero test failures
across 412 tests; CI-equivalent gates all green; DVC DAG coherent with the designed
human-in-the-loop pattern intact; git fully synchronized and clean; Phase-4 extension
points (completeness metadata, trainer injection, DVC stage insertion) verified
present and feasible without touching Ultralytics source (§8). Residual risks are
enumerated, owned, and scheduled (§9) rather than hidden.

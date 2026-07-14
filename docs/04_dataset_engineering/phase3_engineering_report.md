# Phase-3 Engineering Report — Custom Dataset Collection & Annotation

> Final audit for the Phase-3 tooling build. Status: **tooling complete,
> collection not started** — see §9/§10 for what remains and why.

---

## 1. Objectives

Build a production-grade custom data collection and annotation workflow for
the 8 classes with no adequate public-dataset coverage
(`gas_cylinder`, `medicine_strip`, `wet_floor`, `walking_stick`,
`support_handle`, Indian `stove`/`cupboard`, `passport`), reusing the
Phase-2 dataset engineering platform wherever possible, so the repository
becomes production-ready to collect, annotate, validate, version, and
integrate custom Indian-home data toward Dataset v1.0.0.

## 2. Implemented features

| Area | What shipped | Where |
|---|---|---|
| Configuration | Capture workflow config (paths, session grammar, image intake, consent, IAA thresholds, governance targets) | `configs/capture_config.yaml`, `src/dataset/capture/config.py` |
| Privacy | Mandatory EXIF/GPS stripping at ingest, verified clean post-strip | `src/dataset/capture/exif.py` |
| Consent | PII-free local registry verification (format, resolution, withdrawal, house match) | `src/dataset/capture/consent.py`, `data/consent/README.md` |
| Ingest | Inbox → validated session: corruption/size/duplicate gates, session manifests, aggregate manifest | `src/dataset/capture/ingest.py`, `scripts/dataset/08_ingest_capture_session.py` |
| Annotation import | CVAT-compatible YOLO import, class-order verification, session-scoped validation, staging, finalize | `src/dataset/capture/annotations.py`, `scripts/dataset/09_import_annotations.py` |
| Dual-annotator agreement | Greedy IoU matching, per-class agreement gates (`wet_floor` R24 override) | `src/dataset/capture/agreement.py` |
| Progress tracking | Per-class/house/room/lighting targets vs governance minimums | `src/dataset/capture/progress.py`, `scripts/dataset/10_capture_progress.py` |
| House-level splitting | `leave_one_house_out` strategy (was a reserved stub) | `src/dataset/splitting/leave_one_house_out.py` |
| Eval-set integrity | Lock mechanism + leakage guards | `src/dataset/capture/ingest.py` (`lock_eval_set`), `scripts/qa/run_full_qa.py` (`check_eval_overlap`, `check_house_exclusivity`) |
| DVC integration | Frozen ingest stages, merge dependency wiring | `dvc.yaml` |
| Documentation | Operational runbook, consent template, updated governance docs | `docs/04_dataset_engineering/capture_annotation_runbook.md`, `docs/03_engineering_appendix/consent_form_template.md` |

## 3. Dataset collection workflow

Consent (local registry) → capture (per docs/03 §10.3 capture-variations
checklist, tightened `wet_floor` protocol with paired dry negatives) →
`08_ingest_capture_session.py` (validate, strip EXIF/GPS, name, manifest) →
`09_import_annotations.py --stage/--compare/--finalize` (CVAT import,
class-order check, IAA, promote labels) → `10_capture_progress.py`
(targets) → `dvc commit -f ingest_custom_captures` → `dvc push`. Full SOP:
`capture_annotation_runbook.md`.

## 4. Annotation workflow

Tool-agnostic ingestion built around CVAT's "YOLO 1.1" export (zip or
directory; any tool producing the same shape works). The single most
dangerous CVAT failure mode — a task built from a subset or reordered
label list, which shifts every class ID with no other observable
signal — is caught deterministically by `verify_class_order()` against
`configs/data.yaml`, CRITICAL, before anything is staged. Dual annotation
is measured, not assumed: `compare_annotators()` computes per-class
agreement via greedy IoU matching, gated per class
(`annotation.iaa.min_agreement`, with a lower `wet_floor` override wired
to the R24 decision — see §8).

## 5. QA process

Existing Phase-2 QA (`check_annotations.py` structural checks,
`dataset_stats.py`, license gate, label completeness, blur/low-light) is
extended, not duplicated:

- `check_eval_overlap` — exact SHA-256 **and** flip-robust perceptual
  near-duplicate check between the locked eval set and all train-facing
  data. CRITICAL on any hit. Eval images are compared only against the
  frozen train-facing set, never against each other, so eval-internal
  duplicates are never mistaken for leakage (verified by a dedicated unit
  test).
- `check_house_exclusivity` — WARNING when a house contributes to both
  training and eval.
- Both are opportunistic: `{"available": false}` before any Phase-3 data
  exists, so `qa_check` stays green on every machine/branch without
  regressing Phase-2 behavior (verified against the real smoke dataset).

## 6. DVC integration

Human-produced data enters `dvc.lock` via two **frozen** stages
(`ingest_custom_captures`, `ingest_eval_set`) rather than a normal stage
or a standalone `dvc add` pointer — both alternatives were evaluated and
rejected (rationale in `docs/04_dataset_engineering/README.md` §6). `dvc
repro` never touches them; the human loop is ingest → `dvc commit -f` →
`git commit dvc.lock` → `dvc push`. `merge_datasets` gained a dependency
on `data/raw/custom_captures` and the `sources.custom_captures` param so
content/config changes correctly invalidate the merge. Verified: `dvc
dag` renders both stages correctly (`ingest_custom_captures` feeds
`merge_datasets`; `ingest_eval_set` is intentionally disconnected); `dvc
repro --dry qa_check` confirms frozen stages are skipped and the rest of
the chain (downloads → remap → merge → split → QA) is unaffected.

## 7. Testing results

- **129 new unit tests** (offline, `tmp_path`, synthetic Pillow images,
  zero network) across 7 new + 4 extended test files — covering config
  loading/validation, consent verification, EXIF strip/inspect (incl.
  orientation baking), ingest happy-path/rejects/append-safety/eval-lock/
  tamper-detection, annotation import/class-order/coverage/duplicate-line
  validation, IAA math (IoU edge cases, shifted/missing/class-swapped
  boxes, wet_floor gate override), progress aggregation, and the LOHO
  split strategy (house integrity, holdout, solo-group parity with
  `group_aware`, determinism).
- **1 end-to-end integration test**
  (`tests/integration/test_capture_workflow.py`) simulating the entire
  human workflow on synthetic data: inbox (incl. a GPS-EXIF photo, a
  duplicate, an undersized reject) → ingest → two-annotator CVAT-style
  export → stage → IAA compare (pass) → finalize → merge alongside a
  synthetic public source (custom priority verified) → `group_aware` +
  `leave_one_house_out` splits (session/house integrity asserted for
  both) → the real `check_annotations.py` CLI (0 criticals) → progress
  report → eval-set ingest with a deliberate train-duplicate →
  `check_eval_overlap` flags it → fixed → re-verified clean → locked →
  re-ingest refused.
- **Full suite**: 412 passed (unit + integration), 0 failures.
- **Quality gates**: `black --check`, `ruff check`, `mypy` (strict scope
  `src/dataset src/utils src/config src/logging`) all clean on every
  milestone.
- Not run in this phase: the Windows+Ubuntu × py3.10/3.12 CI matrix
  (requires a pushed branch/PR) — all commands were verified locally on
  Windows/py3.14; no platform-specific code was introduced (pathlib
  throughout, no OS-conditional branches).

## 8. wet_floor (R24) status

Gate defined and wired end-to-end (config threshold, IAA computation,
verdict logic, documented protocol); **not yet evaluated** — no pilot
session has been captured. `docs/01_executive_implementation_plan/risk_register.md`
records R24 with the gate and an explicit "pending" status.

## 9. Documentation updates

New: `capture_annotation_runbook.md` (10-section operational SOP),
`consent_form_template.md`, `data/consent/README.md`, this report.
Updated: `docs/04_dataset_engineering/README.md` (§1 diagram, §5 LOHO, §6
DVC rationale, §7 tooling-shipped reframe, new §8 eval guards + R24),
`docs/03_engineering_appendix/annotation_guide.md` (§10.1 CVAT/passport
rules, §10.2 tightened wet_floor guidance, new §10.6 CVAT quick
reference), `risk_register.md` (R24 row), root `README.md` (Phase-3
section + docs links), `CHANGELOG.md` (Unreleased → Added).

## 10. Remaining risks

- **First `dvc push` not yet done** (risk C1, pre-existing from Phase-2) —
  hard gate before any real collection starts; documented as runbook
  step 0.
- **Zero real capture sessions exist** — every code path is verified on
  synthetic data only; the first few real sessions should be treated as
  a soak test of the tooling (EXIF stripping on real phone JPEGs from
  varied manufacturers, real CVAT export quirks).
- **Roboflow slugs unpopulated** (`sources.roboflow.datasets: []`) —
  blocks 4 classes from public coverage independent of Phase-3; checklist
  in runbook §9.
- **R24 undecided** — see §8.
- **CI matrix not exercised for Phase-3 code** — local validation only;
  first PR/branch push will be the real cross-platform check.

## 11. Technical debt

None identified that blocks Phase-4/5. Noted but intentionally deferred
(not debt, explicit design choices with documented rationale):
`ingest_eval_set` is not a `qa_check` dependency (§5); greedy (not
Hungarian) IoU matching for IAA (documented approximation, adequate for a
QA signal); JPEG re-encode on EXIF strip is not bit-identical
(provenance hashes are computed post-strip, so this is correct by
construction, not a gap).

## 12. Readiness for Phase 4/5

Phase 4 (dataset QA & versioning) is largely already delivered by the
Phase-2 platform plus this phase's eval-set guards — once real data
exists, releasing Dataset v1.0.0 is the runbook §10 checklist, not new
engineering. Phase 5 (training) remains correctly gated: `train_yolo11n`
stays `frozen: true` in `dvc.yaml` until a dataset release unfreezes it.
No Phase-2 regressions: existing tests pass unchanged, and `qa_check`
behavior is byte-for-byte identical when Phase-3 directories are absent
(verified against the real smoke dataset).

---

Previous: [capture_annotation_runbook.md](./capture_annotation_runbook.md)

Related: [README.md](./README.md), [reproduction_log.md](./reproduction_log.md)

# Phase-5 M2 Verification Runbook — CVAT Round-Trip

> Operational SOP for the human verification loop that turns auto-annotation
> candidates into trusted labels. Tooling reference:
> `docs/07_dataset_production/README.md`. Architecture/design: ADR-P5-03
> (CVAT round-trip), ADR-P5-04 (verification ledger).

---

## 0. Pre-flight: self-hosted CVAT

**Custom-capture images must never leave the machine (consent posture) —
CVAT must be self-hosted (Docker), not the public cvat.ai instance.**

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
```

Record the CVAT version actually running (`docker compose exec cvat_server
python manage.py --version` or the footer of the web UI) here once deployed,
so a future re-deploy pins the same version:

> **CVAT version pinned for Phase-5:** _(record at first deployment)_

Public-image batches (M2 drill, M7+ scale-up) may use the same self-hosted
instance — there is no reason to run two CVAT deployments, and doing so
would only fragment task history.

---

## 1. Generate candidates

```bash
python scripts/dataset/12_auto_annotate.py --verify-determinism
```

See `auto_annotation_runbook.md` for weight pinning and prompt tuning. This
step is a normal (non-frozen) DVC stage — `dvc repro auto_annotate` also
works and is the reproducible form once weights are pinned.

---

## 2. Build verification batches

```bash
python scripts/dataset/13_build_verification_batches.py
dvc commit -f build_verification_batches
```

For each backend with a candidates artifact, writes one
`data/annotation/batches/vb{NNN}_<backend>/` per batch (images ranked by
expected verification gain, chunked to `verification.batch_size`, default
200) containing:

- `batch_manifest.json` — lifecycle `created` → `exported` → `staged` →
  `verified` → `imported`; pins `candidate_run.candidates_sha256` (a later
  candidate regeneration never silently changes an in-flight batch);
  records `target_classes`, `iaa_sample` (this batch's dual-annotation
  images, `verification.iaa_sample_fraction` — default 10% — picked
  deterministically at creation time).
- `preannotations.zip` — "YOLO 1.1" pre-annotation package. `obj.names` is
  always the **full ordered 23-class taxonomy**, never a batch-scoped
  subset. Per-image label files are the base merged label **UNION** this
  batch's candidate detections for its target classes.

A shared `data/annotation/batches/cvat_labels.json` is (re)written every run
— the exact taxonomy-order label-constructor spec for CVAT task creation
(§3.2). `build_verification_batches` is a **frozen** DVC stage (mirrors
Phase-3's frozen ingest stages) — regenerating batches mid-flight would
orphan live CVAT tasks whose pre-annotations no longer match a fresh plan.

**In-flight protection:** an image already claimed by a `created`/
`exported`/`staged` batch is never re-batched. `imported` releases the
claim (a later batch may legitimately target a different, still-unverified
class on the same image).

---

## 3. Create the CVAT task

1. **New task → Labels → Raw**, paste the contents of
   `data/annotation/batches/cvat_labels.json` (already in exact taxonomy
   order — this is the step that kills the manual label-order mistake
   class before it can happen).
2. Upload the batch's images.
3. **Upload annotations** → format "YOLO 1.1" → the batch's
   `preannotations.zip`. Verify the task shows both the trusted (base) boxes
   and the new candidate boxes for the target classes.
4. Record the CVAT task id/URL — it lands in the batch manifest's
   `cvat_task_ref` at import time (§5), not created automatically here.
5. Assign reviewers. If `batch_manifest.json`'s `iaa_sample` is non-empty,
   assign it to **two independent reviewers** (dual-annotation, §4/§6);
   everything else in the batch needs only one.

---

## 4. Review in CVAT

Reviewers correct/confirm boxes for the batch's `target_classes` **only**.
Boxes for other (already-trusted) classes are visible for context but must
not be edited — `14_import_verified_batch.py` hard-fails the whole batch on
any byte-level change to a non-target-class line (§5, R31). If a
`passport` or other privacy-sensitive box appears incidentally in frame,
follow `annotation_guide.md` §10.2 (blur before export).

Export each review as **YOLO 1.1** (Export dataset → format `YOLO 1.1`).

---

## 5. Import

```bash
python scripts/dataset/14_import_verified_batch.py \
    --batch vb001_yolo_world \
    --export exports/vb001_review.zip \
    --verifier anno_1
```

If the batch has a non-empty `iaa_sample`, a second reviewer's independent
export is **required**:

```bash
python scripts/dataset/14_import_verified_batch.py \
    --batch vb001_yolo_world \
    --export exports/vb001_review.zip \
    --dual-export exports/vb001_review_2.zip \
    --verifier anno_1
```

What happens, in order:

1. **Class-order check** (reused from Phase-3's `verify_class_order`) —
   aborts immediately if the export's label list doesn't match
   `configs/data.yaml` ID-for-ID.
2. **IAA gate** (only when `iaa_sample` is non-empty): agreement over the
   sample's target-class boxes, via the same dual-annotator instrument
   Phase-3 capture sessions use. Below `verification.min_agreement`
   (default 0.70 — lower than capture's 0.75 because verifying pre-labels
   is an easier task than de-novo labeling) the batch is set back to
   `staged` with the measured `iaa_agreement` recorded, **nothing is
   imported**, and exit code is 1. Adjudicate in CVAT and re-export.
3. **Non-target byte-equality check** — every non-target-class line must
   match the base merged label exactly (order-independent). Any edit
   aborts the whole batch (all-or-nothing; nothing partial is ever
   committed) — revert the accidental edit and re-export.
4. **Verdict recording** — one `present_labeled` (boxes present) or
   `verified_absent` (reviewer confirmed the class truly isn't there) verdict
   per (image, target class) into `data/annotation/verification_ledger.json`.
5. **Delta labels** — `data/annotation/verified_labels/<stem>.txt`, target-
   class boxes only, for images with at least one delta box.
6. Batch → `imported`.

Re-running the exact same export is **idempotent** (safe to retry after a
transient failure elsewhere in the pipeline).

**Re-verification / disagreement:** the ledger hard-fails a conflicting
verdict for an already-settled (image, class) cell unless the import names
`--supersedes <prior-batch-id>`. A same-class re-verification therefore
always happens through a **new batch** targeting that cell again, never by
re-importing the old one.

```bash
dvc commit -f import_verified_annotations
git add dvc.lock data/annotation/verification_ledger.json
git commit -m "data: import vb001_yolo_world (charger, wire)"
dvc push
```

The ledger is `cache: false` (small, git-tracked, append-only audit trail —
like the QA reports); `verified_labels` is DVC-cached (grows with scale).

---

## 6. IAA sampling in practice

`verification.iaa_sample_fraction` (default 0.10) is applied **per batch**,
not per session — `select_iaa_sample` picks evenly spaced images from the
batch's sorted file list, deterministically, so the sample is reproducible.
A batch small enough that 10% rounds to zero still gets exactly one
process-check image. Cumulative per-class IAA across all batches is
reported in the M4 dataset quality report — this per-batch gate protects
against process failure (a reviewer clicking through without looking), not
a statistical claim about overall label quality.

---

## 7. Masking shrinks as the ledger grows

After any import, re-running `12_auto_annotate.py` (or `dvc repro
auto_annotate`) will target fewer (image, class) cells — verified cells
(either verdict) are read via `LedgerView` and never re-targeted. M3 wires
the same ledger into `generate_completeness` so the Phase-4 masked-loss
safety net shrinks exactly as verification grows.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| `does not match the taxonomy` | CVAT task's label list was a subset or reordered | Recreate the task from `cvat_labels.json` (§3.1); this batch's export cannot be salvaged |
| `edited a trusted box` | A non-target-class box was moved/resized in CVAT | Revert the edit in CVAT, re-export |
| `has no resolved trusted-class set` | Provenance/label_completeness disagree (merge stage is stale) | Re-run `merge_datasets` |
| IAA gate fails repeatedly for one reviewer pair | Genuine disagreement, not a fluke | Adjudicate together in CVAT before re-exporting; do not lower `min_agreement` to force a pass |
| `conflicting verdict` on import | A prior batch already settled this (image, class) differently | Confirm this is an intentional re-verification, then pass `--supersedes <prior-batch-id>` |

---

Previous: [README.md](./README.md), [auto_annotation_runbook.md](./auto_annotation_runbook.md)

Related: [capture_annotation_runbook.md](../04_dataset_engineering/capture_annotation_runbook.md)
(the IAA instrument this reuses), [risk_register.md](../01_executive_implementation_plan/risk_register.md) (R30–R32)

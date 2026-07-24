# Annotation V2 — Realization Runbook

**Purpose.** The Phase-5 Annotation V2 work (PRs #9 merged, #10 open) landed the
*code + config + tests* for P1–P5, P7, P9. Several steps were deliberately
**deferred** because they need a GPU, large model downloads, extra packages, or
heavy pipeline re-runs that must be re-validated. This runbook is the exact,
ordered set of commands to realize them on a GPU-equipped machine.

**Conventions**
- `PY=.venv/Scripts/python.exe` · `DVC=.venv/Scripts/dvc.exe` (Windows venv).
- Every model weight is **sha256-pinned**; a backend `load()` hard-fails on an
  empty or mismatched pin, printing the computed digest to paste into the config
  (the *pin-bootstrap* flow). This is by design — never disable it.
- **Do not** flip `mode: smoke` → `full`, run `dvc push`, cut a release, or tag —
  all out of scope (same constraints as the original build).
- After any data-mutating stage, `qa_check` must stay **0-critical** and the
  **train/val/test leakage checks must stay green**. That is the gate for every
  section below.

---

## 0. Prerequisites

```bash
# GPU + CUDA torch already present (ultralytics 8.4.96 is installed).
# Extra packages for P6 / P8 only:
.venv/Scripts/pip install sahi        # P6 sliced inference
.venv/Scripts/pip install fiftyone    # P8 review UI
# Pause OneDrive during heavy re-runs (cache is at C:\dvc_cache, off-tree).
```

Confirm the cache is still off OneDrive and writable:
```bash
$DVC cache dir            # → C:\dvc_cache
$DVC config --local cache.type   # → copy  (NOT hardlink; see build-state memory)
```

---

## 1. Pin YOLOE + SAM2 weights (unblocks P5)

Download the weights into `models/annotators/`, then bootstrap the pins.

```bash
# YOLOE (open-vocab seeder). ultralytics fetches on first construction:
$PY -c "from ultralytics import YOLOE; YOLOE('yoloe-11l-seg.pt')"
mv yoloe-11l-seg.pt models/annotators/    # or download directly there
# SAM2 refiner (tighter masks than MobileSAM):
$PY -c "from ultralytics import SAM; SAM('sam2.1_b.pt')"
mv sam2.1_b.pt models/annotators/
```

Bootstrap each pin (run once; copy the printed digest into the config):
```bash
# The backend prints: "Computed digest for yoloe-11l-seg.pt: <sha256> — record it…"
$PY -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" models/annotators/yoloe-11l-seg.pt
$PY -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" models/annotators/sam2.1_b.pt
```
Paste those into `configs/annotation.yaml`:
- `auto_annotation.backends.yoloe.weights_sha256`  (and set `enabled: true`)
- `auto_annotation.refinement.weights: models/annotators/sam2.1_b.pt` + its `weights_sha256`

Leave `yolo_world` enabled too, or disable it if YOLOE fully replaces it — the
two are interchangeable behind the same contract.

---

## 2. Wire batched inference into the orchestrator (finishes P5)

The `yoloe`/`yolo_world` backends already expose `annotate_batch` (per-image
identical, single forward pass). `scripts/dataset/12_auto_annotate.py` still
loops `annotate()` per image (see its main image loop). Change it to:

1. Chunk the targeted images (e.g. 16–32 per batch).
2. Call `backend.annotate_batch(paths, target_ids)` — note targeting differs per
   image, so either batch only images sharing a target set, or pass the union
   and re-filter per image after.
3. Keep the **SAM refine pass per-image** (it prompts SAM with each image's
   boxes) — refine after the batch returns.
4. Preserve deterministic ordering (sorted image order) so the candidates
   artifact stays reproducible.

Validate determinism after wiring:
```bash
$PY scripts/dataset/12_auto_annotate.py --verify-determinism
```

---

## 3. Run auto-annotation at 14k scale (unblocks the report refresh)

```bash
$DVC repro auto_annotate
```
- Writes `data/annotation/candidates/<backend>/candidates.json` for the real
  14,005-image `data/merged`.
- GPU-heavy. Candidates are **advisory only** — they never touch `labels/`
  (ADR-P5-01 invariant). Nothing enters training without human verdicts.

---

## 4. Refresh the stale L4/L5 reports (realizes P1)

The coverage/quality reports were last built at 188-image scale; the new
image-count staleness guard (`sweep_l4_l5_reports`) now FLAGS them until
regenerated.

```bash
$DVC repro coverage_report dataset_quality_report
$PY scripts/qa/run_full_qa.py --exit-zero-on-warnings
```
**Gate:** `annotation_qa_report.json` → `l4_l5_reports.problems_count == 0`
(the "image count 188 != 14005" warnings are gone), verdict ∈ {pass, warn}.

---

## 5. Realize finer dedup (P7) — recover scarce-class frames

P2 showed 96% of the 6,368 local "near-dup" drops are perceptual-only (coarse
8×8 hash over-merging distinct frames). Tune, re-merge, and **re-check leakage**.

```bash
# configs/dataset_sources.yaml → dedup:
#   hash_size: 16          # 256-bit, discriminates distinct frames
#   hamming_threshold: 5   # absolute bits — a 256-bit hash makes 5 much stricter;
#                          # start at 5, raise cautiously only if true dups survive
$DVC repro merge_datasets split_train_val_test generate_completeness qa_check
```
**Gates (all mandatory):**
- `qa_check` → `train_val_leakage` / `train_test_leakage` = **PASS** (0). If
  leakage appears, the threshold is too loose — raise `hamming_threshold`.
- `merged_manifest.json` → `local_captures.duplicates` should drop well below
  6,368; `accepted` rises toward the ~19–22k range (P2 estimate: +2,000–3,500).
- Re-run §3–§4 afterward (merged changed → candidates + reports are stale).

---

## 6. Annotation-quality GT eval (realizes P9)

Score the auto-annotator against the held-out eval set. First produce predicted
labels by running the annotator over the eval images (GPU), then score (CPU):

```bash
# Produce data/annotation/eval_predictions/labels/ by running the enabled
# backend over data/eval/indian_home_v0/images (a small driver over
# backend.annotate_batch, writing YOLO txt per image).
$PY scripts/qa/annotation_gt_eval.py \
    --gt-labels data/eval/indian_home_v0/labels \
    --pred-labels data/annotation/eval_predictions/labels \
    --iou 0.5 --min-precision 0.4
# → data/qa_reports/annotation_gt_eval.json  (per-class P/R/F1/mean-IoU)
```
`--min-precision 0.4` prints the classes whose measured precision is below the
bar — **do NOT enable prompts for those** (feeds §7).

---

## 7. Gate P4 prompt expansion on measured precision

The V2 plan's "prompt all 23 classes" was deferred here (ADR-P5-02 scope-honesty).
Using the §6 report, in `configs/annotation.yaml` add prompts **only** for classes
whose GT precision ≥ the bar (e.g. 0.4). Leave the rest empty — prompting a
low-precision class floods human verification with junk (risk R30). Re-run §3
after editing prompts (the prompt fingerprint invalidates the stale run).

---

## 8. P6 — SAHI sliced inference for tiny objects

**Code shipped:** `src/dataset/annotation/sliced.py` (`plan_slices`,
`remap_box_to_full`, `nms_per_class`, `annotate_sliced`) — dependency-free,
reuses `annotate_batch`; config in `configs/annotation.yaml` → `slicing`.
Remaining to realize (GPU):

- In `scripts/dataset/12_auto_annotate.py`, when `slicing.enabled`, call
  `annotate_sliced(backend, image, priority_target_ids, SliceConfig(...))` for
  the **priority classes only** (SAHI is slow; skip large objects), then SAM
  refine + validate as usual.
- Set `slicing.enabled: true` and tune `slice_*`/`overlap_ratio` for the eval
  imagery. Optional: swap the `sahi` package in behind `annotate_sliced`
  (uncomment it in `requirements-annotation.txt`) — the geometry contract is
  unchanged.
- Confirm tiny-class candidate counts rise vs the non-sliced run.

---

## 9. P8 — CVAT → FiftyOne review UI

**Code shipped:** `src/dataset/annotation/fiftyone_review.py` — the format
bridge (YOLO ↔ FiftyOne, `build_review_dataset`, `launch_app`,
`export_reviewed_labels`), `fiftyone` lazy-imported and in
`requirements-annotation.txt`. Remaining to realize (needs the package + a real
batch):

```bash
.venv/Scripts/pip install fiftyone
```
- In script 13 (`build_verification_batches`), offer the FiftyOne surface as an
  alternative to the CVAT zip: `build_review_dataset(batch_id, samples,
  class_names)` from the batch images + `build_preannotation_labels` output,
  then `launch_app(dataset)` for the reviewer.
- In script 14 (`import_verified_batch`), point it at
  `export_reviewed_labels(dataset, out_dir, name_to_id)` output — it's plain
  YOLO txt, so the **existing** `verified_import` guards
  (`check_non_target_labels_unchanged`, `extract_deltas`) + ledger verdicts run
  unchanged.
- **Parity gate (mandatory):** on one batch, assert the FiftyOne path produces a
  byte-identical ledger delta to the CVAT path (add a regression test). The
  shared `verified_import` machinery makes this hold by construction; the test
  pins it.
- Keep the YOLO-1.1 CVAT path as a documented fallback.

---

## 10. P7 follow-ups (data-recovery, needs judgment)

- **Remap-table recovery** (`scripts/dataset/20_ingest_local_zips.py`
  `_ARCHIVE_CLASS_REMAPS`): the per-slug tables drop in-taxonomy secondary
  classes (cupboard/sink→`Chair`, sink→`Stove`, walking_stick→`knife`/`person`,
  bed→`bottle`). Before adding them, **verify the exact source class-name
  strings** (inspect the ZIP `data.yaml`s via the `_inspect_classes.py` debug
  script) and **assess noise** — walking_stick is a "noisy multi-class dataset";
  some drops were deliberate. Then re-ingest (`dvc repro ingest_local_zips`) and
  re-run §5.
- **Rebalancing:** source `medicine_bottle` (0 instances) via the Roboflow H-B
  track in `configs/dataset_sources.yaml`; consider oversampling the tiny classes
  in the split config. Re-run merge→split→qa.

---

## Rollback

Every stage is DVC-tracked and cache-backed (`C:\dvc_cache`). To undo a bad
re-run: `git checkout -- dvc.lock && $DVC checkout`. Config edits are plain git
reverts. Weights are pinned, so a wrong download fails loudly at `load()` rather
than silently corrupting a run.

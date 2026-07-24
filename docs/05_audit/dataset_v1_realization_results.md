# Dataset v1.0 — Realization Results & Readiness Verdict

**Date:** 2026-07-24 · Follows `dataset_v1_readiness_review.md`. Records the
CPU realization executed on `main`, the final dataset audit, and the v1.0 /
training-readiness verdicts. Branch: `dataset-v1-dedup-realization`.

---

## What was executed (and corrected)

- **Council + Phase-A verification** → surfaced a critical finding: the working
  branch `phase-5-annotation-quality-v2` was cut off an old `main` and **had no
  local_captures integration**; the 14,005-image `data/merged` on disk was a
  stale cross-branch artifact. **Corrected** by merging PR #10 → `main` and
  moving all realization onto `main` (which has local_captures + P1–P9). No data
  lost.
- **Phase E — dedup realization (R3/R4), CPU, validated:** `hash_size:16`
  (256-bit aHash), re-ran `merge_datasets → split → generate_completeness →
  qa_check`.
- **Deferred by decision (needs a bigger GPU box):** Phases B/C/F/G realization —
  `auto_annotate` at scale, YOLOE/SAM2 download+pin, FiftyOne workflow, eval GT
  scoring. The RTX 3050 6 GB is too small for YOLOE-11L+SAM2 over 17k images.

## Dedup recovery (measured)

| Metric | hash_size:8 | **hash_size:16** | Δ |
| --- | --- | --- | --- |
| local_captures accepted | 13,820 | **17,232** | **+3,412** |
| local_captures dup-dropped | 6,368 | 2,956 | −3,412 |
| Total merged images | 14,005 | **17,422** | +3,417 |
| Total boxes | 63,760 | **88,312** | +24,552 |
| Splits (train/val/test) | 11,217/1,394/1,394 | **13,952/1,735/1,735** | — |

Recovery landed across safety-critical classes (gas_cylinder +516, wet_floor
+1,399, medicine_strip +791, knife +653, stove +106 boxes). Matches the P2
estimate (~2–3.5k frames).

## Phase H — Final Dataset Audit (17,422-image build)

| Check | Result |
| --- | --- |
| Reproducibility | ✅ Pipeline reproducible from committed code (`dvc.lock` updated, `hash_size:16`) |
| train_val_leakage / train_test_leakage | ✅ **PASS (0)** — looser dedup did NOT leak across splits (group-aware split held) |
| invalid_yolo_format / class_ids / bbox_out_of_bounds / zero_area / corrupted_images | ✅ PASS (0 critical) |
| duplicate_images | ✅ PASS |
| License gate / eval-overlap | ✅ not critical |
| **Critical issues** | ✅ **0** |
| duplicate_annotations | ⚠️ WARNING (8) |
| empty_label_files | ⚠️ WARNING (20 — the negatives, expected) |
| Image quality (blur/low-light) | ⚠️ 2,726 flagged → `image_quality_quarantine.json` (P3 bucket) |
| L4/L5 reports (coverage/quality) | ⚠️ **STALE @188** — P1 guard correctly flags (needs GPU `auto_annotate`) |
| Class balance | ⚠️ gini **0.73** (↑ from 0.69), imbalance 318.9× — recovery added volume, not balance; `wet_floor` dominates (18,057) |
| Empty class | ❌ **`medicine_bottle` = 0 instances** (safety-critical; needs sourcing) |

## Phase I — Dataset v1.0 Readiness Verdict

### 🔴 NOT READY FOR DATASET v1.0 FREEZE

The dataset is **materially improved and now reproducible**, but v1.0 freeze
requires the items below. Blockers by severity:

| Sev | Blocker | Resolution (owner: runbook §) |
| --- | --- | --- |
| **Critical** | `medicine_bottle` = 0 instances (safety-critical class, no data) | Source via Roboflow H-B track (runbook §10 / R7) + re-merge |
| **High** | No measured annotation quality (P/R/IoU) — v1.0 quality unproven | GPU: `auto_annotate` at 17k → P9 GT scoring (runbook §3,§6) |
| **High** | Coverage / dataset-quality reports stale @188 (release gate RG3 inputs) | GPU: refresh after `auto_annotate` (runbook §4,§7) |
| **Medium** | Class imbalance gini 0.73 (`wet_floor` 18k vs tiny classes) | Oversample/cap in split; source scarce classes (R7) |
| **Medium** | Realization on a branch; DVC outputs not pushed | Merge `dataset-v1-dedup-realization`; `dvc push` when approved |
| **Low** | 2,726 blur/low-light images not yet quarantine-filtered | Apply P3 quarantine to train-facing union |

### What IS ready (green)

✅ Reproducible 17,422-image pipeline · ✅ 0 critical QA · ✅ zero train/val/test
leakage · ✅ +3,412 scarce-class frames recovered · ✅ P1–P9 code complete & tested.

## Phase J — Training Readiness

### 🔴 NOT READY FOR (frozen-v1.0) YOLO TRAINING — but a BASELINE run is viable

- **Full v1.0 training: not recommended yet** — `medicine_bottle` has 0 examples
  (a safety-critical class the model cannot learn), annotation quality is
  unmeasured, and coverage for the 6 open-vocab classes hasn't been realized
  (candidates unverified). Training a "final" model now bakes in these gaps.
- **A BASELINE experiment IS defensible now:** training labels come from trusted
  sources + local captures (not from unverified auto-annotation candidates), the
  dataset is reproducible and leakage-clean. A baseline establishes a reference
  mAP and surfaces per-class weakness empirically — *labelled as v0.x baseline,
  not v1.0*.

### Phase-6 training roadmap (when v1.0 blockers clear)

1. **Config:** `yolo11n`, imgsz 640, 150 epochs, AdamW lr0 1e-3, patience 25
   (`configs/training/yolo11n_config.yaml`); target mAP50 ≥ 0.70, safety-class
   recall ≥ 0.80, ≥15 FPS on Pi 5.
2. **Augmentation:** mosaic + mixup off for final epochs; heavy HSV/blur/
   perspective given indoor domain; class-balanced sampling to fight gini 0.73.
3. **A/B experiments:** (a) hash_size 8 vs 16 dataset (does recovery help mAP?);
   (b) Phase-4 missing-annotation mitigation on/off; (c) auto-annotation
   candidates verified vs not.
4. **Eval protocol:** `evaluate_yolo11n` on the locked `data/eval/indian_home_v0`
   (unseen-home); per-class P/R/mAP; baseline-vs-mitigated deltas (absolute
   numbers underestimate — partial labels).
5. **Logging:** WandB (runs, per-class curves, confusion matrix, sample
   predictions). **Error analysis:** worst per-class FN/FP, small-object recall,
   safety-critical confusion. **Retraining trigger:** after medicine_bottle
   sourcing + annotation realization → v1.0 → full training.

---

## FINAL STATUS

> ## 🔴 NOT READY — for Dataset v1.0 freeze or final YOLO training.
> **Progress:** pipeline reconciled & reproducible on `main`; dedup realized
> (+3,412 frames, 17,422 images, leakage-clean, 0 critical). **Remaining
> blockers:** medicine_bottle (0 data), GPU annotation realization + measured
> quality, coverage/quality report refresh, imbalance. A **v0.x baseline
> training** run is defensible now to establish a reference.

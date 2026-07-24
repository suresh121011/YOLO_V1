# Dataset v1.0 Readiness — Council Review & Realization Plan

**Date:** 2026-07-24 · **Reviewer:** structured multi-lens review (the
`llm-council` skill is not registered in this environment; this is the
equivalent). **Verdict up front: 🔴 NOT READY for Dataset v1.0 freeze or YOLO
training.** The realization work is real, blocking work — not cosmetic polish.

---

## Phase A — Repository Verification (evidence)

| Check | Result |
| --- | --- |
| PRs | #8 **merged**, #9 **merged** (P1/P2/P3/P4/P5/P7), #10 **open** (P9/P6/P8/runbook) |
| Branch | `phase-5-annotation-quality-v2`, 5 commits ahead of `origin/main` (all in PR #10) |
| Git working tree | Clean except pre-existing: modified `data/qa_reports/full_build_preflight.*`; untracked `Dataset/`, `data/raw/local_captures/`, 3 `_debug/_inspect_*.py` scripts |
| GPU | ✅ RTX 3050 **6 GB** (small — OOM risk for YOLOE-11L + SAM2) |
| Packages | ultralytics 8.4.96 ✅, dvc 3.67.1 ✅, **sahi ❌ missing**, **fiftyone ❌ missing** |
| Model weights | `yolov8x-worldv2.pt`, `mobile_sam.pt` present; **YOLOE / SAM2 ❌ not downloaded** |
| **DVC pipeline** | **🔴 EVERY stage out-of-date vs `dvc.lock`** (see below) |

### The blocker: whole-pipeline reproducibility drift

`dvc status` reports **all** stages changed — deps AND outs — cascading from the
top:

```
download_roboflow / download_wider_face  ← committed downloader-script + wider_face config drift
   → remap_classes → merge_datasets       ← + dedup.py (P7), merge.py, config dedup
      → split → generate_completeness → qa_check   ← + run_full_qa.py (P1/P3)
      → auto_annotate                      ← + 12_auto_annotate.py, src/dataset/annotation (P4/P5), annotation.yaml
         → coverage_report → dataset_quality_report
```

The flagged upstream files are **git-clean (committed)**, so this is
*committed-code-vs-stale-`dvc.lock`* drift, not uncommitted scratch. Meaning:
**the on-disk `data/merged`, `data/processed`, and all QA reports were produced
by an older pipeline and no longer match the code that claims to produce them.**
Freezing v1.0 now would freeze an unreproducible artifact.

Two distinct drift sources are entangled and must be separated before any rebuild:
1. **Intended V2 changes** (mine): `dedup.py` hash_size, `run_full_qa.py` guards,
   annotation configs — these are *supposed* to change behaviour and were merged.
2. **Pre-existing drift — RESOLVED (R0):** the flagged files were last changed by
   committed **M7 full-mode-prep** work (`8c2e506` WIDER FACE class_caps,
   `fde808d` Roboflow per-slug license derivation) plus P7 (`dae7ca1`) — all
   landed **after** the last `dvc.lock` update (`528b420`). So it is legitimate
   committed evolution, not corruption. A controlled `dvc repro` is safe, **but
   the rebuilt dataset will differ** from the current 14,005-img / 63,760-box
   snapshot (wider_face handling + hash_size:16 dedup change composition). v1.0
   numbers must be re-measured post-rebuild, not carried over.

---

## LLM-Council Review (7 lenses)

| Lens | Verdict | Key finding / condition |
| --- | --- | --- |
| **Dataset Engineering** | 🔴 Blocked | `data/merged` not reproducible; unexplained wider_face/downloader drift means a rebuild could change composition. Must diff the drift, decide intended vs accidental, then rebuild deterministically before freeze. |
| **CV / YOLO** | 🟡 Conditional | 6 GB GPU is tight for YOLOE-11L-seg + SAM2 — expect OOM; use `yoloe-11s` or CPU-fallback / smaller imgsz, or run realization on a bigger box. mAP claims impossible until a clean dataset + eval run exist. |
| **Annotation** | 🟡 Conditional | P1–P9 code is sound and tested (~190 tests), but `auto_annotate` has never run at 14k scale; candidates are stale @188. Prompt-gating (Phase G) has **no** measured precision yet → cannot enable prompts. |
| **MLOps / DVC** | 🔴 Blocked | The whole DAG is dirty. Realization = a **controlled, staged `dvc repro`** with per-stage validation, not feature work. `dvc.lock` must be regenerated and committed. Cache is off-OneDrive (`C:\dvc_cache`, type=copy) — good. |
| **QA & Validation** | 🟡 Conditional | Gates exist and pass on the *stale* artifacts. After rebuild, leakage + license + eval-overlap gates must be re-asserted. The P1 staleness guard will (correctly) flag the 188-scale reports until refreshed. |
| **Software Architecture** | 🟢 Sound | Candidate→ledger→overlay invariant (ADR-P5-01) intact; new modules (sliced, fiftyone_review, gt_eval) are additive, lazy-imported, tested. No architectural debt introduced. |
| **Release Engineering** | 🔴 Blocked | No release/tag may be cut against a dirty pipeline. `record_release` / training / eval stages are frozen. v1.0 freeze requires: reproducible build + green gates + regenerated reports + a written provenance of the drift resolution. |

### Council consensus

1. **v1.0 freeze is blocked** by the whole-pipeline reproducibility drift — this
   dominates everything else.
2. The remaining work is **realization + reconciliation**, correctly framed as
   *no new features* (the council found none needed for v1.0).
3. **Environment reality:** this 6 GB laptop can do the **CPU** re-runs
   (merge/split/completeness/qa — dedup is CPU) but is marginal for the **GPU**
   `auto_annotate` over 14k imgs (hours + OOM risk), and lacks `sahi`/`fiftyone`
   + YOLOE/SAM2 weights. Some phases are **not safely executable here**.
4. **Do NOT blind-`dvc repro`**: the unexplained wider_face/downloader drift must
   be diffed and understood first, or the rebuild silently changes the dataset.

---

## Consensus Implementation Plan (ranked: priority · dependency · risk)

Legend: 🟢 executable on this box · 🟡 executable but slow/risky here · 🔴 needs a
bigger GPU box / installs / network.

| # | Task | Phase | Priority | Depends on | Risk | Where |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | **Diff & explain the wider_face/downloader/`sources` drift** vs `dvc.lock`; decide intended vs accidental; revert or document | A | **Critical** | — | Med (read-only diff) | 🟢 |
| R1 | Merge PR #10 (P9/P6/P8) so `main` holds all V2 code before rebuild | A | High | — | Low | 🟢 |
| R2 | Clean working tree: commit/ignore `full_build_preflight.*`, remove `_debug_*` scripts or move to `scripts/dataset/dev/` | A | Med | — | Low | 🟢 |
| R3 | **Dedup realization**: `hash_size:16`, re-run `merge_datasets`→`split`→`completeness`→`qa_check`; **leakage gate must stay green**; quantify recovered images | E | High | R0 | Med (CPU, ~30–60 min, mutates data/merged) | 🟡 |
| R4 | Refresh completeness/QA artifacts (falls out of R3) | D | High | R3 | Low | 🟡 |
| R5 | Install `sahi`+`fiftyone`; download+pin YOLOE + SAM2 weights (verify sha256) | B/C | High | R1 | Med (network, MongoDB on Win, OOM) | 🔴 |
| R6 | Wire batching + sliced inference into `12_auto_annotate.py`; run `auto_annotate` at 14k | B | High | R3,R5 | High (GPU hours, 6 GB OOM) | 🔴 |
| R7 | Refresh L4/L5 reports (`coverage_report`,`dataset_quality_report`) | D | High | R6 | Low | 🟡 |
| R8 | FiftyOne workflow parity test (byte-identical ledger delta vs CVAT on 1 batch) | C | High | R5 | Med | 🔴 |
| R9 | Produce eval-set predictions + run `annotation_gt_eval.py` (P/R/F1/IoU, per-class, small-object) | F | High | R5,R6 | Med (GPU) | 🔴 |
| R10 | **Prompt gating** from R9 metrics (enable/disable/needs-data/tune) — evidence-only | G | High | R9 | Low | 🟢 (after R9) |
| R11 | Final dataset audit (dupes/corrupt/empty/imbalance/bbox stats/leakage/version) | H | High | R3,R7 | Low | 🟡 |
| R12 | v1.0 readiness decision + freeze checklist OR blocker list | I | Critical | R11 | — | 🟢 |
| R13 | Training-readiness decision + Phase-6 roadmap (only if R12 passes) | J | Critical | R12 | — | 🟢 |

**Dependency order:** R0 → R1/R2 → R3 → R4 → (R5 → R6 → R7/R8/R9 → R10) → R11 → R12 → R13.

---

## What is safely executable in THIS environment now

- ✅ **R0** (diff the drift — read-only), **R1/R2** (merge PR, tidy tree),
  **R3/R4** (dedup + artifact refresh — CPU, but mutates `data/merged`; needs
  your go-ahead as it's a ~30–60 min data-mutating rebuild), **R10/R12/R13**
  (analysis, once inputs exist).
- 🔴 **R5–R9** need `sahi`/`fiftyone` installs (fiftyone pulls MongoDB — heavy on
  Windows), YOLOE/SAM2 downloads, and GPU `auto_annotate` over 14k images on a
  6 GB laptop (hours; likely OOM with the -11L/SAM2 combo). These are **not**
  advisable to run blindly here — they belong on a bigger GPU box, or with
  explicit consent + the `-11s` model + reduced batch/imgsz.

**Therefore v1.0 cannot be reached in this session alone.** The safe path is:
execute R0→R4 here (reconcile the CPU pipeline, realize dedup, refresh QA), then
gate R5–R9 on your environment/consent, then finish R10–R13.

---

## Preliminary v1.0 verdict

🔴 **NOT READY — Dataset v1.0 freeze blocked.** Blockers by severity:

| Sev | Blocker | Resolution |
| --- | --- | --- |
| **Critical** | Whole-pipeline `dvc.lock` drift → `data/merged` unreproducible | R0 + controlled staged `dvc repro` (R3, R6) with green gates |
| **Critical** | Unexplained wider_face/downloader drift (composition risk) | R0: diff, decide, document/revert before rebuild |
| **High** | `auto_annotate` never run at 14k; candidates + L4/L5 reports stale @188 | R6 + R7 (GPU) |
| **High** | No measured annotation quality (P/R/IoU) → prompt gating & training-readiness unprovable | R9 (GPU) → R10 |
| **Medium** | Dedup recovery (hash_size:16) not realized; ~2–3.5k scarce-class frames still dropped | R3 (CPU) |
| **Medium** | `sahi`/`fiftyone` uninstalled; YOLOE/SAM2 unpinned | R5 |
| **Low** | Working-tree housekeeping (preflight, debug scripts) | R2 |

**Training readiness (Phase J):** cannot be assessed until R11 — but on current
evidence (unreproducible dataset, 0 measured annotation quality, `medicine_bottle`
empty, gini 0.69 imbalance) training would be **premature**.

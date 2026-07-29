# Dataset Changelog

Versioning: `dataset-v{major}.{minor}.{patch}` — major = taxonomy/split
reset; minor = ≥100 new images; patch = label/QA/metadata fixes
(docs/03_engineering_appendix/dvc_pipeline.md). Every release = QA green →
entry here → git tag.

---

## dataset-v0.6.0 — 2026-07-29

First full-mode build (`mode: full`) and the first release validated on a machine
that did not build it. Release track per
[ADR-P5-13](../docs/07_dataset_production/adr/ADR-P5-13-v060-local-capture-release-track.md):
gates RG1–RG8. **This release certifies the data, not that the data trains a good
model** — it carries no training evidence, by design.

| Source | Images accepted | Notes |
|---|---|---|
| COCO 2017 (train2017) | 4,951 / 5,300 | 349 rejected by indoor filter; per-class `class_caps` of 1,200 on all 10 contributed classes |
| Open Images V7 | 1,881 / 2,021 | 138 filtered, 2 duplicates; caps Door 1,200 / Cupboard 2,000 / Gas stove 2,000 (data-driven, see docs/04 §1.1) |
| WIDER FACE | 97 / 97 | research-only license, gated by RG7 |
| Local captures (23 Drive ZIPs) | **16,926 / 26,339** | 3,763 duplicates removed, 5,650 without labels |
| Negatives (COCO) | 497 / 500 | 3 near-duplicates removed; intentionally empty labels |
| **Total merged** | **24,352** | **104,289 boxes** |

- **Split:** group-aware 80/10/10, seed 42 → 20,588 train / 1,882 val / 1,882
  test. The realised ratio is 84.5/7.7/7.7 because groups cannot be broken.
  **Leakage 0/0** (train↔val, train↔test).
- **QA: 0 critical**, 505 warnings, 0 info. The warnings are 497 intentional
  empty negative labels + 8 duplicate annotations; every structural check
  (format, class ids, bbox bounds, zero-area, missing/corrupt files,
  inconsistent pairs) is 0.
- **Completeness:** mean 0.9677, p10 0.8955, 331 images (1.4%) below 0.5,
  residual missing ≈ 2,371 boxes (~2.2%).
- **Balance:** Gini 0.6238 (was 0.7183 at v0.1.0-smoke), imbalance ratio 327.1.
  The ratio worsened slightly because the floor class did not move while the
  ceiling fell.
- **License:** research scope. `wider_face` is noncommercial and enumerated;
  `allow_noncommercial: true`. A commercial build requires the documented v1.1
  path.
- **Reproducibility:** verified 2026-07-29 in a clean `python:3.12-slim` Linux
  container off `main` — 67,734 objects pulled from S3, `dvc repro qa_check`
  rc=0, regenerated report identical to the Windows-built one. Full record in
  `docs/04_dataset_engineering/reproduction_log.md`.

### Known limitations

1. **`completeness.json` over-claims trust for 69.5% of images.** `merge_datasets`
   records one source-level `label_completeness` entry for `local_captures`
   holding the **union** of all 23 slugs' trusted classes, so each of the 16,926
   local-capture images inherits ~**21.7 falsely-trusted classes of 23**.
   `mean_trusted_classes_per_image` is **18.724** where the per-slug manifests
   support ≈ 3.6, and `masked_cell_fraction` is **0.1859** where they support
   ≈ 0.84. Consequence: an unlabelled person in a `bed`-slug frame is treated as
   verified-absent and supervised as background.
   **Do not train on this release with `missing_annotation_mitigation` enabled.**
   No image, label or split is affected. Deferred to `dataset-v0.7.0` with the
   evidence and reasoning in
   [ADR-P5-14](../docs/07_dataset_production/adr/ADR-P5-14-per-slug-completeness-policy.md),
   and pinned by `tests/unit/test_completeness_policy_granularity.py`.
2. **`charger` is below the v1.0 floor** — 110 boxes against 200. The only public
   route is Roboflow, whose slugs are unpopulated.
3. **No custom Indian-home captures.** RG9's channel
   (`data/raw/custom_captures`) reports 0 images / 0 houses / 0 per class. The
   local-capture archives are a different provenance channel and are not
   household captures. Eight classes still require own data: `medicine_strip`,
   `stove`, `gas_cylinder`, `passport`, `cupboard`, `wet_floor`,
   `walking_stick`, `support_handle`.
4. **No training or A/B evidence** (RG10). `eval_report.json` and
   `ab_benchmark/` do not exist; the locked `indian_home_v0` eval set is empty.
5. **`medicine_strip` is still Roboflow-sourced** and marked TEMPORARY in
   `configs/dataset_sources.yaml` — replace with own data before v1.0.
6. **`dvc pull` exits 1 on a clean clone.** Not data loss — all 67,734 objects
   transfer and the dataset checks out completely. `record_release` and
   `evaluate_yolo11n` are declared in `dvc.yaml` but absent from `dvc.lock`, so
   `dvc pull` tries to materialise outs that have never been produced. This
   release produces `data/releases` and clears half of it; the rest rides with
   RG10.
7. **`train_yolo11n`'s recorded dep hashes match no committed content**, so that
   training run is not reproducible. Invisible to `dvc status` on every platform
   because the stage is frozen. Must be cleared by re-running training — not by
   re-stamping hashes — before RG10 evidence is recorded.

---

## dataset-v0.1.0-smoke — 2026-07-14

Smoke-scale validation build proving the acquisition pipeline end-to-end.
**Not a training dataset** — per-class minimums are intentionally unmet.

| Source | Images accepted | Notes |
|---|---|---|
| COCO 2017 (val2017) | 53/60 | 7 rejected by indoor filter; 10 classes via remap |
| Open Images V7 (validation) | 57/60 | Door 39 / Cupboard 37 / Gas stove 6 boxes |
| WIDER FACE (val) | 60/60 | 1,254 face boxes; research-only license (gated) |
| Negatives (COCO) | 18/20 | 2 removed as near-duplicates; empty labels |
| Roboflow Universe | skipped | no datasets configured / no API key |
| **Total merged** | **188** | 2 cross-source duplicates removed, 10 filtered |

- Split: group-aware 80/10/10, seed 42 — zero leakage.
- QA: 0 critical; warnings expected at smoke scale (9 empty classes =
  Roboflow-gated 4 + Phase-3 custom 5; 18 intentional empty negative labels;
  8 blurry + 1 low-light flagged).
- Known follow-ups before dataset-v1.0.0: populate Roboflow slugs + licenses,
  flip `mode: full`, custom Indian-home captures (Phase-3), locked
  `eval-indian-home-v0` set.

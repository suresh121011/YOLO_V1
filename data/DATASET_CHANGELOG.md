# Dataset Changelog

Versioning: `dataset-v{major}.{minor}.{patch}` — major = taxonomy/split
reset; minor = ≥100 new images; patch = label/QA/metadata fixes
(docs/03_engineering_appendix/dvc_pipeline.md). Every release = QA green →
entry here → git tag.

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

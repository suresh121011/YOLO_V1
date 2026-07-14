# 04 — Dataset Engineering & Governance

> Phase-2 companion document. Operational truth for the dataset pipeline lives here;
> the original specification remains in `docs/03_engineering_appendix/`.

---

## 1. Pipeline overview

```
data/raw/<source>/{images,labels}/ + manifest.json     ← downloaders (scripts/dataset/01–04, 06)
   → 05_remap_classes.py       (per-source class-ID remap, in place)
   → 07_merge_datasets.py      (indoor filter + flip-robust dedup → data/merged + lineage)
   → split_dataset.py          (group-aware split → data/processed/{images,labels}/{train,val,test})
   → scripts/qa/run_full_qa.py (QA AFTER split — leakage check needs splits)
```

- **Smoke vs full:** `configs/dataset_sources.yaml` → `mode: smoke` caps every source at
  `smoke.limit_per_source` images. The full dataset build is *one command*: set
  `mode: full`, then `dvc repro`.
- **Reproducibility:** all downloads are annotation-manifest-first with per-image fetch;
  every stage is deterministic given the config + seed; DVC tracks stage inputs/outputs.
- **Naming resolution (audit fix):** new acquisition scripts use the numbered scheme from
  `dvc.yaml` (`01_…07_…`). The proven existing scripts (`split_dataset.py`,
  `check_annotations.py`, `dataset_stats.py`) keep their names; `dvc.yaml` points at them.

## 2. License register

| Source | License | Commercial use | Gate |
|---|---|---|---|
| COCO 2017 | Annotations CC BY 4.0; images retain Flickr photographer licenses | Generally yes (verify per-image for redistribution) | — |
| Open Images V7 | Annotations CC BY 4.0; images CC BY 2.0 | Yes, with attribution | — |
| Roboflow Universe | **Varies per dataset** — must be recorded per slug in `dataset_sources.yaml` before the full build | Varies | Per-dataset review required |
| WIDER FACE | **Research-only / non-commercial** | **No** | `allow_noncommercial` flag |
| Custom captures | Proprietary, consent-gated | Yes (with consent) | Consent reference mandatory |

`allow_noncommercial: true` is acceptable for this research/assignment phase. **Any
commercial or shippable dataset build must set it to `false`**, which drops WIDER FACE —
the `face` class then requires an alternative source or custom capture. The QA license
report fails a build that violates this gate.

## 3. Label-completeness policy (cross-source false-negative prevention)

Merging partially-labeled sources creates silent false-negative supervision: WIDER FACE
labels faces but not persons; COCO labels persons but not faces. An unlabeled visible
object teaches the detector to ignore it.

Policy:
- Every source declares `trusted_classes` — the classes it labels **exhaustively**.
- The merge stage propagates this into `data/merged/merged_manifest.json`
  (`label_completeness`) and QA reports it.
- Training (Phase-5) must consume this metadata (e.g., per-source loss masking or
  source-scoped sampling). Until then it is a **recorded, visible risk**, not a silent one.
- `negatives` images are verified to contain none of the 23 classes (trusted for absence).

## 4. Privacy, PII & DPDP notes

The taxonomy includes direct PII/health signals: `passport` (ID document),
`medicine_strip`/`medicine_bottle` (health), `face`/`person` (biometrics-adjacent).

- Public-source images: used under their licenses; passports in custom captures must have
  personal details blurred (per `docs/03_engineering_appendix/annotation_guide.md`).
- Custom capture sessions (Phase-3) require a signed consent record; manifests store only
  `consent_reference` (an ID) — **never embed PII in manifests or reports**.
- Runtime logging already redacts face/person bounding boxes (`src/logging/`).
- A DPDP Act compliance review is scheduled for V2 (docs/02 security_privacy.md); dataset
  distribution outside the team is out of scope until then.

## 5. Split governance

- Strategy is configuration, not code: `configs/dataset_split_config.yaml` →
  `split.strategy` (`group_aware` default; `stratified_group` available;
  `kfold`/`leave_one_house_out` reserved — see `src/dataset/splitting/registry.py`).
- Groups (video/burst/capture-session) never straddle splits; capture sessions from
  Phase-3 use `CaptureSessionManifest.house_id` as the future LOHO grouping key.
- Dedup runs **before** splitting, with flip-robust perceptual hashing
  (`dedup.check_flips`) because Roboflow Universe datasets are frequently pre-augmented
  copies of COCO — plain hashing would let augmented twins straddle train/val.
- Leakage verification runs at split time and again in QA. Zero tolerance.

## 6. Versioning & change control

- Scheme: `dataset-v{major}.{minor}.{patch}` (major = taxonomy/split reset; minor = ≥100
  new images; patch = label/QA fixes) — per `docs/03 dvc_pipeline.md`.
- Every release: QA green (0 critical) → `data/DATASET_CHANGELOG.md` entry → git tag +
  DVC-tracked data. Smoke validation is tagged `dataset-v0.1.0-smoke`.
- The tracked `.dvc/config` carries **no machine-specific cache path** (default: in-repo
  `.dvc/cache`). If the repo lives inside a cloud-synced folder (OneDrive etc.), relocate the
  cache **locally, per machine** via the untracked override — never in the shared config:
  `dvc cache dir --local <path-outside-synced-tree>` (writes `.dvc/config.local`, gitignored).
  Cloud sync on the cache causes corruption/thrash; excluding `data/` from sync is also
  recommended.
- **DVC remote (`storage`)**: an S3 remote is configured as the default in
  `.dvc/config` at `s3://elderly-assistant-mlops/datasets/yolo_v1`. Activation
  steps (run on the machine that holds the smoke data):
  1. Create the S3 bucket `elderly-assistant-mlops` (or update the URL via
     `dvc remote modify storage url s3://<bucket>/<prefix>` for S3-compatible providers;
     set `endpointurl` for R2/B2/MinIO).
  2. `pip install "dvc[s3]"` and export credentials (see `.env.example`); credentials must
     never enter tracked config — use env vars, an AWS profile, or
     `dvc remote modify --local` (writes gitignored `.dvc/config.local`).
  3. `dvc push` — uploads the cache for every `dvc.lock` output; from then on
     `dvc push`/`dvc pull` is part of every dataset release (definition of done).
  4. Gate check: on any clean machine, `dvc pull && dvc repro qa_check` must succeed —
     see `reproduction_log.md` in this directory for executed reproduction tests.
  **Phase-3 data collection must not start before step 3 is done** — until the first
  `dvc push`, the dataset exists on a single machine (risk C1 in the Phase-2 review).

## 7. Explicit Phase-2 descope (approved)

The 8 classes requiring custom Indian-home capture (`medicine_strip`, `wet_floor`,
`walking_stick`, `support_handle`, `passport`, Indian `stove`/`cupboard`/`gas_cylinder`)
are **Phase-3 human tasks** and are excluded from dataset-v1.0.0 acceptance counts. The
ingest path ships ready in Phase-2 (`data/raw/custom_captures/` + `CaptureSessionManifest`).
Likewise, a locked real-home eval set (`eval-indian-home-v0`) should be captured in
Phase-3; mAP measured only on web-photo validation data is treated as an optimistic bound,
not product truth.

Known modeling risk (logged as R24): `wet_floor` may be ill-posed as a bounding-box class
(amorphous, low-texture region). Revisit class definition before Phase-5 training.

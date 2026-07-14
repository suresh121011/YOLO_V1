# 04 — Dataset Engineering & Governance

> Phase-2 companion document. Operational truth for the dataset pipeline lives here;
> the original specification remains in `docs/03_engineering_appendix/`.

---

## 1. Pipeline overview

```
data/raw/<source>/{images,labels}/ + manifest.json     ← downloaders (scripts/dataset/01–04, 06)
data/raw/custom_captures/{images,labels,manifests}/    ← 08_ingest_capture_session.py (Phase-3,
                                                           human-in-the-loop; frozen DVC stage)
   → 05_remap_classes.py       (per-source class-ID remap, in place; custom captures already
                                 taxonomy-ID, no remap needed)
   → 07_merge_datasets.py      (indoor filter + flip-robust dedup → data/merged + lineage;
                                 custom captures get top dedup priority)
   → split_dataset.py          (group-aware / leave-one-house-out split →
                                 data/processed/{images,labels}/{train,val,test})
   → scripts/qa/run_full_qa.py (QA AFTER split — leakage + eval-set overlap checks need it)

data/eval/indian_home_v0/                              ← locked real-home eval set (Phase-3,
                                                           never merged into train/val/test)
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
- Custom capture sessions (Phase-3) require a signed consent record (local-only registry,
  never committed — `data/consent/README.md`); manifests store only `consent_reference`
  (an ID) — **never embed PII in manifests or reports**.
- Every captured image has EXIF/GPS metadata stripped at ingest
  (`src/dataset/capture/exif.py`) before it ever touches `data/raw/custom_captures/`.
- Runtime logging already redacts face/person bounding boxes (`src/logging/`).
- A DPDP Act compliance review is scheduled for V2 (docs/02 security_privacy.md); dataset
  distribution outside the team is out of scope until then.

## 5. Split governance

- Strategy is configuration, not code: `configs/dataset_split_config.yaml` →
  `split.strategy` (`group_aware` default; `stratified_group` and
  `leave_one_house_out` available; `kfold` reserved — see
  `src/dataset/splitting/registry.py`).
- Groups (video/burst/capture-session) never straddle splits. `leave_one_house_out`
  (Phase-3) raises the leakage unit from capture-session to **house**: all
  sessions from one house share a split, derived from `CaptureSessionManifest.house_id`
  via merged-filename pattern matching (`SplitContext.house_pattern`).
  `holdout_houses` forces named houses entirely into test — the mechanism
  behind the locked `eval-indian-home-v0` set (§7, houses used for eval
  never contribute training data). Public-source images (no house match)
  degrade to `group_aware` behavior automatically.
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
- **Custom captures and the eval set enter DVC as `frozen: true` stages**
  (`ingest_custom_captures`, `ingest_eval_set` in `dvc.yaml`), not as normal
  stages with the capture inbox as a dependency. A normal stage was rejected:
  `dvc repro` deletes stage outs before re-running, which would silently wipe
  real ingested photos on any machine that runs `dvc repro` without the
  inbox present. A `dvc add` pointer was also rejected — it would split
  dataset tracking across two mechanisms (dvc.lock stage outs vs standalone
  `.dvc` files), breaking the single-mechanism invariant established in
  Phase-2. The human loop instead is: ingest → annotate →
  `dvc commit -f <stage>` → `git commit dvc.lock` → `dvc push` (full SOP:
  `capture_annotation_runbook.md`). `ingest_eval_set` is deliberately **not**
  a dependency of `qa_check` — that would break `dvc repro qa_check` on any
  machine/branch before an eval set exists. QA reads the eval set
  opportunistically instead (§8, `{"available": false}` pre-Phase-3).

## 7. Phase-3 tooling (shipped)

The 8 classes requiring custom Indian-home capture (`medicine_strip`, `wet_floor`,
`walking_stick`, `support_handle`, `passport`, Indian `stove`/`cupboard`/`gas_cylinder`)
are **Phase-3 human tasks**: collection, annotation and validation are now fully
tooled (`src/dataset/capture/`, `scripts/dataset/08–10`); only the actual
photography, CVAT annotation, and the account/credential setup they require
remain human work. Operational SOP: `capture_annotation_runbook.md`.

- **Ingest** (`08_ingest_capture_session.py`): inbox → validated session —
  corruption/size/duplicate gates, **mandatory EXIF/GPS stripping**
  (`src/dataset/capture/exif.py`), session-prefixed sequential naming,
  per-session `CaptureSessionManifest` + rebuilt aggregate `SourceManifest`
  (feeds the §2 license report automatically via the existing
  `data/raw/*/manifest.json` glob).
- **Consent** (`src/dataset/capture/consent.py`): a local-only, gitignored
  registry (`data/consent/`, never a DVC output) resolves `consent_reference`
  IDs to pseudonymous house IDs and withdrawal state — no PII ever enters
  git or S3.
- **Annotation import** (`09_import_annotations.py`, CVAT "YOLO 1.1" or any
  YOLO-format export): verifies the export's class order matches
  `configs/data.yaml` ID-for-ID (**the CVAT footgun** — a subset/reordered
  label list silently shifts every class ID with no other detectable
  signal), validates session-scoped coverage, stages per annotator, and
  computes inter-annotator agreement (`src/dataset/capture/agreement.py`,
  greedy IoU matching) before finalize.
- **Progress** (`10_capture_progress.py`): per-class counts vs the
  ≥200-instance governance minimum, 2,000-image total, house/room/lighting
  coverage, and withdrawn-consent flags — `data/qa_reports/capture_progress.*`.
- A locked, leakage-checked real-home eval set (`eval-indian-home-v0`)
  should be captured in Phase-3 (§8); mAP measured only on web-photo
  validation data is treated as an optimistic bound, not product truth.

## 8. Eval-set guards & the wet_floor (R24) decision gate

Two QA checks (`scripts/qa/run_full_qa.py`, opportunistic —
`{"available": false}` before any eval data exists) protect the eval set's
validity, merged into `annotation_qa_report.json` under `eval_set`:

- **`check_eval_overlap`** — CRITICAL if the locked eval set shares any
  image with train-facing data, checked two ways: exact SHA-256 match and
  flip-robust perceptual near-duplicate (same aHash+Hamming-distance method
  as merge-time dedup, §5). Eval images are compared only against the
  frozen train-facing set, never against each other, so eval-internal
  duplicates are never mistaken for leakage.
- **`check_house_exclusivity`** — WARNING (not CRITICAL) if a `house_id`
  appears in both training captures and the eval set: training is still
  valid, but the "unseen home" claim the eval set is meant to support
  weakens.

**wet_floor (risk R24):** logged risk that `wet_floor` may be ill-posed as a
bounding-box class (amorphous, low-texture region, easily confused with dry
shiny marble). Collected as a normal bbox class under a tightened protocol
(paired wet/dry negatives), with a measurable pilot gate: the first
~50-image dual-annotated wet_floor session must reach inter-annotator
agreement ≥ `annotation.iaa.wet_floor_min_agreement` (0.60,
`configs/capture_config.yaml`) or the class is demoted to scene-level
(class ID 20 stays reserved — no taxonomy renumber). Full protocol:
`capture_annotation_runbook.md` §8. Decision and evidence recorded in
`docs/01_executive_implementation_plan/risk_register.md` R24.

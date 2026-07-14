# Phase-3 Capture & Annotation Runbook

> Operational SOP for collecting and annotating custom Indian-home data.
> Tooling reference: `docs/04_dataset_engineering/README.md` §7–8. Class
> definitions and capture guidance: `docs/03_engineering_appendix/annotation_guide.md` §10.

---

## 0. Pre-flight gate

**Phase-3 collection must not start before the first `dvc push` has been
done** (docs/04 §6, risk C1). Until the smoke dataset is pushed to the S3
remote, the dataset exists on a single machine — starting real collection
before that is pushed risks losing it. Check:

```bash
dvc status -c        # cloud status; empty diff means the remote is in sync
```

**OneDrive / cloud-synced repos:** never point `--inbox` at a folder inside
the synced tree. Photo inboxes can hold thousands of files; syncing them
mid-write causes corruption and thrash. Use a local, non-synced folder
(e.g. `C:\capture_inbox` on Windows) and pass it explicitly:

```bash
python scripts/dataset/08_ingest_capture_session.py --inbox C:\capture_inbox ...
```

The DVC cache should already be relocated per-machine (docs/04 §6,
`dvc cache dir --local`) — verify this before a large capture push.

---

## 1. Consent (before any camera comes out)

1. Get the signed consent form from the household (offline artifact — never
   digitized into the repo). Template: `docs/03_engineering_appendix/consent_form_template.md`.
2. Assign a pseudonymous house ID (`h01`, `h02`, …) — never derived from a
   name or address.
3. Add a record to the **local, gitignored** registry
   (`data/consent/consent_registry.yaml`, format in `data/consent/README.md`):
   ```yaml
   CONSENT-h01-2026-001:
     house_id: h01
     granted_on: "2026-07-20"
     scope: dataset-training
     withdrawn: false
   ```
4. The `consent_id` becomes the `--consent-ref` passed to every ingest for
   that house.

**Withdrawal SOP:** if a participant withdraws consent, set
`withdrawn: true` on their record. Run `10_capture_progress.py` — it flags
every session referencing the withdrawn ID. Remove those images/labels from
`data/raw/custom_captures/`, `dvc commit -f ingest_custom_captures`, and cut
a `dataset-v{x}.{y}.{z+1}` patch release removing the affected instances.

---

## 2. Capture (per session)

A **session** = one visit to one room, one lighting condition. Follow the
capture-variations checklist (`annotation_guide.md` §10.3) and the formal
class definitions (§10.4) — Indian-specific positives matter (Prestige/Pigeon
stoves, HP/Bharat Gas/Indane cylinders, Godrej almirah, Indian squat toilet).

For **wet_floor** specifically, follow the tightened protocol in §10.2/§10.4:
every wet-floor session must include at least one **paired dry-floor
negative** of the same surface (the known confuser is shiny dry marble).

Photos land in the local inbox (see §0). Session ID grammar:
`h{NN}_{room}_s{NNN}` (e.g. `h01_kitchen_s001`) — house and room are parsed
from it, so get it right; it must match a room in `capture_config.yaml`.

---

## 3. Ingest

```bash
python scripts/dataset/08_ingest_capture_session.py \
    --session-id h01_kitchen_s001 \
    --lighting daylight \
    --device "Pixel 7" \
    --date 2026-07-20 \
    --consent-ref CONSENT-h01-2026-001 \
    --classes gas_cylinder,stove
```

This validates every image (corruption, minimum dimension, intra-session
duplicates), **strips EXIF/GPS metadata**, copies into
`data/raw/custom_captures/images/` with session-prefixed sequential names,
and writes/updates the session manifest + aggregate manifest.

- Re-running the same `--session-id` against a refilled inbox **appends**
  (sequence numbers continue); already-ingested images are rejected as
  duplicates — safe to re-run.
- `--dry-run` validates without writing.
- Exit 0 = clean, 2 = some rejections (check the log — usually junk/blurry
  shots), 1 = structural failure (bad session ID, consent problem).

---

## 4. Annotate (CVAT, dual-annotator)

1. Create a CVAT task for the session's images.
2. **Load the label list from `configs/data.yaml`, in taxonomy order, all 23
   classes** — not a filtered subset, not reordered. This is the single
   most dangerous CVAT footgun: a task built from a subset or reordered
   label list silently shifts every class ID, and the numbers stay "valid"
   so nothing downstream can catch it except the class-order check below.
3. Two annotators label independently (per governance: dual-annotator
   workflow). If the session includes a `passport`, verify personal details
   are blurred before export (annotation_guide.md §10.2).
4. Export each annotator's work as **CVAT "YOLO 1.1"** (Export dataset →
   format `YOLO 1.1`). See §10.6 in annotation_guide.md for the export
   dialog walkthrough.

Import each export:

```bash
python scripts/dataset/09_import_annotations.py \
    --session h01_kitchen_s001 --stage \
    --export exports/asha.zip --annotator asha

python scripts/dataset/09_import_annotations.py \
    --session h01_kitchen_s001 --stage \
    --export exports/ravi.zip --annotator ravi
```

`--stage` verifies the class order against `configs/data.yaml` (CRITICAL,
aborts on mismatch), checks session coverage and line-level YOLO format, and
warns if a declared class (`--classes` at ingest) has zero boxes.

Compare the two annotators:

```bash
python scripts/dataset/09_import_annotations.py \
    --session h01_kitchen_s001 --compare
```

Writes `data/qa_reports/iaa_h01_kitchen_s001.{json,md}`. Exit 0 = agreement
meets the gate (`configs/capture_config.yaml` `annotation.iaa.min_agreement`,
0.60 for `wet_floor` — see §8 below); exit 2 = below gate, adjudicate
disagreements in CVAT (edit → re-export → re-stage → re-compare) before
finalizing.

Finalize the chosen annotator's labels:

```bash
python scripts/dataset/09_import_annotations.py \
    --session h01_kitchen_s001 --finalize --from asha
```

Copies labels into `data/raw/custom_captures/labels/`, marks the session
`finalized`, and updates class counts.

---

## 5. Check progress

```bash
python scripts/dataset/10_capture_progress.py
# or: make capture-progress
```

Reports per-class counts vs the ≥200-instance target, total images vs 2,000,
house/room/lighting coverage, annotation-status breakdown, and consent
anomalies. `--fail-under-targets` exits 1 while targets are unmet — useful
in a CI/reminder context, not a blocking gate for normal work.

---

## 6. Record in DVC and push

```bash
dvc commit -f ingest_custom_captures
git add dvc.lock
git commit -m "data: ingest h01_kitchen_s001 (stove, gas_cylinder)"
dvc push
```

`ingest_custom_captures` is a **frozen** DVC stage — `dvc repro` never runs
it (a repro on a fresh machine would otherwise delete and try to
regenerate real photos from nothing). `dvc commit -f` is how the ingested
data gets recorded into `dvc.lock` after a real ingest run. This is the
same convention as the `train_yolo11n` frozen stage.

**dvc.lock CRLF caveat (finding F1):** if `dvc commit` reports the lock
file as changed on every run with no content difference, check line
endings — `git config core.autocrlf` should be consistent across
contributors' machines on this repo.

Once the **first** session is finalized, flip
`sources.custom_captures.enabled: true` in `configs/dataset_sources.yaml`
(the finalize command prints this reminder) so the merge stage picks it up
on the next `dvc repro`.

---

## 7. Eval set (`eval-indian-home-v0`)

The locked evaluation set gives an honest "unseen home" mAP, distinct from
web-photo validation data. Same ingest tooling, different target:

```bash
python scripts/dataset/08_ingest_capture_session.py \
    --dataset eval --session-id h05_hall_s001 --lighting daylight \
    --consent-ref CONSENT-h05-2026-001 --classes ...
```

**Use houses that never contribute training data.** After annotating (same
§4 flow with `--dataset eval`), verify no leakage and lock it:

```bash
python scripts/qa/run_full_qa.py --eval-dir data/eval/indian_home_v0
# eval_set.overlap.exact_overlap_count and .near_overlap_count must be 0
# eval_set.house_exclusivity.shared_houses must be empty (WARNING if not)

python scripts/dataset/08_ingest_capture_session.py --dataset eval --lock-eval
dvc commit -f ingest_eval_set
git commit -m "data: lock eval-indian-home-v0"
dvc push
git tag eval-indian-home-v0
```

Locking is **immutable by design** — a locked set refuses further ingest.
Any expansion means cutting `indian_home_v1` (a new root, new stage out),
never editing the locked set in place.

---

## 8. wet_floor (R24) pilot gate

`wet_floor` is collected as a normal bbox class, but its viability as a
bbox target is not yet settled (risk R24 — amorphous, low-texture region,
easily confused with dry shiny marble). This is measurable, not a guess:

1. Collect and dual-annotate the **first ~50-image wet_floor session**
   (paired wet/dry negatives per §2).
2. Run `09_import_annotations.py --compare`.
3. **If per-class IAA ≥ 0.60 (the configured `wet_floor_min_agreement`) →
   keep as a bbox class**, continue collecting toward the 200-instance
   target.
4. **If below 0.60 → demote to scene-level.** wet_floor bboxes are excluded
   from the dataset-v1.0.0 acceptance table; class ID 20 stays reserved in
   `configs/data.yaml` (no taxonomy renumber, no dataset major bump); the
   wet/dry photo pairs still feed a Phase-6 SmolVLM2 scene-classification
   evaluation set instead.
5. Record the outcome (agreement score, decision) in
   `docs/01_executive_implementation_plan/risk_register.md` R24 and
   `data/DATASET_CHANGELOG.md`.

A second checkpoint applies regardless of the pilot outcome: if kept, and
Phase-5 baseline training yields wet_floor AP50 < 0.30, reopen the
demotion path.

---

## 9. Roboflow slug selection (unblocks 4 more classes)

`medicine_bottle`, `charger`, `wire`, `gas_cylinder` have partial Roboflow
Universe coverage (`configs/dataset_sources.yaml` `sources.roboflow.datasets:
[]` — currently empty). This is a separate, faster-to-unblock human task:

1. Search Roboflow Universe per class (search terms: "medicine bottle
   detection", "electrical wire detection", "LPG cylinder detection",
   "phone charger detection").
2. For each candidate dataset, record: slug (`workspace/project`), version,
   **license** (must not be silently non-commercial — check against
   `allow_noncommercial`), and its class name → taxonomy name mapping.
3. Populate `sources.roboflow.datasets` in `configs/dataset_sources.yaml`
   per the commented template already in the file.
4. Set `ROBOFLOW_API_KEY` (see `.env.example`) and re-run
   `download_roboflow`.

---

## 10. Release checklist — Dataset v1.0.0

1. All 8 custom classes ≥ 200 instances (or wet_floor formally demoted per
   §8), ≥ 3 houses represented, ≥ 2,000 total custom images
   (`10_capture_progress.py` reports `targets_met: true`).
2. Roboflow slugs populated (§9); `mode: full` set.
3. `eval-indian-home-v0` locked, zero overlap, zero shared houses (§7).
4. `dvc repro` full build; `python scripts/qa/run_full_qa.py` → 0 critical.
5. `data/DATASET_CHANGELOG.md` entry (template in annotation_guide.md §10.5).
6. Git tag `dataset-v1.0.0`; `dvc push`.
7. Unfreezes `train_yolo11n` for Phase-5.

---

Previous: [README.md](./README.md)

Related: [annotation_guide.md](../03_engineering_appendix/annotation_guide.md),
[consent_form_template.md](../03_engineering_appendix/consent_form_template.md),
[risk_register.md](../01_executive_implementation_plan/risk_register.md)

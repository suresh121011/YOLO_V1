# Scenario Clip Collection Plan — First Batch (100 Accepted Clips)

**Status:** ready for field collection. M7 real-world acceptance **PASSED 10/10** on 2026-08-11.
**Protocol:** [clip_capture_protocol.md](clip_capture_protocol.md) — consent, sanitisation, ingest.
**This document:** what to film, how much, who films it, and what makes a clip count.

---

## 1. M7 component status (Phase A)

Audited against the code, not against the documentation that describes it.

| M7 component | Status | Evidence | Remaining issue |
|:---|:---|:---|:---|
| MP4 metadata stripper | **working, defect fixed** | `m7_acceptance` A3→A5: GPS `+12.9716+077.5946/` present before, zero residual after | — |
| ffprobe read-back | **working, proven falsifiable** | A6 re-runs the detector on the pre-strip file and it still fires, so A5 is a real negative | — |
| `handler_name` gate | **redesigned** | A9/A10 | Was a defect — see §2 |
| `encoder` suppression | **fixed** | `-fflags +bitexact` added; A5 shows it gone | — |
| `scenario-video` consent scope | **working** | A7: a `dataset-training` consent is refused with the scope message | — |
| Consent registry path | **fixed** | Was hardcoded; now read from `consent.registry_path` | — |
| `clips:` config block | working | `test_the_repo_config_carries_a_clips_block` | — |
| Image extensions not widened | working | `test_image_extensions_were_not_widened_to_video` | — |
| Clip manifest / schema | **extended** | provenance, review lifecycle, `pre_dwell` — see §2 | — |
| Frozen DVC ingest stage | working | `dvc repro --dry` shows it frozen; `--verify-all` exits 0 on an empty tree | Nothing to commit until clips exist |
| Positive/negative validation | working | 15 tests in `test_clips.py` | — |
| 30 % negative floor | working, **now enforced at set level** | `test_an_all_positive_batch_fails` | — |
| Capture protocol doc | working | `clip_capture_protocol.md` | — |
| Regression tests | working | 346 scenario-engine tests | — |
| **Real clip ingested** | **YES** | `data/qa_reports/m7_acceptance_report.json` | Fixture is synthetic; see §2.4 |

## 2. What the real-world run changed

### 2.1 A defect the unit tests could not have found

`FORBIDDEN_METADATA_KEYS` required `handler_name` to be **absent** after stripping. Against real
ffmpeg 9.0 that is impossible: `handler_name` lives in the MP4 `hdlr` box, which is **structural and
mandatory**, and the muxer rewrites it on every remux. Neither `-map_metadata -1` nor
`-metadata:s handler_name=` removes it — both were tried and measured.

The gate was therefore unsatisfiable: **every clip would have been rejected**, forever. That is the
worst kind of privacy control, because the pressure to switch it off is irresistible and the person
switching it off learns that the gate is noise.

Fixed by changing the *kind* of check. `handler_name` is now validated by **value** against
`NEUTRAL_HANDLER_NAMES` — the names our own remux writes. A device-chosen handler
(`Samsung Video Handler`, `GoPro AVC`) still fails, because that genuinely identifies the phone.
`encoder` stayed in the absence list because it *is* removable, with `-fflags +bitexact`.

### 2.2 Permission is now exactly one thing

The manifest had no way to record where an external clip came from, and `consent_reference` was
unconditionally required. Together those meant a third-party video could only be ingested by
**inventing a consent id** — recording a consent no human ever gave.

Now `SourceProvenance` splits the two worlds and refuses to blur them:

- `own_capture` → permitted by `consent_reference`. Licence fields must be empty.
- `external` → permitted by a **licence from an allowlist**, with `source_url`, `source_platform`,
  `creator` and `license_url` all mandatory. `consent_reference` must be empty.

The licence is an allowlist, not a free string, because "free to use" and "publicly viewable" are the
two phrases most likely to end up in that field.

### 2.3 A clip is not a dataset member until a human says so

Added `review_status` (`pending`/`accepted`/`rejected`), `reviewed_by` and `rejection_reason`.
Acceptance requires a named reviewer; rejection requires a reason. `metadata_stripped` is machine-
verified and deliberately does **not** count as review — sanitisation says nothing about whether a
private document or an unapproved person is in frame.

### 2.4 The acceptance fixture is not a dataset member

The ingested clip is synthetic (`ffmpeg testsrc2`), lives in a temporary tree, and uses a temporary
consent registry under reserved house id `h99`. It proves the pipeline; it is not a scenario clip and
must never be counted toward the 100. Deliberately not committed to `data/scenario_engine/clips/`.

## 3. The scenario reality check — read this before filming anything

**Most short clips of a hazard are correctly _negatives_.** This is the single most important fact in
this document, and it is not a limitation — it is the design working.

Every active scenario carries a temporal requirement. A clip can only be an **engine-positive** if it
runs long enough for that requirement to elapse:

| Scenario | Needs | Filmable as a positive? |
|:---|---:|:---|
| SC-BTH-001 wet floor | **3 s** | ✅ yes, a 15 s clip |
| SC-BTH-002 no grab bar | **60 s** | ✅ yes, ~70 s |
| SC-COR-001 cord in walkway | **120 s** | ✅ yes, ~130 s |
| SC-BTH-003 prolonged bathroom | **150 s** | ✅ yes, ~160 s |
| SC-MED-001 medicine left out | **170 s** | ✅ yes, ~180 s |
| SC-MOB-001 walking aid left behind | **360 s** (6 min) | ⚠️ long-form exception |
| SC-KIT-002 knife left out | **660 s** (11 min) | ⚠️ long-form exception |
| SC-KIT-001 unattended cooking | **960 s** (16 min) | ⚠️ long-form exception |
| SC-SYS-001 nobody seen | **28 800 s** (8 h) | ❌ not filmable |

`max_duration_s` was raised from 120 s to **200 s** for exactly this reason: at 120 s, SC-COR-001,
SC-BTH-003 and SC-MED-001 could not be demonstrated at all.

**The consequence for the collection brief.** A 20-second clip of *a person at a stove with a gas
cylinder* is not an `SC-KIT-001` positive. The engine will not fire, and it is right not to — the
scenario is "cooking left **unattended**", which needs the person to leave and stay gone for fifteen
minutes. Such a clip is a **negative with `negative_kind: pre_dwell`**, and it is genuinely valuable:
it is the alarm-fatigue guard, the test that the device does not nag someone who is standing right
there cooking.

`scripts/scenarios/33_clip_dataset_report.py` enforces this. A positive clip shorter than its
scenario's requirement is reported as impossible, with the relabelling instruction. Nothing is thrown
away; it is relabelled.

## 4. Reconciling the requested scenario list

The collection brief asked for scenario groups that do not exist in this repository. Each was checked
against `configs/scenarios/` rather than assumed.

| Requested | Reality | What to film instead |
|:---|:---|:---|
| Hydration — person + water bottle | **SC-SAF-005 `rejected`.** Hydration monitoring needs to know whether someone drank; a bottle in frame does not say that | Collect as `out_of_taxonomy` negatives — a bottle must not trigger a medicine alert |
| Electrical — person + charger | No scenario. `charger` is a detector class with no rule | Detection-only clips + `SC-COR-001` negatives |
| Documents — person + passport | `passport` is the **privacy-disabled** class; no scenario | Do **not** film passports. See §5 |
| Computer — laptop, monitor | No scenario | Detection-only clips; useful as background context |
| Bedroom — bed, chair, cupboard | No scenarios | Detection-only clips |
| Wandering — person + door | **SC-SAF-008 `rejected`.** `door` has no open/closed state | Detection-only |
| Fall risk `SC-FAL-001` | No such id. **SC-SAF-001 fall detection is rejected** | `SC-BTH-001` wet floor is the real fall-risk scenario |
| `SC-ELC-002` | No such id | — |
| Medical — medicine strip/bottle | ✅ **SC-MED-001 exists** | Positives at ≥180 s |
| Bathroom — toilet, sink | ✅ **SC-BTH-002** (toilet + no grab bar). Sink → SC-SAF-006 rejected | SC-BTH-002 positives |
| Kitchen — stove, knife, cylinder | ✅ **SC-KIT-001/002** exist but are absence-based | Mostly `pre_dwell` negatives |
| Mobility — walking stick, grab bar | ✅ **SC-MOB-001, SC-BTH-002** | Mixed |

Nothing here was silently renamed. Scenario ids and semantics are unchanged; the brief's groupings are
mapped onto them.

## 5. Two hard prohibitions

**Do not film passports, Aadhaar cards, PAN cards, bank documents or prescriptions.** `passport` is
class 8 in the taxonomy and is **disabled at runtime for privacy**. There is no scenario that uses it
and there never will be. A clip containing an identity document is a privacy incident regardless of
consent.

**Do not stage a real hazard.** No actual slips, no gas released, no live electrical exposure, no
unsafe knife handling. Wet floor is filmed by wetting a floor and *keeping people away from it*. If a
genuine hazard exists in a household, fix it — do not film it.

## 6. The 100-clip matrix

75 positives-or-detection clips, 30 negatives (30 %). Ids follow
`{house}_{room}_{session}_c{NNN}` — the repository's clip grammar. The `SC-XXX-NNN_POS_NNN` style in
the brief is **not** used, because the existing grammar is what makes extracted frames group as one
leakage unit for free.

| # | Scenario / purpose | Engine label | Clips | Length | Room |
|:--|:---|:---|--:|:---|:---|
| 1 | SC-BTH-001 wet floor | positive | 8 | 15–20 s | bathroom, kitchen |
| 2 | SC-BTH-002 no grab bar near toilet | positive | 8 | 70–90 s | bathroom |
| 3 | SC-COR-001 cord in walkway | positive | 6 | 130–150 s | corridor, hall |
| 4 | SC-BTH-003 prolonged bathroom | positive | 5 | 160–180 s | bathroom |
| 5 | SC-MED-001 medicine left out | positive | 6 | 180–200 s | bedroom, hall |
| 6 | SC-MOB-001 walking aid left behind | positive, long-form | 2 | 6–7 min | bedroom, hall |
| 7 | SC-KIT-002 knife left out | positive, long-form | 2 | 11–12 min | kitchen |
| 8 | SC-KIT-001 unattended cooking | positive, long-form | 2 | 16–17 min | kitchen |
| 9 | Kitchen composition (person cooking, stove + cylinder + knife present) | **negative `pre_dwell`** | 10 | 15–20 s | kitchen |
| 10 | Bathroom composition (person present, grab bar visible) | **negative `absence`** | 4 | 15–20 s | bathroom |
| 11 | Walking stick **in use** by a person | **negative `assistive`** | 4 | 15–20 s | any |
| 12 | Shiny/dry floor (wet-floor confuser) | **negative `confuser`** | 4 | 15–20 s | bathroom, hall |
| 13 | Water bottle / plastic bottle near person (medicine confuser) | **negative `confuser`** | 3 | 15–20 s | kitchen, bedroom |
| 14 | Hand-held objects: phone, remote, spoon, pen, keys, wallet, charger | **negative `out_of_taxonomy`** | 5 | 15–20 s | any |
| 15 | Normal activity: sitting, reading, laptop use, walking | **negative `absence`** | 5 | 15–20 s | hall, bedroom |
| 16 | Detection-only context: bed, chair, cupboard, door, book, monitor | **negative `absence`** | 26 | 15–20 s | all rooms |
| | **Total** | | **100** | | |

**Positives: 39. Negatives: 61 (61 %).** Comfortably over the 30 % floor — which is the honest
consequence of §3, not padding. Most of what the brief called positives are `pre_dwell` negatives, and
they carry real test value.

Per-scenario accepted counts must each reach **4** (`min_clips_per_scenario`), which rows 1–8 satisfy.

## 7. Variation requirements

Do not collect 100 near-duplicates. Across the batch, vary:

- **≥ 3 households** (`h01`, `h02`, `h03` …) — the existing `min_houses` target
- **Lighting**: `daylight`, `tubelight`, `dim`, `night_flash`, `mixed` — dim and evening are where the
  detector is weakest and are the ones most often skipped
- **Camera position**: corner-mounted, shelf height, doorway. One camera per room, fixed — a hand-held
  pan produces footage the runtime will never see
- **Distance**: 1.5 m, 3 m, across-room
- **Person**: at least 3 different people across the batch
- **Object placement**: not the same corner of the same counter every time

Each negative must be shot in the **same room and lighting as its positive**, or it tests the lighting
rather than the confuser.

## 8. Five-member allocation

Each member owns a house or a set of rooms and works independently. Ids are pre-allocated so two
people cannot collide.

| Member | Scope | Clip id range | Clips | Key scenarios |
|:---|:---|:---|--:|:---|
| **1** | Kitchen + electrical | `h01_kitchen_s001_c001…c025` | 25 | SC-KIT-001/002 long-form; row 9 `pre_dwell`; charger/wire detection |
| **2** | Medical + hydration | `h02_bedroom_s001_c001…c018` | 18 | SC-MED-001 positives; bottle confusers (row 13) |
| **3** | Mobility + fall risk | `h02_hall_s002_c001…c018` | 18 | SC-BTH-001 wet floor; SC-MOB-001 long-form; assistive negatives |
| **4** | Bathroom + bedroom + general | `h03_bathroom_s001_c001…c020` | 20 | SC-BTH-002, SC-BTH-003; rows 10, 16 |
| **5** | Context + **dataset QA owner** | `h03_hall_s002_c001…c019` | 19 | Rows 14, 15, 16; runs the accounting report and owns acceptance |

Member 5 is the only one who sets `review_status: accepted`, and must not accept their own clips
without a second pair of eyes on them.

### Per-member checklist

1. Confirm a **`scenario-video`** consent record exists for your house. Image consent does not cover
   video. Two houses, two records.
2. `python scripts/scenarios/32_ingest_scenario_clips.py --init`
3. Film. Check length against §3 **before** deciding positive or negative.
4. Ingest each clip with its expectation:
   ```bash
   python scripts/scenarios/32_ingest_scenario_clips.py \
       --clip-id h01_kitchen_s001_c009 --source kitchen_cooking_01.mp4 \
       --scenario-id SC-KIT-001 --lighting tubelight \
       --consent-ref CONSENT-h01-2026-002 \
       --expect no-alert --negative-kind pre_dwell --annotator m1
   ```
5. `python scripts/scenarios/33_clip_dataset_report.py --exit-zero` to see progress.
6. Hand to member 5 for review. Do not set `accepted` yourself.

## 9. QA gates (Phase M)

Machine-checked at ingest — a clip cannot enter without passing these:

- clip id grammar · not already ingested · ffmpeg present · consent scope `scenario-video` ·
  consent not withdrawn · consent house matches · extension · file size · duration in `[10, 200]` s ·
  fps ≥ 15 · **metadata stripped and verified by read-back** · neutral handler name ·
  scenario exists in the compiled artifact · positive states a deadline · negative states a kind

Machine-checked at report time:

- ≥ 30 % negatives · ≥ 4 accepted per scenario · no unsanitised clips ·
  **no positive shorter than its scenario's temporal requirement** · licence complete for every
  external clip · consent complete for every own-capture clip

Human-checked, and the reason `review_status` exists:

- Is the scenario actually demonstrated, or merely the objects present?
- Is the polarity right — in particular, is a "positive" really `pre_dwell`?
- Faces of anyone who did not consent? Identity documents? Prescription labels? Screens showing
  personal messages?
- Adequate lighting, tolerable camera movement, relevant objects clearly visible?
- Is this a near-duplicate of a clip already accepted?

## 10. Accounting (Phase N)

```bash
python scripts/scenarios/33_clip_dataset_report.py --target 100
```

Writes `data/qa_reports/clip_dataset_report.json` with totals, polarity split and negative fraction,
per-scenario coverage with shortfalls, own-capture vs external, Indian-home count, privacy and consent
completeness, licence completeness, and the rejection list with reasons.

**The report counts `accepted` clips, not files.** 100 manifests with `review_status: pending` is
zero clips, and the report says FAIL. `test_pending_clips_do_not_count_toward_the_target` exists to
keep that true.

## 11. External sources (Phase H)

Permitted to *supplement* Indian-home collection, never to replace it — the dataset's purpose is
Indian homes, and `indian_home` is counted in the report.

Allowed licences: `CC0-1.0`, `CC-BY-3.0/4.0`, `CC-BY-SA-3.0/4.0`, `Pixabay-Content-License`,
`Pexels-License`, `public-domain`. Anything else is refused at manifest construction; add to
`ALLOWED_EXTERNAL_LICENCES` deliberately, after reading the terms, in a reviewed change.

Every external clip records `source_url`, `source_platform`, `creator`, `license`, `license_url`,
`download_date`, `original_video_id`, and the `start_time_s`/`end_time_s` trim window so a reviewer can
return to the source and see the same frames. **A publicly viewable video is not automatically
reusable.** If the licence cannot be established, the clip does not enter the dataset.

---

Related: [clip_capture_protocol.md](clip_capture_protocol.md) ·
[ADR-P6-10](adr/ADR-P6-10-clips-as-a-frozen-stage.md) · [negative_register.md](negative_register.md) ·
[scenario_taxonomy.md](scenario_taxonomy.md)

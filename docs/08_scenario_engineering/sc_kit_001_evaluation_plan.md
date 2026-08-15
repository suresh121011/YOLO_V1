# SC-KIT-001 Evaluation Plan — Dataset v0.1

**Branch:** `phase-6-sc-kit-001-evaluation` (to create from `phase-6-scenario-rule-engine` at `754b74f`)
**Date:** 2026-08-15
**Author:** Automated planning; awaiting human review

---

## 1. Repository Findings

### Git State (verified)
- **Branch:** `phase-6-scenario-rule-engine` at `754b74f`
- **Working tree:** Clean; only untracked `SC-KIT-001 — Kitchen Cooking Risk/`
- **Main:** Untouched at `6a2959c`
- **Phase-6 M0–M9:** All present (commits `2ce43c9` → `754b74f`)

### SC-KIT-001 Definition (source of truth)
From `configs/scenarios/SC-KIT-001.yaml`:
- **Trigger:** `all: [in_room(kitchen), present_for(stove, 60s), ever_seen(person), absent_for(person, 900s)]`
- **Risk:** CRITICAL, priority 5
- **Dwell:** `min_dwell_seconds: 120`
- **Total time to fire:** ≥ 1080 seconds (18 minutes)
- **Status:** `draft` (not `active`)
- **Required objects (derived):** stove, person
- **Optional:** knife, gas_cylinder

### Critical finding: ALL scenarios are `status: draft`
The `ScenarioRuleEngine` only loads `active` scenarios (line 155: `runnable = [s for s in authored if s.status is Status.ACTIVE]`). With zero active scenarios, the runtime raises `ScenarioRuntimeError`. The offline evaluator must handle this by loading scenarios directly, bypassing the status filter, while clearly documenting that evaluation runs against a draft scenario.

### Raw Clip Inventory
22 clips in `SC-KIT-001 — Kitchen Cooking Risk/` across three folders:

| Category | Count | Duration | FPS | Resolution | Source |
|:---------|:------|:---------|:----|:-----------|:-------|
| POSITIVE | 12 | 6.5–33.1 s | 24–59.9 | 768×432 to 2560×1440 | iStock/Pexels stock |
| NEGATIVE | 6 | 5.5–56.0 s | 24–59.9 | 768×432 | iStock stock |
| EDGE | 4 | 13.0–19.6 s | 24–25 | 768×432 | iStock stock |

### YOLO Model
Latest weights: `models/benchmarks/models/baseline_r0/weights/best.pt` (5.4 MB, YOLO11n, 23 classes)

---

## 2. Current Phase-6 State

Phase-6 M0–M9 complete. The infrastructure below exists and will be reused unchanged:

| Component | Location | Reuse |
|:----------|:---------|:------|
| ClipManifest + SourceProvenance + ExpectedOutcome | `src/scenario_engine/clips.py` | ✅ Full reuse |
| ClipRequirements + intake gates | `src/scenario_engine/clips.py` L344–491 | ✅ Full reuse |
| MP4 metadata strip + read-back | `src/scenario_engine/clips.py` L260–338 | ✅ Full reuse |
| probe_media (duration/FPS) | `src/scenario_engine/clips.py` L199–230 | ✅ Full reuse |
| verify_clip_consent | `src/scenario_engine/clips.py` L497–572 | ✅ Bypassed for external |
| validate_clip_set | `src/scenario_engine/clips.py` L1005–1066 | ✅ Full reuse |
| impossible_positives | `src/scenario_engine/clip_dataset.py` L157–189 | ✅ Full reuse |
| build_report + DatasetReport | `src/scenario_engine/clip_dataset.py` L192–303 | ✅ Full reuse |
| Ingest CLI | `scripts/scenarios/32_ingest_scenario_clips.py` | ⚠️ Extend for external |
| Dataset report CLI | `scripts/scenarios/33_clip_dataset_report.py` | ✅ Full reuse |
| ScenarioRuleEngine | `src/scenario_engine/runtime.py` | ⚠️ Need draft-mode bypass |
| EvalContext | `src/scenario_engine/context.py` | ✅ Full reuse |
| EventMemory | `src/pipeline/event_memory.py` | ✅ Full reuse (fresh per clip) |
| YOLODetector | `src/pipeline/detector.py` | ✅ Full reuse |
| evaluate_trigger | `src/scenario_engine/trigger.py` | ✅ Full reuse |
| Detection, BoundingBox, Alert | `src/pipeline/__init__.py` | ✅ Full reuse |
| 346 scenario-engine tests | `tests/unit/scenario_engine/` | ✅ Must remain green |

---

## 3. What Is Missing

| # | Gap | What to build |
|:--|:----|:-------------|
| 1 | Ingest CLI lacks `--external` mode | Extend `32_ingest_scenario_clips.py`: add `--external` flag + provenance args, skip consent-ref requirement |
| 2 | Batch ingest script | `scripts/scenarios/35_batch_ingest_external.py`: loops over clips, renames, calls ingest for each |
| 3 | Offline scenario evaluator | `scripts/scenarios/34_evaluate_scenario_clips.py`: YOLO → EvalContext → engine per clip, handles draft status |
| 4 | Expected-vs-predicted comparator | Part of evaluator output: compares manifest expectations vs actual engine output |
| 5 | Evaluation report generator | Generates `docs/08_scenario_engineering/sc_kit_001_evaluation_report.md` |
| 6 | v0.2 collection spec | Documents what a genuine SC-KIT-001 positive requires |

---

## 4. Naming Decision

**Decision: Use repository-canonical grammar `h{NN}_{room}_s{NNN}_c{NNN}`**

The `clip_collection_plan.md` §6 explicitly records the decision:

> The `SC-XXX-NNN_POS_NNN` style in the brief is **not** used, because the existing grammar is what makes extracted frames group as one leakage unit for free.

All existing infrastructure — `parse_clip_id()`, `ClipRequirements.validate_clip_id()`, the ingest CLI, the dataset report, the leakage unit extractor — depends on this grammar.

Convention for external/stock clips: **`h00`** as the external-footage house_id. This follows the pattern where `h99` is reserved for test fixtures.

```
h00_kitchen_s001_c001  through  h00_kitchen_s001_c012   (12 from POSITIVE folder)
h00_kitchen_s002_c001  through  h00_kitchen_s002_c006   (6 from NEGATIVE folder)
h00_kitchen_s003_c001  through  h00_kitchen_s003_c004   (4 from EDGE folder)
```

Session numbers separate collection categories so the distinction is traceable without embedding labels in filenames. The actual engine-truth labels are in the manifests.

**Rejected alternative:** `SC-KIT-001_POS_001` — would require modifying `clip_id_pattern` regex, `parse_clip_id()`, `validate_clip_id()`, and downstream tooling. Cosmetic improvement, higher breakage risk.

---

## 5. Ground-Truth Policy

### Separation of collection label from expected engine outcome

| Field | Meaning | Controlled by |
|:------|:--------|:-------------|
| `collection_label` | What the collector saw: positive/negative/edge | Human judgement |
| `polarity` | Engine expectation: positive (expects alert) / negative (expects silence) | Temporal analysis of the rule |
| `negative_kind` | Why it's negative | Rule structure |
| `expected.fires` | Does the engine fire? | Rule × clip duration |

For SC-KIT-001 with these clips:

| Collection folder | Engine polarity | Engine expected | negative_kind | Reason |
|:------------------|:---------------|:----------------|:-------------|:-------|
| POSITIVE | **negative** | fires: false | pre_dwell | Clip ≤ 33s; rule needs ≥ 1080s. Hazard composition present but temporal condition unmet |
| NEGATIVE | **negative** | fires: false | absence / confuser | Required objects absent or wrong context |
| EDGE | **negative** | fires: false | pre_dwell | Ambiguous composition; temporal condition unmet regardless |

**All 22 clips are engine-negatives.** This is the correct, honest classification. The `collection_label` goes in the `notes` field as context; the `polarity` and `expected` fields carry the engine truth.

---

## 6. Manifest Decision

Reuse existing `ClipManifest` schema exactly. Every field is already defined:

**Auto-extracted (no human input needed):**
- `clip_id`, `session_id`, `house_id`, `room` — from filename grammar
- `duration_s`, `fps` — from `probe_media()`
- `sha256` — from `compute_file_hash()`
- `metadata_stripped`, `ffmpeg_version` — from `strip_metadata()`
- `rule_hash_at_label_time` — from compiled artifact

**Human-provided:**
- `scenario_id` — SC-KIT-001 (same for all)
- `polarity` — negative (same for all, given temporal analysis)
- `negative_kind` — pre_dwell / absence / confuser (per-clip judgment)
- `lighting` — per-clip observation
- `notes` — collection_label, content description
- `provenance` — source_type: external, platform, URL, license
- `review_status`, `reviewed_by` — after human QA

---

## 7. Privacy Policy

**Reuse existing MP4 metadata stripper.** No changes to the privacy pipeline:
- `strip_metadata()` runs `ffmpeg -y -i <src> -map_metadata -1 -map_chapters -1 -fflags +bitexact -c copy <dst>`
- Read-back assertion via `FORBIDDEN_METADATA_KEYS` + `NEUTRAL_HANDLER_NAMES`
- All clips are stock footage (no household GPS risk), but stripping is still applied because it is a hard gate

**Audio:** Stock footage audio is not needed for YOLO evaluation. The existing stripping command preserves audio. Adding `-an` is a minor improvement but changes the existing `strip_metadata()` function signature. Proposed: document that audio is present but unused; defer `-an` to a separate change if desired.

---

## 8. Licensing Policy

External clips use `SourceProvenance(source_type="external")`.

Allowed licences (from `ALLOWED_EXTERNAL_LICENCES`):
`CC0-1.0`, `CC-BY-3.0`, `CC-BY-4.0`, `CC-BY-SA-3.0`, `CC-BY-SA-4.0`, `Pixabay-Content-License`, `Pexels-License`, `public-domain`

**Issue:** iStock clips may not carry a licence from this allowlist. The `SourceProvenance.from_mapping()` will reject them. Options:
1. If clips were purchased with a valid iStock licence → add `iStock-Standard` to `ALLOWED_EXTERNAL_LICENCES` (requires deliberate review)
2. If clips are Pexels → `Pexels-License` is already allowed
3. If licence is undetermined → clips enter as `review_status: pending` with a licence-gap note, and are **not included in the frozen dataset**

**Licensing is a hard freeze gate.** Unlicensed clips do not enter Dataset v0.1.

---

## 9. Validation Gates

### Reused from M7 (no new code)
| Gate | Implementation | Status |
|:-----|:---------------|:-------|
| File: readable, valid extension, valid size | `ClipRequirements.check_file()` | ✅ Exists |
| Media: duration in [10, 200]s, FPS ≥ 15 | `ClipRequirements.check_media()` | ✅ Exists |
| Naming: clip_id grammar | `ClipRequirements.validate_clip_id()` | ✅ Exists |
| Privacy: metadata stripped | `strip_metadata()` + read-back | ✅ Exists |
| Manifest: completeness, polarity rules | `ClipManifest.from_mapping()` | ✅ Exists |
| Set: no duplicates, ≥ 30% negatives | `validate_clip_set()` | ✅ Exists |
| Temporal: positive not shorter than rule | `impossible_positives()` | ✅ Exists |
| External: licence from allowlist | `SourceProvenance.from_mapping()` | ✅ Exists |
| Review: accepted requires reviewer | `ClipManifest.from_mapping()` L880 | ✅ Exists |
| Staleness: rule_hash match | `ClipManifest.is_stale()` | ✅ Exists |

### Clips that may fail existing gates
- **FPS < 15:** One clip at 24 FPS passes (≥ 15), but some are at 24/25 which is fine. 59.9 FPS also passes.
- **Duration < 10s:** `istockphoto-2161844640` at 6.5s and `istockphoto-1474772225` at 5.5s **will fail** `min_duration_s: 10.0`. These clips will be documented as rejected by the intake gate.
- **All clips pass the positive-shorter-than-rule check** because none are positives.

### New validation (add during implementation)
- Orphan check: manifest without video file, video without manifest
- Duplicate SHA256 detection across clips

---

## 10. Evaluation Architecture

### The offline evaluator (`scripts/scenarios/34_evaluate_scenario_clips.py`)

The evaluator must solve two problems the live runtime does not face:

**Problem 1: All scenarios are `status: draft`.**
The `ScenarioRuleEngine.__init__()` raises `ScenarioRuntimeError` when zero scenarios are active. For offline evaluation, the evaluator loads the scenario directly via `load_scenario_files()`, selects SC-KIT-001, and evaluates its trigger via `evaluate_trigger()` against an `EvalContext` — the same predicate path the runtime uses. The state machine (dwell, cooldown, hysteresis) is replicated from the runtime's `_evaluate_one()`.

**Problem 2: EventMemory isolation.**
Each clip gets a **fresh** `EventMemory` instance. Cross-clip state contamination would create false temporal events. The evaluator enforces:

```
for clip in accepted_clips:
    memory = EventMemory(window_size=int(clip.fps * 20))  # scale to actual FPS
    # Load scenario fresh (draft OK for evaluation)
    scenario = load_scenario("SC-KIT-001")
    state = EventState()  # fresh state machine
    
    for frame_idx, bgr_frame in read_frames(clip):
        detections = detector.detect(bgr_frame, frame_id=frame_idx)
        memory.update(detections)
        context = EvalContext(
            detections=tuple(detections),
            memory=memory,
            fps=clip.fps,
            frame_id=frame_idx,
            room="kitchen",
        )
        condition_true = evaluate_trigger(scenario.trigger, context)
        # Apply state machine: dwell, cooldown, etc.
        record_decision(frame_idx, condition_true, state)
    
    save_clip_result(clip.clip_id, decisions, alerts)
```

### FPS correctness
The evaluator passes the clip's actual FPS to `EvalContext.fps`. Temporal predicates (`present_for`, `absent_for`) use this to convert frame counts to seconds. A 60 FPS clip produces twice the frames of a 30 FPS clip but the same number of seconds.

`EventMemory.window_size` is scaled to `clip.fps × 20` to ensure at least 20 seconds of temporal history, matching or exceeding the live runtime's default (150 frames ≈ 10s at 15 FPS).

---

## 11. Metrics

### What can be measured with Dataset v0.1

Since all 22 clips are engine-negatives (expected: no alert):

| Metric | Measurable? | Expected value |
|:-------|:-----------|:---------------|
| True Negatives | ✅ | 22 (if engine stays silent on all) |
| False Positives | ✅ | 0 (any alert on a negative clip) |
| False Positive Rate | ✅ | 0.0 (FP / (FP + TN)) |
| True Positives | ❌ | No genuine positives exist |
| False Negatives | ❌ | No genuine positives exist |
| Precision | ❌ | Undefined (no TP or FP expected) |
| Recall | ❌ | **Cannot be claimed** |
| F1 | ❌ | **Cannot be claimed** |

### What must NOT be claimed
- **Do not report recall or F1.** There are no true positives.
- **Do not claim sensitivity.** The dataset tests specificity only.
- State explicitly: `true_positive_coverage: 0`

### Additional YOLO-level metrics (per clip)
- Classes detected per frame
- Confidence distributions
- Object presence/absence timeline
- Detection consistency across frames

---

## 12. True-Positive Limitation + v0.2 Specification

### Why v0.1 has no true positives
SC-KIT-001 requires:
1. `present_for(stove, 60s)` — stove visible for 60s
2. `ever_seen(person)` — person seen at least once
3. `absent_for(person, 900s)` — person gone for 15 minutes
4. `min_dwell_seconds: 120` — condition must hold for 2 minutes

**Minimum clip duration for a true positive: ~1080 seconds (18 minutes)**

The longest collected clip is 56 seconds.

### v0.2 collection specification

To produce a genuine SC-KIT-001 positive, a clip must follow this timeline:

```
T = 0s:       Person + stove visible (kitchen scene established)
T = 0–60s:    Person visible, stove visible (present_for(stove, 60s) accumulates)
T = 60s:      Person leaves frame
T = 60–960s:  Stove visible, person absent (absent_for(person, 900s) accumulates)
T = 960s:     absent_for(person, 900s) first becomes true
T = 960–1080s: min_dwell_seconds accumulates (all conditions true for 120s)
T = 1080s:    ENGINE FIRES — first alert expected

Expected result: Alert fires at approximately T = 1080s (±120s for dwell)
```

**Recording requirements for v0.2:**
- Duration: ≥ 1100 seconds (≈ 18.5 minutes)
- Camera: Fixed, stationary (not hand-held)
- Content: Kitchen with stove visible throughout
- Action: Person cooks briefly, then leaves kitchen entirely
- Variations: ≥ 3 different kitchens, different lighting (daylight, tubelight, dim)
- Count: ≥ 4 clips (minimum per-scenario coverage)

---

## 13. Council-Substitute Review

### A. Architecture / Repository Integration

**Assessment: Sound.** The M7 infrastructure covers nearly everything. The three gaps (external ingest CLI mode, offline evaluator, comparator) are genuinely missing and cannot be faked with existing tools. The draft-status issue is the trickiest: the evaluator must bypass the runtime's active-only filter without modifying the runtime itself. Loading the scenario directly and calling `evaluate_trigger()` with a manual state machine is the correct approach — it tests the same predicate path without changing production code.

**Risk: EventMemory window_size.** The default 150 frames at 15 FPS gives 10s of history. At 60 FPS, that's only 2.5s. For clips with high FPS, the window must be scaled. Since SC-KIT-001 needs 900s of absence, the EventMemory's `frames_since_seen()` method uses the total frame counter (not the window), so this is safe for `absent_for`. The window only affects `consecutive_frames()`, which SC-KIT-001 does not use.

### B. Dataset / Evaluation / Data Governance

**Assessment: Honest.** The ground-truth policy correctly separates collection labels from engine truth. All clips being engine-negatives is the honest consequence of a 960s temporal requirement and short clips — the `clip_collection_plan.md` §3 anticipated this exact situation. Claiming recall without true positives would be a methodological fabrication.

**Risk: Sample size.** 22 clips is too small for meaningful statistical inference. Report raw counts with explicit n=22 caveat. A false positive rate of 0/22 is different from a false positive rate of 0/1000.

**Risk: Some clips may be rejected by intake gates** (duration < 10s). This reduces the evaluated set further. Document rejections with reasons.

### C. Safety / Privacy / Clinical Risk / False-Positive Analysis

**Assessment: Adequate.** Stock footage has no household GPS risk, but the stripping gate is still applied (correct — it's a hard gate, not optional). External clips cannot fabricate consent (the schema enforces this). The iStock licensing issue is a genuine blocker for distribution but not for internal evaluation.

**Risk: Generalisability.** Stock footage kitchens are not Indian home kitchens. Detection performance may differ significantly on real deployment scenes (different lighting, stove types, gas cylinder placement, camera angles). The evaluation report must clearly state this limitation.

**Risk: Draft status.** Running the evaluation against a draft scenario means the evaluation results are preliminary. If the scenario is modified during clinical review before becoming active, the evaluation results become stale (detectable via `rule_hash`).

---

## 14. Rejected Alternatives

| Decision | Chosen | Rejected | Why |
|:---------|:-------|:---------|:----|
| Clip naming | `h00_kitchen_s001_c001` | `SC-KIT-001_POS_001` | Repository ADR, breaks existing tooling |
| Evaluator | Standalone script with manual state machine | Modify `ScenarioRuleEngine` to accept drafts | Production code should not change for evaluation |
| Draft handling | Load scenario directly, evaluate trigger manually | Set `status: active` temporarily | Would bypass the clinical review gate |
| Consent | External clips use licence | Fabricate consent records | Explicitly forbidden in schema |
| Metrics | Raw counts, no recall/F1 | Report recall from negatives | Methodologically invalid |
| Audio | Preserve (document as unused) | Strip with `-an` | Requires modifying `strip_metadata()` |

---

## 15. Implementation Milestones

### M1: Dataset inventory + manifest preparation
- Create branch `phase-6-sc-kit-001-evaluation`
- Probe all 22 clips (metadata extraction)
- Create clip_id mapping (stock filename → canonical id)
- Extend ingest CLI for external clips (add `--external` + provenance args)
- **Commit:** `dataset-plan-and-manifest`

### M2: Privacy/license/consent validation
- Rename clips to `data/clip_inbox/`
- Run `strip_metadata()` on each clip
- Validate licensing status
- Batch ingest via extended CLI
- Identify and document rejected clips (duration < 10s)
- **Commit:** `clip-validation`

### M3: Ground-truth + dataset freeze
- Human review: assign polarity, negative_kind per clip
- Set `review_status` for reviewed clips
- Run `validate_clip_set()` and `build_report()`
- Generate dataset v0.1 manifest
- **Commit:** `ground-truth-and-freeze`

### M4: YOLO batch inference
- Run YOLO on every accepted clip
- Save per-frame detection JSON
- Record YOLO detection statistics
- **Commit:** `baseline-evaluation-yolo`

### M5: Scenario engine offline evaluation
- Create `scripts/scenarios/34_evaluate_scenario_clips.py`
- Replay detections through scenario trigger + state machine
- Record per-frame decisions
- Verify EventMemory reset between clips
- Verify FPS-correct temporal evaluation
- **Commit:** `baseline-evaluation-engine`

### M6: Expected vs predicted + failure analysis
- Compare manifests vs evaluation results
- Generate evaluation CSV
- Classify any failures by root cause
- Generate baseline metrics
- **Commit:** `evaluation-analysis`

### M7: Report + v0.2 spec
- Generate final evaluation report
- Document true-positive v0.2 collection specification
- **Commit:** `evaluation-report`

### M8: Tests + lint
- Add new tests for external ingest, evaluator, comparator
- Verify all existing tests green
- Run ruff, black, mypy
- **Commit:** (rolled into relevant milestones)

---

## 16. Risks

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| iStock licence not in allowlist | Clips cannot enter frozen dataset | Document limitation; enter as `pending` |
| Clips rejected by duration gate (< 10s) | Reduced dataset size | Document rejections; 20/22 clips likely pass |
| All scenarios are draft | Runtime refuses to load | Evaluator uses direct trigger evaluation |
| Stock footage ≠ Indian home | Results don't generalise | Document limitation prominently |
| No true positives | Cannot measure sensitivity | v0.2 spec with 18-min recording procedure |
| EventMemory contamination | False temporal events | Fresh memory per clip; test enforces it |
| FPS variation (24–60) | Temporal miscalculation | Pass actual FPS to EvalContext |

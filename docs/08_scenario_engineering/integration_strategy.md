# Integration Strategy — YOLO · Tracking · VLM · Voice · Caregiver

**Milestone:** Phase 6 M9 · **Status:** complete
**Acceptance:** *no further code changes are required when the model lands.*

## Why this document exists, and why it is not only a document

The knowledge layer (M0–M8) was built against a system with **no trained model**. Everything downstream
of the detector — the scenario engine, the alert queue, the TTS sink — has therefore only ever been
exercised against synthetic detections. The risk this milestone addresses is not that any single
component is wrong; it is that the *seams between them* were never load-bearing, so nobody found out
what was missing.

That risk is not answerable with prose. "No code changes are required" is unfalsifiable as a sentence,
so M9 ships it as a command:

```bash
python scripts/qa/model_landing_check.py
```

| Check | What it proves |
|:--|:---|
| L1 | The weights file exists and loads |
| L2 | `configs/data.yaml` declares a taxonomy at all |
| L3 | The weights carry **exactly** that taxonomy — same names *and* same ids |
| L4 | Every class the authored scenarios need is detectable and enabled |
| L5 | At least one scenario is active and flag-enabled (the engine can start) |
| L6 | The pipeline assembles end to end through the composition root |

A check ends in `pass`, `fail`, `blocked` or `skipped`. **`blocked` means the engineering is finished
and a human step is outstanding** — it is kept distinct from `fail` so that a pending clinical review
never reads as a broken pipeline, and a broken pipeline can never hide behind a pending clinical review.
`tests/unit/test_model_landing_check.py` pins that distinction, because a landing check that cannot fail
is worse than no check: it certifies the thing it never tested.

Run today, against a repository with no weights, it reports:

```
L1 [FAIL]     Weights file exists                 (no model has been trained yet)
L2 [PASS]     configs/data.yaml declares a taxonomy        23 classes
L3 [SKIPPED]  Weights carry exactly the declared taxonomy  needs L1
L4 [PASS]     Every class the scenarios need is usable     fingerprint recorded
L5 [BLOCKED]  At least one scenario is active              9 draft, 0 active
L6 [SKIPPED]  The pipeline assembles                       needs L1
```

That is the honest picture. Two skips and a block, waiting on a model and a clinician — not on code.

---

## What auditing the five seams actually found

Writing this document meant reading each seam rather than describing it, and the seams were in worse
condition than the components. Five defects, all pre-existing, numbered continuing from the thirteen
found earlier in Phase 6:

| # | Defect | Consequence |
|:--|:---|:---|
| **14** | `Alert.patient_facing` was documented as "when False, nothing is spoken to the resident" and enforced **nowhere** | **7 of the 9 scenarios are `patient_facing: False`.** All seven would have been announced aloud to the resident, including `SC-BTH-002`, whose entire design is that it never speaks to them. A quiet-hours `caregiver_only` scenario would have spoken at 3am — the precise behaviour that setting exists to prevent |
| **15** | `Alert.caregiver_channel` had no consumer, and — contrary to ADR-P6-09's claim that it was "carried and logged" — was not logged either | All 9 scenarios declare a channel. The fourth stage of the product output went nowhere and left no trace that it had |
| **16** | `runtime.smolvlm_every_n_frames` and `runtime.smolvlm_timeout_ms` were dead: the interval was hard-coded to `5` in `process_frame`, and the timeout was stored by the analyzer and never consulted | The VLM's entire control surface was decorative |
| **17** | Nothing checked the loaded weights' class list against `configs/data.yaml` | The model-landing failure mode itself: weights trained on a different or renumbered class list run happily and reason about the wrong world |
| **18** | Eight of the ten `components:` flags were read by no code at all | `configs/feature_flags.yaml` opens "All runtime behavior is controlled here. No code changes required." An operator setting `tts_output: false` was still spoken to |

Defects 14 and 15 were **masking each other**, which is why neither was visible from either side. With
everything spoken and nothing routed, the device appeared to work. Fixing 14 alone would have silenced
seven of nine scenarios completely — the sink from 15 is what makes honouring `patient_facing` safe
rather than merely quiet.

Defect 18 is defect 1 wearing a different hat. M1 fixed how the flag file is *loaded*; nobody then
checked that each individual flag was *consumed*. `tests/unit/test_feature_flags_are_live.py` now
requires every flag to be either wired or explicitly labelled `NOT WIRED`, in the YAML and in the test's
`UNWIRED` map. Adding an inert flag fails; wiring one and forgetting to unlabel it also fails, so the
labels cannot rot into pessimism either.

---

## 1 · YOLO — the model

**Seam:** `YOLODetector(model_path=..., expected_classes=...)`, constructed by
`src/app/factory.py::build_pipeline`.

**Ready.** The detector already applies per-class thresholds correctly (M1 fixed the pre-filter that
made the low safety thresholds unreachable), suppresses disabled classes before they become a
`Detection`, and normalises boxes to the frame-relative `cx/cy/w/h` the spatial predicates expect.
Weights load as `.pt`, `.onnx` or `.tflite` through the same Ultralytics call.

**Added in M9.** `expected_classes` and `TaxonomyMismatchError`. `SystemConfig` now carries
`class_names` from `configs/data.yaml`, so the detector still reads no YAML itself — that is
`SystemConfig`'s own documented contract.

The check compares **ids as well as names**, which is the part that is easy to get wrong. The R24
decision demotes `wet_floor` to scene level *with class id 20 reserved and no renumbering*, so ids are
load-bearing across the whole system; a model that shuffles ids while keeping every name would
otherwise pass and mislabel every detection it makes. Names matter for the same reason from the other
direction: the scenario engine's inverted index, its per-class confidence floors, the capability map,
and the privacy suppression of `passport` are all keyed by name.

**When the model lands, in order:**

1. Place the weights at `models/yolo11n/weights/best.pt` (or pass `--model`).
2. `python scripts/qa/model_landing_check.py` → expect L1–L4 green.
3. If L3 fails, the fix is in **training**, not here: retrain or re-export against `configs/data.yaml`.
   Do not "translate" class names at inference — a rename map is a second taxonomy, and
   ADR-P6-05 exists because the repo already carries eight partial duplicates of the 23-class list.
4. Set `expected_hash` on the detector once the weights are the release artifact, so a corrupted or
   swapped file is caught by more than its filename.

**Explicitly not in this seam:** accuracy. `model_landing_check.py` proves the model *fits*, never that
it is *good*. mAP, per-class recall and the confusion matrix belong to the evaluation suite.

---

## 2 · Tracking — the seam that does not exist, and must not be pretended into existence

**Status: deliberately absent. No code, no stub, no placeholder.**

Nothing in the system tracks identity across frames. `EventMemory` counts *classes*, not instances:
"a person has been present for 40 seconds" is really "the class `person` has been detected in
consecutive frames for 40 seconds". With two residents, one leaving and one arriving, that statement
stays true while the premise behind it — that this is the same person — silently fails.

Every current scenario is written to survive this, and that is a design constraint, not an accident:

* `absent_for(person, N)` — no identity needed; nobody at all is a class-level fact.
* `present_for(person, N)` — the dwell claims *someone* has been there, and every authored message says
  so. `SC-BTH-003` speaks of "time in the bathroom", never "you have been in the bathroom".
* `near(person, stove)` — geometric, per-frame, identity-free.

**What tracking would unlock, and what it would cost.** Re-identification is what "*the resident* has
not moved in two hours" needs, and it is the one honest route to several rejected scenarios. It is out
of scope here for a specific reason: `Detection` is a LOCKED contract, and a `track_id` would be an
additive change under its own ADR — but the real cost is not the field. It is that a track id is a
claim about *sameness of a person*, which is a stronger claim than anything this system currently
makes, and one that a 23-class static-object detector cannot support on its own. The
[negative register](negative_register.md) covers what this means for fall detection, wandering and
person identification.

**The rule for anyone adding it later:** a tracker arrives with the scenarios that need it and the
evidence that it is reliable enough for them, or it does not arrive. A `track_id` that is right 80% of
the time is worse than no tracking, because the scenarios written on top of it will assume 100%.

---

## 3 · VLM — SmolVLM2

**Seam:** `SmolVLM2Analyzer.analyze(frame, detections, frame_id) -> SceneContext | None`, fused into
detections by `ConfidenceFusion`, then passed to `evaluate(..., context=...)`.

**Ready.** Graceful degradation is real: a load failure, an inference failure or an unparseable response
all return `None`, and the pipeline runs YOLO-only. The safety invariant is real too — fusion can only
ever *raise* a confidence (`min(1.0, max(det.confidence, fused))`), so the VLM cannot cancel an alert.
`ScenarioRuleEngine` treats `SceneContext` as advisory: it is recorded in the explanation and gates no
decision.

**Fixed in M9.** The sampling interval now comes from `runtime.smolvlm_every_n_frames` instead of a
hard-coded `5`, and `smolvlm_timeout_ms` is finally applied — a result that overran its budget is
**discarded** rather than returned.

That discard is a post-hoc check, not a cancellation, and the docstring says so: nothing here can
interrupt `model.generate` mid-call, and pretending otherwise would be the more misleading design. What
it does guarantee is that a description of the world several seconds ago is never fused into the
present, where it would raise confidence on detections it did not see. Degrading to YOLO-only is the
safe direction precisely because fusion only ever increases confidence.

**Known limitation, unresolved on purpose.** `analyze` runs **synchronously on the main thread**. On a
Pi, a 3-second VLM call blocks detection for 3 seconds — and `performance_budget.md` allocates the
whole frame far less than that. The mitigations available today are the interval and the flag
(`smolvlm_analysis: false` by default). Moving the VLM to its own thread with a
most-recent-result-wins buffer is the right fix, and it needs a measured budget on the real device
rather than a guess made here.

---

## 4 · Voice — Piper TTS

**Seam:** `PiperTTS.speak(text, priority)`, fed by `_next_speakable_alert()`.

**Ready.** `AlertQueue` arbitrates by severity, the backlog survives across frames, `max_alerts_per_minute`
is enforced with CRITICAL bypassing the cap, and queue overflow evicts the lowest-priority *queued*
message rather than the arriving one (all M1).

**Added in M9 — this is the seam that was most wrong:**

* **`patient_facing` is honoured** (defect 14). Non-speakable alerts are *skipped* rather than left in
  the queue, so one silent alert cannot gag a speakable one behind it, and they do not consume the
  speech budget. The caregiver sink has already received them by that point.
* **Language selection is real.** `runtime.tts_language` picks from a scenario's open-ended `messages`
  map (ADR-P6-09), with fallback to English rather than to silence: a resident who hears the wrong
  language still hears a warning.
* **The voice file wins a disagreement with the config.** Feeding Devanagari to an `en_IN` voice does
  not fail — it produces confident gibberish at someone who may be alone. `resolve_speech_language`
  compares the configured language against the loaded voice's filename tag and logs the mismatch.
* **`tts_output: false` genuinely silences the device** (defect 18).
* **`tts_speed` is wired**, and `configs/feature_flags.yaml` now reads `0.9` rather than `1.0` — 0.9 was
  Piper's own default and therefore the rate the device has always actually spoken at. Wiring a dead
  key must not quietly change behaviour on the way past.

**Not wired, and labelled as such:** `tts_volume`. Piper writes a WAV to a player this code does not
always own (`sounddevice` when present, `aplay`/PowerShell otherwise), so a level honoured on one path
and ignored on the other is less predictable than none. The OS mixer is the volume control.

**When a Hindi or Marathi voice lands:** drop the `.onnx` and `.onnx.json` into `models/tts/`, point
`tts_model_path` at it, set `runtime.tts_language`, and add that language key to each scenario's
`messages`. No code change. A scenario missing the key logs a warning and speaks English.

---

## 5 · Caregiver

**Seam:** `BaseCaregiverSink` (Protocol, `src/pipeline/__init__.py`), defaulting to
`LocalCaregiverSink` → `logs/caregiver.jsonl`.

ADR-P6-09 added `caregiver_channel` with no consumer and recorded that "M9 specifies the sink". This is
that sink, and it is **local only**: no network, no push service, no telephony.

**Why local rather than none.** A field that nothing reads is indistinguishable from a feature that
works. Every one of the nine scenarios declares a channel, and seven are `patient_facing: False` —
with `patient_facing` now honoured and no sink, those seven would produce *nothing observable
anywhere*. A local record makes the stage testable now.

**Why local rather than remote.** A caregiver transport is a product with its own consent surface,
retention policy, failure semantics, and an identity for the caregiver — none of which exist. Writing a
plausible one would manufacture the appearance of an escalation path that nobody receives.

**`push_and_call` is honest about its own impotence.** No telephony exists, so such an alert is recorded
with `escalation_pending: true` and logged at WARNING ending `NOBODY HAS BEEN CALLED`. The system's
stated non-negotiable is that escalation terminates at a human; when it cannot reach one, it must say so
loudly rather than write a line that reads as delivered.

Three further behaviours, each pinned by a test:

* `notify()` returns `False` **only** for `channel: none`, so "the caregiver was not told about this
  one" is distinguishable from "a channel was requested and nothing happened".
* An unrecognised channel is **delivered**, not dropped, and marked `unknown_channel`. Over-notifying a
  caregiver is recoverable; a silently withheld alert is not.
* Records carry class *names*, never bounding boxes. `StructuredLogger` redacts person/face geometry
  from frame logs, and a second alert-shaped log leaking it would reopen that hole in a different file.

**When a real caregiver app arrives:** implement `BaseCaregiverSink` in a package above `src.pipeline`,
pass it to `build_pipeline(caregiver_sink=...)`, and flip `caregiver_sync`. `LocalCaregiverSink` stays
as the offline fallback — a device on a dropped WiFi link must still record what it would have sent.

---

## What still requires code, stated plainly

The acceptance criterion is about the *model* landing. These are the things that will still need code
when other pieces land, so that nobody reads the criterion more broadly than it is meant:

| Trigger | Work required |
|:---|:---|
| A caregiver app or push service | A remote `BaseCaregiverSink` implementation. The seam exists; the transport does not |
| The VLM on real hardware | Moving `analyze` off the main thread once the device gives a measured budget |
| Scenarios that need identity | A tracker, a `track_id` on `Detection` under its own ADR, and the evidence that it is reliable enough |
| Field deployment | `rule_hot_reload`, `thermal_monitoring`, `active_learning_logging` — all labelled NOT WIRED |
| A second camera | `room` is per-pipeline today; two rooms means two pipelines and an arbitration policy between them |

## Verification

```bash
python scripts/qa/model_landing_check.py          # the M9 acceptance check
python scripts/qa/validate_phase6.py              # the M6 correctness gate — must stay 8/8
pytest tests/unit/pipeline tests/unit/scenario_engine
pytest tests/unit/test_feature_flags_are_live.py  # no flag may silently do nothing
pytest tests/unit/test_model_landing_check.py     # the landing check can still fail
```

Related: [ADR-P6-04](adr/ADR-P6-04-rule-engine-injection.md) ·
[ADR-P6-05](adr/ADR-P6-05-class-capability-map.md) ·
[ADR-P6-09](adr/ADR-P6-09-alert-contract-extension.md) ·
[ADR-P6-11](adr/ADR-P6-11-caregiver-sink-is-local-only.md) ·
[negative_register.md](negative_register.md)

# Domain Research Report — Elderly Home Safety

## Purpose

Phase 3 deliverable. Summarises the authoritative guidance behind the scenario taxonomy and translates
each finding into an engineering requirement. Its operational output is the `evidence` field contract:
what a scenario must cite, and what a citation does *not* license.

## Dependencies

Reads:
- `negative_register.md`

Used By:
- `requirements_specification.md`
- `src/scenario_engine/validators/` (evidence-strength gate)
- `configs/scenarios/SC-*.yaml` (`evidence` blocks)

---

## 1. Frameworks that define scope

**Katz ADL** (1963) — bathing, dressing, toileting, transferring, continence, feeding. **Lawton–Brody
IADL** (1969) — telephone use, shopping, food preparation, housekeeping, laundry, transport,
**medication management**, finances.

These are the right vocabulary for deciding what is *in scope*, and they are not detection targets. The
critical engineering consequence is inverted from the obvious reading: **food preparation and medication
management are IADLs to be preserved, not interrupted.** A system that warns an older adult about the
knife they are cooking with, or repeatedly asks whether they took their medicine, is degrading the exact
capacities the framework says to protect. This is the clinical basis for deleting `knife_near_person` and
`medicine_reminder` rather than tuning them.

> **Requirement.** Scenarios that interrupt a preserved ADL/IADL require a materially higher evidence bar
> than environmental findings, and default to caregiver-facing rather than patient-facing.

## 2. Falls and the home environment

Roughly 28–35% of people aged 65+ fall each year, rising to 32–42% above 70. The highest-risk areas
inside the home are consistently the bedroom, bathroom, and stairs; the most frequently observed hazards
are absent bathroom grab rails, absent non-slip surfaces in showers, single-sided stair rails, and no
light switch reachable from the bed.

The CDC's STEADI *Check for Safety* checklist walks the home room by room — floors, stairs and steps,
kitchen, bedroom, bathroom — and prescribes concrete one-time fixes: remove or secure throw rugs, coil or
tape cords next to the wall, install grab bars beside and inside the tub and next to the toilet, add
non-slip strips, improve lighting and add night lights, keep frequently used items within reach.

Two observations drive the design:

- **The prescribed intervention is almost always a one-time environmental modification by a caregiver**,
  not a real-time warning to the resident. That is why cord-on-floor and missing-grab-bar become
  caregiver-facing assessment findings rather than spoken alerts.
- **Several high-value STEADI items are unsupportable here.** Stairs are a STEADI item and there is no
  `stairs` class (though `staircase` is already a capture room). Lighting is a STEADI item and no class
  encodes light state. These belong in the negative register, not in aspirational scenarios.

> **Requirement.** A scenario's `evidence` must record `supports: hazard | intervention | detection_method`.
> STEADI backs the *hazard*; nothing in the literature backs our *detector*. `supports: hazard` therefore
> never licenses `detectability: direct`. This separation blocks the most common self-deception in this
> kind of work — citing a guideline to legitimise a detection you cannot actually make.

## 3. Alarm fatigue is the dominant failure mode

The monitor-alarm literature documents non-actionable alarm rates of 74–99%, with desensitisation and
delayed response as the consequence. Telecare data shows the same shape: unused social alarms rose from
17.7% to 23.5% over twelve months, with *decreasing* perceived safety among users.

Applied to the six current rules, projected day-one volume is ~370 alerts with essentially none
actionable. The endpoint is not annoyance; it is a device that is muted or unplugged, which is a 100%
false-negative rate. **An ignored safety system is more dangerous than no system**, because it displaced
the human vigilance that would otherwise exist.

> **Requirement.** Cooldown is not fatigue control — a cooldown on a permanently-true condition is a
> metronome. Mandatory instead: dwell time, an explicit clear condition, an event state machine,
> escalation rather than repetition (capped), three separate budgets (per-scenario cooldown, global rate
> limit, per-scenario daily budget), quiet hours, and context-based suppression.

> **Requirement (the non-obvious one).** Dwell time and recall are *not* in tension for slow hazards. A
> wet floor persists for minutes; an unattended stove for minutes. Requiring 3 s or 120 s of sustained
> detection costs almost no true positives while eliminating flicker-driven false positives. **Spend the
> precision budget on dwell, not on raising confidence thresholds** — which lets detection thresholds stay
> recall-tuned *and* the device stay quiet. This refines the project's existing "prioritise recall over
> precision" posture into: recall at the *detector*, precision at the *scenario*.

## 4. Human factors in patient prompts

Guidance for speaking to older adults, several points of which contradict the current message set:

- **Lead with the observation, then one action.** "The floor near the sink looks wet" before any
  instruction. Name the *place* — an unlocalised warning cannot be acted on.
- **Short.** ≤12 words, ≤6 seconds spoken. Front-load the important word: presbycusis costs
  sentence-final content first.
- **Never startle.** "Careful!", "Watch out!", "Danger" — startle is itself a fall mechanism. This is not
  a stylistic preference.
- **Never ask a question the system cannot hear the answer to.** There is no ASR.
- **Never state what cannot be observed.** `configs/risk_rules.yaml:54` asserts the stove "appears to be
  on".
- **Never imply surveillance.** "I can see your medicine" (`:63`) announces that the device is watching
  the resident's health items.
- **Preserve agency and avoid nagging or infantilising.** No "again", "remember", "don't forget".
- **Vary wording within an escalation ladder, but keep the opening utterance identical across days** —
  predictability is the accommodation for cognitive impairment; monotony within one event is the irritant.

**Localisation.** Hindi prompts to an elder must use आप with honorific-plural verbs (the current strings
do this correctly — worth locking as a validator). Code-mixing is the natural register: urban Indian
elders use "gas cylinder", "regulator", "medicine", "bathroom" as loanwords, and insisting on शुद्ध हिन्दी
reduces comprehension. Hindi alone is also insufficient coverage for the target population.

**Accessibility.** This is a speech-only device, which is an accessibility single point of failure for a
population with high presbycusis prevalence. A low-frequency attention tone before speech, per-severity
volume, and a visual channel are required; the existing 880 Hz beep is a *failure* indicator and must not
be overloaded as an attention cue.

> **Requirement.** Message linting is a validator, not a review convention: banned-assertion lexicon,
> banned-startle lexicon, word-count bound, honorific check, Devanagari presence check, and EN/HI action-
> clause parity.

## 5. Regulatory and privacy posture

**Non-medical-device positioning.** The line is crossed by claiming detection, diagnosis, prevention, or
monitoring of a disease or condition. Excluded permanently: fall detection, emergency detection,
medication-adherence monitoring, hydration monitoring, health-status assessment, and the verbs "prevents",
"protects", "ensures", "guarantees". A capability statement and a first-run caregiver acknowledgement are
required in the product surface, not only in a document.

**India's DPDP Act 2023.** The personal/domestic exemption sits in the applicability section and protects
*the household*, not the vendor. A company building, shipping, and supporting this product — and
especially one ingesting deployment frames for training — is doing commercial processing as a Data
Fiduciary, and the exemption does not apply. The DPDP Rules 2025 phase in through to 2027, so treating
compliance as a later-version concern is no longer comfortable for a product shipping into Indian homes.

**The `face` class is the sharp edge.** A face bounding box in domestic video is personal data about an
identifiable individual even with no embedding computed, and `face` adds essentially nothing over
`person` for safety purposes. Recommendation: drop `face` and `passport` from the *deployed* head rather
than relying on a config toggle that has already been demonstrated not to work.

**Video is a categorical escalation from stills.** Clips capture gait, posture, undress, toileting, and —
unavoidably — third parties who never consented, including domestic workers who are in a structurally
unequal position to refuse. Required additions: a separate consent instrument and scope, household-wide
notice with per-person opt-in, a room exclusion list enforced in config, retention limits with automatic
deletion, a visible recording indicator, a physical camera-off control the resident can operate, and MP4
metadata stripping — which does not currently exist, since `exif.py` handles image EXIF while MP4 carries
GPS in a different container atom.

> **Requirement.** A `claim_class` field plus a validator rejecting regulated verbs in any user-facing
> string, and a `capability_disclaimer` required on every `detectability: inferred` scenario, surfaced to
> the caregiver alongside the alert.

## 6. Evidence field contract

```yaml
evidence:
  - source_id: CDC-STEADI-CheckForSafety-2017
    section: "Bathroom"
    item: "Install grab bars by the toilet and in the tub/shower"
    strength: guideline        # guideline | systematic_review | rct | observational | expert_opinion | none
    supports: hazard           # hazard | intervention | detection_method
    note: "Backs the HAZARD. Does NOT back camera-based detection of it."
```

Three validators enforce it:

1. `strength: none` or empty `evidence` forbids `risk_level ∈ {HIGH, CRITICAL}` and forbids
   `caregiver_channel: push`.
2. `supports: hazard` never licenses `detectability: direct`.
3. `detectability: rejected` rows stay in the repository with a `rejection_reason`.

## 7. Evidence status by category

| Category | Status | Cite |
|:---|:---|:---|
| Floor hazards — cords, clutter in walkways | Strong | `CDC-STEADI:CheckForSafety:Floors` |
| Bathroom — grab bars, non-slip surfaces | Strong | `CDC-STEADI:CheckForSafety:Bathroom` |
| Wet/slippery floors | Strong as a hazard; weak for camera detection (R21/R24) | STEADI + internal R24 gate |
| Mobility-aid availability and use | Strong | `WHO:StepSafely:2021` |
| Unattended cooking | Strong for the hazard; weak support from these classes (no stove state) | Fire-safety guidance + detectability caveat |
| Stairs, lighting, high-shelf reaching | Strong hazards, **not supportable** — no such classes | Negative register |
| Prolonged bathroom occupancy | Moderate for the underlying claim; the inference is ours | `detectability: inferred` |
| Alarm fatigue as a design constraint | Strong | Monitor-alarm and telecare-abandonment literature |
| Medication adherence by object detection | **No evidence base. Invented** | — |
| Hydration from bottle presence | **Invented** | — |
| Knife-proximity warnings | **Invented**, and contradicts IADL preservation | — |
| Gas-cylinder visibility warnings | **Invented** — LPG guidance concerns leak checks and tubing, not visibility | — |

## Sources

- [CDC STEADI — Check for Safety home fall-prevention checklist](https://www.cdc.gov/steadi/pdf/steadi-brochure-checkforsafety-508.pdf)
- [CDC STEADI initiative](https://www.cdc.gov/steadi/index.html)
- [Lawton–Brody IADL scale](https://geriatrictoolkit.missouri.edu/funct/Lawton_IADL.pdf)
- [Home and environmental hazard modification for fall prevention (review)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8246567/)
- [Older adult falls in the community: does an unsafe home environment have a risk role?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10737246/)
- [The Digital Personal Data Protection Act, 2023 (MeitY)](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
- [DPDP Act 2023 and DPDP Rules 2025 compliance guide (EY India)](https://www.ey.com/en_in/insights/cybersecurity/decoding-the-digital-personal-data-protection-act-2023)

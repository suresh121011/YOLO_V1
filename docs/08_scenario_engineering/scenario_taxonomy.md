# Scenario Taxonomy

## Purpose

Phase 4 deliverable. Defines the category scheme, the id grammar, and the shape of the initial
scenario set — including which categories are deliberately empty and why.

## Dependencies

Reads:
- `negative_register.md`
- `domain_research_report.md`
- `requirements_specification.md`

Used By:
- `configs/scenarios/SC-*.yaml`
- `src/scenario_engine/compile.py`

---

## 1. Id grammar

`SC-<CAT>-<NNN>`, matching `^SC-[A-Z]{3}-\d{3}$`. Ids are immutable and never reused, including
after deprecation — the file stays on disk, which *is* the reservation (ADR-P6-06).

## 2. Categories

Most categories are **rooms**, sharing the vocabulary in `configs/capture_config.yaml` so that
`<CAT>`, `category`, and `room_context` all validate against one list. A room is a deployment-time
constant — one camera per room — not a per-frame geometric zone (ADR-P6-07).

| Code | Category | Populated? |
|:---|:---|:---|
| `KIT` | Kitchen | yes |
| `BTH` | Bathroom | yes |
| `BED` | Bedroom | yes |
| `COR` | Corridor | yes |
| `HAL` | Hall | — folded into `COR`; the supportable findings are identical |
| `BAL` | Balcony | **empty** — no supportable scenario with these classes |
| `POO` | Pooja room | **empty** — same |
| `STA` | Staircase | **empty and important**: stairs are one of the strongest STEADI items and there is **no `stairs` class**. This is a capability gap, not an oversight |

Three categories are cross-cutting rather than rooms:

| Code | Category | Purpose |
|:---|:---|:---|
| `MOB` | Mobility | Walking-aid availability and use — the highest-value class in the taxonomy |
| `MED` | Medication | Caregiver-side observation only; never a patient prompt |
| `SYS` | System health | Presence-unknown and silent-failure conditions, exempt from suppression |
| `SAF` | Safety register | `detectability: rejected` rows — the versioned negative register |

## 3. The three shapes a scenario can take

Recognising these keeps the set honest, because most supportable findings are **not** real-time
alarms:

**Environmental finding** — a static home-safety defect. Caregiver-facing, `patient_facing: false`,
`max_per_day: 1`, LOW risk. The CDC STEADI intervention is almost always a one-time modification by
a caregiver ("install grab bars", "tape cords to the wall"), so telling the resident every two
minutes fixes nothing. Most of the genuinely-supportable set lives here.

**Real-time hazard** — something happening now that the resident can act on. Patient-facing, dwell
required, escalation instead of repetition. Rare, because it needs a hazard that is both observable
*and* actionable within seconds.

**Observation** — a weakly-inferable pattern. `detectability: inferred`, mandatory
`capability_disclaimer`, caregiver digest only, never spoken. The disclaimer is surfaced with the
alert so an inference is never read as an observation.

## 4. Deliberate scarcity

Nine supportable scenarios against eleven rejected ones is the correct ratio for this taxonomy, not
a gap to be closed. 23 static-object classes plus `person`/`face` — no pose, no on/off state, no
tracking — support far fewer real-time scenarios than the product framing implies. Adding more would
mean asserting state the system cannot observe.

The rejected rows carry the same schema and the same review as the active ones. They are compiled
into the artifact's `rejected:` block so the reasoning is versioned alongside the positive set and
the ids stay reserved (ADR-P6-08).

## 5. Classes that carry no scenario, and why

| Class | Why nothing keys on it |
|:---|:---|
| `knife` | Only as *suppression context* and in "left out with nobody present". `knife` + `person` is the definition of cooking, a preserved IADL |
| `medicine_strip`, `medicine_bottle` | Caregiver observation only. There is no schedule, dose, or adherence state |
| `gas_cylinder` | Permanently visible in an Indian kitchen, so presence carries ~zero information. The defensible LPG scenario is a scheduled bedtime cue, which needs a clock the engine does not have |
| `face` | Adds nothing over `person` for safety, and carries the entire biometric argument. Recommended for removal from the deployed head |
| `passport` | No safety use. Already `enabled: false`, and the capability map now enforces that |
| `book`, `laptop`, `monitor`, `charger` | Weak activity context at best |
| `water_bottle` | Presence is not fluid intake |

## 6. Activation gate

Every scenario in the initial set ships as **`status: draft`**.

`status: active` requires `reviewed_by` and `reviewed_on`, and that review is a **clinical judgement
by a qualified human**, not an engineering sign-off. This project already demands two annotators at
IAA ≥ 0.75 to accept a bounding box; an instruction spoken to an elderly person deserves at least as
much. Marking these active without that review would fabricate the sign-off the schema exists to
require.

Consequence, and it is the honest one: `validate_scenarios` reports
`safety-class-uncovered` warnings for every safety-critical class, because coverage counts **active**
scenarios only. Those warnings clear when clinical review happens — a human track, like H-A/H-B/H-C
in Phase 5 — not before.

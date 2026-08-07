# Negative Register — Scenarios Permanently Out of Scope

## Purpose

A versioned, in-repo record of scenarios that **cannot be delivered** with the frozen 23-class taxonomy,
so that no future contributor silently re-adds them. Every entry states the missing capability, not just
a refusal.

## Dependencies

Reads:
- `../../configs/data.yaml` (the 23 classes)

Used By:
- `src/scenario_engine/` validators (`detectability: rejected` rows must carry a `rejection_reason`)
- `requirements_specification.md`

Related:
- `adr/ADR-P6-08-rejected-scenario-negative-register.md`

---

## What the taxonomy actually is

23 **static object** classes plus `person` and `face`. There is no pose estimation, no keypoints, no
tracking identity, no object state (on/off, running/dry, full/empty), no floor plane, no depth, no
audio, and no clock in the rule engine. This is a much narrower instrument than the product framing
implies, and the gap between the two is where a safety product becomes a safety liability.

The rule that governs every entry below: **a scenario may not assert state the system cannot observe.**

## Rejected scenarios

| Scenario | Missing capability | Why it is fabrication, not just imprecision |
|:---|:---|:---|
| **Fall detection / "has fallen"** | Pose, keypoints, floor plane, motion model | A `person` box that becomes wide and low is a person bending, sitting on the floor, praying, or exercising. This is the first thing every family will assume the product does, so the exclusion must be stated in the product surface, not only here |
| **Emergency detection (any kind)** | Everything above | Nothing in this taxonomy detects an emergency. The escalation ladder therefore terminates at a human check-in |
| **"Stove is on" / "left on"** | Flame or heat state | Currently *asserted* at `configs/risk_rules.yaml:54`. The class is `stove`, a physical object, and it is always detected |
| **Gas leak detection** | A gas sensor | A camera cannot see LPG |
| **Medication adherence / "did you take your pills"** | Pill count, schedule, drug identity, ASR | Currently implemented at `configs/risk_rules.yaml:59-64`. Asking an unanswerable question invites a resolve-the-uncertainty second dose — a real harm pathway for narrow-therapeutic-index drugs common in this population |
| **Hydration monitoring** | Fill level, drinking event | `water_bottle` presence is not fluid intake |
| **Tap left running / flooding** | Water-flow state | `sink` is furniture |
| **"Person is unconscious / not moving"** | Pose, vitals, motion analysis | — |
| **Wandering / elopement** | Door open-closed state, threshold-crossing semantics | `door` is a static object with no state |
| **Cognitive-decline or behaviour-change inference** | Longitudinal ADL modelling | Also a medical claim |
| **Knife-handling safety** | Grip, motion, surface/edge classes | See below — the only defensible use of `knife` is as suppression context |
| **Person identification / "who is home"** | Identity | `face` gives no identity, and adding it would be a deliberate step into biometric processing the project has committed against |

## Three live rules that are being deleted, not migrated

These are not merely weak — shipping them degrades safety by training the user to ignore the device.

**`knife_near_person` (HIGH, 60 s).** `detected(knife) AND detected(person)` is the definition of
cooking, a preserved IADL under the Lawton–Brody framework. In a household kitchen that condition holds
for 30–60 minutes a day, producing 30–60 HIGH announcements per meal. Three compounding harms: a sudden
loud voice at a person holding a knife is a *cause* of laceration; ~50 useless HIGH alerts on day one
teach the user that HIGH means nothing; and warning a competent adult that their own knife exists is the
fastest route to the device being unplugged. No version of a knife rule is supportable here. `knife`
becomes an *activity-context* input that **suppresses** other scenarios during meal prep.

**`medicine_reminder` (INFO, 300 s).** See the table. A strip left on a table is a stable object, so
300 s means ~190 identical prompts a day. The only honest medication-adjacent output is a caregiver-side
observation carrying an explicit capability disclaimer, never a patient prompt.

**`gas_cylinder_check` (INFO, 600 s).** `detected(gas_cylinder) AND NOT detected(stove)` in an Indian
kitchen is the normal state — the cylinder sits permanently visible under the platform while the stove
is frequently occluded by the cook or a vessel. The rule therefore fires on *"the stove detector missed
the stove this frame"*: a detector-dropout alarm dressed as a safety check, ~100 times a day including
overnight. Note also that its `message_hi` at `:71` silently drops the "connected" clause present in
`message_en` — nothing validates EN/HI semantic parity today.

If an LPG scenario is wanted, the defensible form is a once-daily scheduled bedtime routine cue, not a
detection trigger.

## Retained, with conditions

| Scenario | Condition of retention |
|:---|:---|
| `wet_floor_hazard` | Requires dwell ≥ 3 s, an explicit clear condition, escalation instead of repetition, and remains gated on the R24 pilot decision. If `wet_floor` is demoted to scene-level, this scenario must auto-disable — which the taxonomy fingerprint cannot detect, hence the class capability map |
| `wire_tripping_hazard` | Re-scoped to a **caregiver-facing, non-spoken, once-per-location home-assessment finding**. The CDC STEADI intervention is "coil or tape cords next to the wall" — a one-time fix by a caregiver. Telling the resident to watch their step every two minutes fixes nothing |
| `stove_unattended` | Intent is sound (unattended cooking is a leading cause of home cooking fires). Requires D4 fixed, a real cooking-session state rather than bare `absent_for`, a message describing only what is observed, and escalation to a caregiver rather than repetition into an empty room |

## What is genuinely supportable and currently missing

**No grab bar near the toilet** — `toilet` or `sink` detected in a room where `support_handle` is never
co-detected across N days. This is a literal CDC STEADI *Check for Safety* bathroom item, a one-shot
finding with a concrete caregiver action, fully supported by these classes, and it never nags the
resident. It is the highest-value scenario available and it is not in the current ruleset.

Others in the same family: cord-on-floor as an assessment finding; walking-stick left away from the
person; clutter accumulation in a corridor. All are **environmental findings** rather than real-time
alarms, which is what makes them quiet.

## Rule for adding to this register

A rejected scenario stays in `configs/scenarios/` as a real file with `detectability: rejected` and a
`rejection_reason`. It is excluded from the compiled runtime array but retained in the artifact's
`rejected:` block, so the reasoning is versioned alongside the positive set and the ID is never reused.

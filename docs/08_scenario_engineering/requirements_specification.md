# Requirements Specification — Scenario & Rule Engine

## Purpose

Phase 2 deliverable. Defines what a Scenario is, what a Rule is, what makes a scenario valid, which
fields are mandatory, and which assumptions are forbidden. This is the contract the schema in
`src/scenario_engine/schema.py` implements and the validators enforce.

## Dependencies

Reads:
- `architecture_review.md`
- `negative_register.md`
- `domain_research_report.md`
- `../../configs/data.yaml`

Used By:
- `src/scenario_engine/schema.py`
- `src/scenario_engine/validators/`

---

## 1. Definitions

**Scenario** — a named, evidence-backed situation in an elderly home that the system can *observe*, with
a declared response. It is the unit of authoring, review, versioning, and clip validation. A scenario
owns a trigger, a risk tier, the messages, the escalation ladder, and its own suppression budget.

**Rule** — the compiled, executable form of a scenario's trigger: a predicate AST evaluated against an
`EvalContext` for one frame plus temporal memory. Authors never write rules; the compiler emits them.
One scenario compiles to exactly one rule.

**Predicate** — a registered pure function over `EvalContext` (`detected`, `absent_for`, `present_for`,
`count`, `near`, `overlaps`, `any_of`, …). Unknown predicate names are a compile-time hard failure that
enumerates the valid set.

**`EvalContext`** — detections *with bounding boxes and multiplicity*, event memory, measured fps, frame
id, and deployment room. Today `_evaluate_condition` receives a bare `set[str]`, which is why spatial
and counting predicates are unimplementable without this type. Defining it is task zero of M1.

**Event** — one instance of a scenario becoming true, with a lifecycle:
`IDLE → PENDING (dwell) → ACTIVE (announced) → ESCALATED → RESOLVED | SUPPRESSED`.
Cooldown guards only the `RESOLVED → PENDING` edge. **The current engine has no event concept** — it
fires, cools down, and fires again identically forever, which is the mechanism behind the projected
~370 alerts/day.

## 2. What makes a scenario valid

A scenario is valid only if **all** hold:

1. **Observable.** Every asserted state maps to something the taxonomy encodes. Since no class encodes
   state, all messages are phrased as appearance or observation.
2. **Non-static trigger.** A trigger that is satisfiable by object presence alone is rejected
   structurally. Every scenario must carry at least one temporal, absence, count, or spatial term.
   This single rule invalidates three of the six live rules at authoring time.
3. **Satisfiable.** `required ∩ forbidden = ∅`, and likewise for `optional ∩ forbidden`.
4. **Detectable in practice.** Every referenced class resolves against `configs/data.yaml` *and* is
   `bbox`-detectable *and* enabled in the class capability map.
5. **Reachable.** `min_confidence` is at or above the detector's effective floor for that class;
   temporal values are within the memory window.
6. **Evidence-proportionate.** `evidence.strength: none` forbids `risk_level ∈ {HIGH, CRITICAL}` and
   forbids `caregiver_channel: push`. If it cannot be cited, it cannot alarm.
7. **Bounded.** `max_per_day`, `max_repeats`, `min_dwell_seconds`, and a `clear_condition` are all
   present.
8. **Reviewed.** `status: active` requires `reviewed_by` and `reviewed_on`. The project already demands
   two annotators at ≥0.75 IAA to accept a bounding box; an instruction spoken to an elderly person
   deserves at least as much.

## 3. Mandatory fields

| Field | Notes |
|:---|:---|
| `scenario_id` | `^SC-[A-Z]{3}-\d{3}$`. Immutable; never reused, including after deprecation |
| `schema_version` | Row shape version, independent of dataset semver |
| `scenario_name`, `category` | Category vocabulary is shared with `configs/capture_config.yaml` rooms |
| `detectability` | `direct` \| `inferred` \| `rejected` |
| `claim_class` | `observation` \| `reminder` \| `inference` |
| `required_objects` | Non-empty |
| `trigger` | Structured predicate tree (not a string) |
| `risk_level` | One of the existing `Severity` names — **not a new vocabulary** |
| `min_confidence`, `min_dwell_seconds`, `clear_condition`, `max_repeats`, `max_per_day`, `cooldown_seconds` | All required; see validity rule 7 |
| `priority` | Total ordering for arbitration when several scenarios fire |
| `messages` | Open-ended `{lang: {...}}` map. At minimum `en` and `hi`; `hi` must contain Devanagari |
| `patient_facing` | Boolean. If false, no prompt is ever spoken |
| `next_best_action` | — |
| `caregiver_channel` | `none` \| `digest` \| `push` \| `push_and_call`. **Not a boolean** |
| `escalation` | Ladder; must terminate |
| `quiet_hours` | `{start, end, behaviour}`; `behaviour: always_speak` is permitted only for CRITICAL |
| `evidence` | List of `{source_id, section, item, strength, supports}` |
| `status` | `draft` \| `active` \| `deprecated` \| `rejected` |
| `rule_hash` | Computed, not authored. sha256 over semantic fields only |

## 4. Optional fields

`optional_objects`, `forbidden_objects`, `spatial_predicates`, `room_context`, `suppress_during`,
`require_alone`, `capability_disclaimer` (**mandatory when `detectability: inferred`**),
`false_positive_notes`, `notes`, `sample_video_needed`, `deprecated_in`, `superseded_by`,
`rejection_reason` (**mandatory when `detectability: rejected`**).

## 5. Forbidden assumptions

These are the assumptions that make a home-monitoring system dangerous. Each maps to a validator.

- **That absence of `person` means absence of a person.** Occlusion, detector dropout, a blocked camera,
  and a person on the floor below the field of view are all indistinguishable from "nobody home". A
  first-class `presence_state` with an explicit `UNKNOWN` value is required, and `UNKNOWN` is a
  *system-health* condition, not a safety one.
- **That one resident lives alone.** Multi-generational households are the norm in the target market.
  `absent_for(person, N)` is satisfied by *any* person, so a grandchild in the kitchen suppresses the
  unattended-stove rule, and prompts intended for the elder are heard by the whole family — a dignity
  problem. Counting predicates and a `require_alone` field are required.
- **That the system is working.** A safety device that fails silently displaces the vigilance that would
  otherwise exist. A heartbeat with an *inverted* alarm (the caregiver is notified when it stops) and a
  `SYSTEM_DEGRADED` scenario class — exempt from all suppression and budgets — are required.
- **That repetition is escalation.** Saying the same thing louder is not a next step.
- **That the caregiver channel is free.** Fatiguing the caregiver is worse than fatiguing the patient;
  the caregiver's attention is what actually protects the resident.
- **That a boolean captures alert routing.** As a boolean, `caregiver_alert` gets set `true` everywhere.
- **That English and Hindi strings stay in sync.** They already do not — `configs/risk_rules.yaml:71`
  drops a clause present in the English. Parity is validated, not trusted.
- **That night is a modifier.** Nocturnal toileting is a high-risk fall window, which makes it the most
  valuable time to be right and the worst time to be wrong. The rule engine currently has no clock at
  all — only `time.monotonic()` for cooldowns.

## 6. Non-functional requirements

- **Latency.** Scenario evaluation ≤ 5 ms/frame, the budget already allocated at
  `../02_technical_architecture_specification/performance_budget.md:30` and already measured by
  `PipelineMetrics.rule_eval_ms`. Enforced by a `tests/performance/` budget test.
- **Determinism.** The set of scenarios that fires, and its order, is a pure function of the detection
  set — independent of file iteration order. Proven exhaustively over subsets of the *referenced*
  classes.
- **Reproducibility.** Compiled artifacts are byte-identical across OSes: LF line endings, sorted keys,
  `ensure_ascii=False`, deterministic row order.
- **Fail-closed loading.** The runtime hard-fails on zero rules, content-hash mismatch, unknown
  predicate, or unknown severity. Today `_load_rules` logs `Loaded 0 rules` at INFO and continues.
- **Privacy.** No `person`/`face` geometry may reach `logs/events.jsonl`, including via `explanation`.

## 7. Future requirements to design for now

Cheap to accommodate today, expensive to retrofit:

- **Locale beyond Hindi.** Marathi, Tamil, Telugu, Bengali, Kannada, Malayalam, Gujarati, Punjabi. Hence
  an open-ended `messages` map rather than the `message_en`/`message_hi` pair baked into `Alert`.
- **A caregiver feedback loop** with four dispositions, not a boolean false-alarm flag:
  `real_and_useful`, `real_but_not_useful`, `false_alarm_wrong_detection`, `false_alarm_wrong_rule`.
  Only the third is a labelling signal — routing rule-logic errors into the training set would teach the
  detector that a correctly-detected stove is wrong.
- **Deployment-data consent as a separate purpose.** Frames from a customer's living room are not
  covered by the capture-session consent scope, and must not enter the training corpus under it.
- **An input channel from the human.** Acknowledge, snooze, camera-off. A system that can only talk has
  the power switch as its sole user control, which is why such devices get unplugged.
- **A VLM contract.** `SceneContext.activity` already overlaps with "scenario". The stated invariant is
  that the VLM is advisory and cannot cancel a rule alert; a deterministic, YOLO-derived activity context
  is what may suppress.

# ADR-P6-09 — Extend the LOCKED `Alert` Additively Rather Than Returning a Parallel Outcome Type

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

The product's stated output is four-stage: risk level → patient prompt → next best action → caregiver
alert. The transport is `Alert` (`src/pipeline/__init__.py:163-193`), and it cannot carry two of those
four:

- `:186` documents `message` as the only field shown or spoken to the user.
- `:193` documents `explanation` as a debug dict, "never shown to user".
- There is no field for `next_best_action` and none for `caregiver_alert`.

`src/pipeline/__init__.py:8` declares these dataclasses **LOCKED** and states that changing them is a
breaking change.

Smuggling the values through `explanation` is specifically unsafe. `src/logging/structured_logger.py:40`
defines `BBOX_REDACT_CLASSES = frozenset({"face", "person"})` and redacts those bounding boxes from frame
logs — but `log_alert` at `:160-174` writes `alert.explanation` **verbatim and unredacted**. Spatial
predicates naturally want `{"person_center": [0.41, 0.62], "dist_to_stove": 0.08}` for debuggability, so
routing product output through `explanation` would also normalise putting person geometry into
`logs/events.jsonl` on a device in an elderly person's home.

## Decision

**Extend `Alert` additively.** New fields, all with defaults, so every existing construction site and
every existing test continues to work unchanged:

| Field | Default | Purpose |
|:---|:---|:---|
| `scenario_id` | `None` | Provenance back to the authored row |
| `next_best_action` | `None` | The action stage of the product output |
| `caregiver_channel` | `"none"` | `none` \| `digest` \| `push` \| `push_and_call` |
| `patient_facing` | `True` | When false, no prompt is spoken |
| `messages` | `None` | Open-ended `{lang: str}` map, superseding the `message_en`/`message_hi` pair for multi-locale support |

`message` remains the single spoken string, so `orchestrator.py:205-206` is untouched. `message_hi` is
retained and deprecated in favour of `messages` rather than removed.

This is recorded as a **non-breaking extension of a locked contract**: additive, defaulted, and
accompanied by a `CHANGELOG.md` entry under `[Unreleased]`. The LOCKED annotation at `:8` is updated to
state that additive-with-default extensions are permitted under a recorded ADR, and that field removal,
renaming, or semantic change remains prohibited.

Separately and independently: `explanation` dictionaries are routed through the same redaction allowlist
as frame logs, with a unit test asserting that no `person` or `face` geometry survives into any logged
explanation. This is required regardless of the field decision above.

## Alternatives considered

1. **A parallel `ScenarioOutcome` returned alongside `list[Alert]`.** Rejected: it changes the
   `BaseRuleEngine.evaluate` signature (`:357-371`), which is the very seam that makes the one-line
   injection in [ADR-P6-04](ADR-P6-04-rule-engine-injection.md) possible; and it forces a second sink and
   a second arbitration path in the orchestrator, so two alert streams would need reconciling before the
   single `max(alerts, ...)` selection at `:205`.
2. **Carry the fields inside `explanation`.** Rejected on the privacy grounds above, and because
   `explanation` is explicitly documented as debug-only — product output living in a debug field is how
   it ends up unshipped and undiscovered.
3. **Leave `Alert` untouched and drop the two stages.** Rejected: it silently reduces the product to
   risk-plus-prompt, and the gap would surface at the first stakeholder demo rather than in review.
4. **A new `ScenarioAlert` subclass of `Alert`.** Rejected: consumers would need `isinstance` checks, and
   any consumer that does not perform them silently loses the extra fields — the same failure as option 3
   with more machinery.

## Consequences

- Positive: the full four-stage output is expressible, and the injection seam stays a single constructor
  parameter.
- Positive: `messages` unblocks locales beyond Hindi without another contract change. The target
  population includes large Marathi, Tamil, Telugu, Bengali, Kannada, Malayalam, Gujarati, and Punjabi
  first-language groups.
- Positive: the redaction fix closes a live privacy hole that exists today, independent of this phase.
- Constraint: the LOCKED contract's rules are now nuanced rather than absolute. The precise permission —
  additive, defaulted, ADR-recorded — is written into `src/pipeline/__init__.py` itself so the nuance
  travels with the code.
- Constraint: `caregiver_channel` has no consumer until the caregiver notification system exists
  (`configs/feature_flags.yaml:19` `caregiver_sync: false`). The field is carried and logged in the
  meantime; M9 specifies the sink.

Related: [ADR-P6-04](ADR-P6-04-rule-engine-injection.md)

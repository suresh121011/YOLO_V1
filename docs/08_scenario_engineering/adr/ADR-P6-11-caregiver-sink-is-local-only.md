# ADR-P6-11 — The Caregiver Sink Is Local-Only, and `push_and_call` Says So

**Status:** Accepted (Phase 6, M9, 2026-08)
**Deciders:** Phase-6 engineering

## Context

[ADR-P6-09](ADR-P6-09-alert-contract-extension.md) added `Alert.caregiver_channel` (`none` | `digest` |
`push` | `push_and_call`) and recorded, under Consequences: *"`caregiver_channel` has no consumer until
the caregiver notification system exists. The field is carried and logged in the meantime; M9 specifies
the sink."*

Two things were true when M9 checked. It had no consumer — and it was **not logged either**:
`StructuredLogger.log_alert` wrote `rule_id`, `severity`, `message`, `frame_id` and `explanation`, and
none of the ADR-P6-09 fields. So from the event log alone there was no way to tell a caregiver-only
alert from a spoken one, or to audit whether a channel had been honoured.

The scale matters. All nine non-rejected scenarios declare a channel, and **seven are
`patient_facing: False`** — `SC-BTH-002` (no grab bar near the toilet) is entirely caregiver-facing by
design. M9 also fixed `patient_facing` being enforced nowhere. Fixing that alone would have taken seven
of nine scenarios from "wrongly spoken to the resident" to "producing nothing observable anywhere".

There is still no caregiver application, no caregiver identity, no consent surface covering
notification, and no retention policy for anything sent off the device.
`configs/feature_flags.yaml` has carried `caregiver_sync: false` since before Phase 6.

## Decision

**Ship a local sink and nothing else.** `BaseCaregiverSink` is a `Protocol` in `src/pipeline/__init__.py`;
`LocalCaregiverSink` appends JSONL to `logs/caregiver.jsonl`. No network, no push service, no telephony.

Four behaviours are load-bearing:

1. **`notify()` returns `False` only for `channel: none`.** "The caregiver is deliberately not told about
   this one" must be distinguishable from "a channel was requested and nothing happened". The second is a
   silently dropped escalation and must never look like the first.
2. **`push_and_call` records `escalation_pending: true` and logs at WARNING**, ending
   `NOBODY HAS BEEN CALLED`. No telephony exists. The product's stated non-negotiable is that escalation
   terminates at a human; when it cannot reach one it says so loudly rather than writing a line that
   reads as delivered.
3. **An unknown channel is delivered, not dropped**, and marked `unknown_channel`. The schema constrains
   the vocabulary at authoring time, so reaching that branch means the two have drifted — and
   over-notifying a caregiver is recoverable in a way a withheld alert is not.
4. **Records carry class names, never bounding boxes.** `StructuredLogger` redacts person/face geometry
   from frame logs; a second alert-shaped log leaking it would reopen that hole in a different file.

Separately, `log_alert` now writes `scenario_id`, `next_best_action`, `caregiver_channel` and
`patient_facing`, making ADR-P6-09's "carried and logged" true.

`caregiver_sync` is redefined to govern **remote** delivery only. The local sink is always on: a device
on a dropped WiFi link must still record what it would have sent.

## Alternatives considered

1. **Leave it unconsumed until a caregiver app exists.** Rejected. Seven of nine scenarios would emit
   nothing at all once `patient_facing` was honoured, and the two defects would go on masking each other
   — everything spoken, nothing routed, and the device looking correct.
2. **Write a real remote sink now (HTTP push, SMS, or e-mail).** Rejected, and this is the important
   one. A transport with no caregiver identity, no consent covering notification and no retention policy
   is not a feature, it is the *appearance* of an escalation path. Anyone reading the code would
   reasonably conclude that alerts reach someone.
3. **Degrade `push_and_call` to `push` silently.** Rejected: it converts "we cannot reach a human" into a
   record indistinguishable from success. This is the exact failure mode that makes a safety system
   dangerous rather than merely limited.
4. **Extend `StructuredLogger` instead of adding a sink.** Rejected: the caregiver stream has different
   consumers, a different lifetime and different redaction requirements from frame telemetry, and
   `digest` needs buffering that a line-oriented logger has no place holding.

## Consequences

- Positive: the fourth product stage becomes observable and testable with no invented transport.
- Positive: `patient_facing` can be honoured safely, because caregiver-only findings still land
  somewhere.
- Positive: the seam for a real app is one constructor parameter (`build_pipeline(caregiver_sink=...)`),
  matching the rule-engine injection in ADR-P6-04.
- Constraint: nothing is delivered off the device. `logs/caregiver.jsonl` must be read by a human or
  collected by whatever ships next, and the digest is only written on `flush()` — which the orchestrator
  calls on shutdown, before anything else in the shutdown path.
- Constraint: the digest is bounded (`max_digest`, default 200) and reports what it dropped. A device
  that runs for months cannot buffer without a limit.

Related: [ADR-P6-09](ADR-P6-09-alert-contract-extension.md) ·
[ADR-P6-04](ADR-P6-04-rule-engine-injection.md) · [../integration_strategy.md](../integration_strategy.md)

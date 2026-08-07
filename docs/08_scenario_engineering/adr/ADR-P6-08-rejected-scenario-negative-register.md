# ADR-P6-08 — Rejected Scenarios Stay In-Repo as Versioned Rows

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

The 23-class taxonomy is static objects plus `person`/`face`. There is no pose, no keypoints, no object
state, no tracking, no depth, no audio. A capability audit found that many of the scenarios the product
framing implies — most prominently **fall detection** — cannot be delivered at all, and shipping them
would mean the system asserting states it cannot observe.

The risk is not that someone builds fall detection deliberately. It is that six months from now a
contributor with no memory of this analysis sees an obvious gap and adds it, and the reasoning that
excluded it exists only in a design document nobody greps.

The harm mechanism is specific and worth stating: a family told the device is an "always-on safety
companion" assumes it detects falls, stops the daily phone check-in, and a real fall goes undiscovered.
**The over-claim causes the harm by displacing a working human process.**

## Decision

A rejected scenario is a **real file** in `configs/scenarios/` with `status: rejected`,
`detectability: rejected`, and a mandatory `rejection_reason` naming the missing capability — not merely
a refusal. The compiler excludes these from the runtime array and retains them in the artifact's
`rejected:` block, so the reasoning is versioned alongside the positive set, appears in the compiled
diff, and the ID is permanently reserved.

`negative_register.md` is the human-readable rendering of that set and states the governing rule: **a
scenario may not assert state the system cannot observe.**

Two validators give it teeth:

- **No unobservable assertions** — every patient- and caregiver-facing string is linted against a
  banned-assertion lexicon (`is on`, `is off`, `has fallen`, `is running`, `you took`, `did not take`,
  `emergency`). Since no class in this taxonomy encodes state, the effective rule is that all messages
  are phrased as appearance or observation.
- **`claim_class`** (`observation` | `reminder` | `inference`) plus rejection of regulated verbs
  (`detect`, `diagnose`, `monitor`, `prevent`, `ensure`, `guarantee`, `protect`) in any user-facing
  string unless explicitly whitelisted. This stops claim-creep at pull-request time rather than at legal
  review.

Additionally, `capability_disclaimer` is **mandatory** whenever `detectability: inferred`, and is
surfaced to the caregiver alongside the alert — so an inference is never presented as an observation.

## Alternatives considered

1. **Document rejections in prose only.** Rejected: prose is not diffed, not validated, and not visible
   at the point where someone adds a scenario file. The register must live where authoring happens.
2. **Delete rejected scenarios entirely.** Rejected: it loses the reasoning and frees the ID for reuse,
   which breaks the immutability guarantee in [ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md).
3. **A separate `rejected/` directory.** Rejected: it forks the schema and the validators, and a rejected
   row benefits from the same class-reference and evidence validation as an active one.
4. **Allow rejected scenarios with a "future" status implying eventual delivery.** Rejected as the most
   dangerous option — it converts a permanent capability limit into a backlog item, which is exactly the
   framing that produces over-claiming.

## Consequences

- Positive: fall detection cannot be quietly re-added; the file, the reason, and the missing capability
  are all present in the tree and in the compiled artifact.
- Positive: the same mechanism documents the three live rules being deleted
  (`knife_near_person`, `medicine_reminder`, `gas_cylinder_check`) with their clinical rationale, so
  their removal is not mistaken for an oversight.
- Constraint: the banned-lexicon validators will occasionally produce false positives on legitimate
  wording. The whitelist is explicit and per-scenario, so an override is visible in review rather than
  silent.
- Constraint: the register must be revisited if the taxonomy ever gains state-bearing classes. Each entry
  names the missing capability precisely so that re-evaluation is mechanical.

Related: [ADR-P6-05](ADR-P6-05-class-capability-map.md), [ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md)

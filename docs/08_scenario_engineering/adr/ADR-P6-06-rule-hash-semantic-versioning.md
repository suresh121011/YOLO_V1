# ADR-P6-06 — `rule_hash` Over Semantic Fields Only, With a Derived-Semver Gate

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

Scenario clips are labelled against a scenario's behaviour: this clip should fire `SC-KIT-003` at HIGH
within 2 s. If the scenario's semantics later change, every clip labelled against the old behaviour is
stale — and the regression suite will keep passing green **against a wrong expectation**, which is worse
than failing.

The design initially conflated two independent axes. `schema_version` versions the *shape* of a row and
is mechanically migratable. Semantics are not migratable at all, and are what invalidates labels.

The repository's own history is the argument for enforcement over policy: release gate RG6 was
unfalsifiable for an entire cycle and certified a release over 21,964 un-pushed objects — the incident is
recorded in the docstring at `src/dataset/release/gates.py:394-421`, along with the resulting rule that a
gate which cannot run must never read green.

## Decision

**`rule_hash`** — a `sha256:`-prefixed digest over the canonical JSON of only the semantically
load-bearing fields: `required_objects` (sorted), `optional_objects`, `forbidden_objects`,
`spatial_predicates`, `min_confidence`, `min_dwell_seconds`, `clear_condition`, `room_context`,
`risk_level`, `caregiver_channel`. It **deliberately excludes** `patient_prompt`/`messages`, `notes`,
`evidence`, `false_positive_notes`, and `scenario_name` — so a Hindi typo fix invalidates zero clips.

Computed exactly as the two existing fingerprints are (`src/dataset/completeness.py:80-85`,
`src/dataset/annotation/base.py:260`): `"sha256:" + sha256(json.dumps(obj, separators=(",", ":"),
ensure_ascii=False))`.

Every clip label record stores `{scenario_id, rule_hash_at_label_time, labeller, labelled_at}`.

**Dataset semver** on `scenarios-v{major}.{minor}.{patch}`, mirroring `data/DATASET_CHANGELOG.md`:

- **major** — anything invalidating existing clip labels: any `rule_hash` change on an active scenario,
  deletion of an active scenario, or a taxonomy fingerprint change.
- **minor** — a new scenario added, or an active scenario deprecated.
- **patch** — prompt, notes, or evidence edits; schema-only migrations.

Because that rule is *derivable*, a validator diffs the previously committed `scenarios.compiled.json`
against the new one, computes the **required** bump, and hard-fails if the declared bump is smaller. This
extends the RG4 pattern (`gates.py:322-332` checks only that a changelog heading exists) into checking
that the version is *correct*.

Gate set, evaluated in the M6 validation milestone: **RG-S1** compile clean · **RG-S2** every active
scenario has ≥1 positive and ≥1 negative clip · **RG-S3** zero clips labelled against a stale `rule_hash`
· **RG-S4** taxonomy fingerprint and capability map match live configs · **RG-S5** changelog heading
exists · **RG-S6** declared bump ≥ derived bump · **RG-S7** every clip carries a consent reference and a
metadata-stripped attestation.

Deprecation: `status: deprecated` + `deprecated_in` + `superseded_by`. The compiler excludes deprecated
scenarios from the runtime array but retains them in a `deprecated:` block, so the runtime never fires
them while the audit trail and the ID reservation survive. A validator hard-fails if any ID present in
the previously committed artifact is absent from the new one — catching deletion deterministically
without walking git history.

`data/SCENARIO_CHANGELOG.md` uses `## scenarios-v0.4.0 — YYYY-MM-DD` headings so an RG4-analogue can grep
them, and adopts the same `### Known limitations` numbered-defect discipline as
`data/DATASET_CHANGELOG.md`.

## Alternatives considered

1. **Hash the whole row.** Rejected: a prompt typo fix or an added `notes` line would invalidate every
   clip for that scenario, which trains contributors to ignore staleness warnings.
2. **A hand-declared `semantic_version` per scenario.** Rejected: hand-maintained version fields drift,
   and nothing would detect the drift. Deriving the required bump is strictly stronger.
3. **Policy only, no gate.** Rejected explicitly on this repository's own evidence — an unenforced
   versioning policy is a document nobody reads.
4. **Re-adjudicate all clips on any change.** Rejected as prohibitively expensive; that is exactly what
   the semantic/cosmetic field split exists to avoid.

## Consequences

- Positive: a risk-level downgrade or an added required object surfaces immediately as a set of stale
  clip labels, and blocks the release rather than silently certifying stale behaviour.
- Positive: cosmetic and translation work stays cheap, which keeps the prompt quality loop fast.
- Constraint: the field partition between semantic and cosmetic is itself a design decision that must be
  revisited whenever a field is added. The list lives in one place in `schema.py` and is unit-tested.
- Constraint: withdrawing consent for a clip that is the sole positive evidence for a scenario drops it
  below RG-S2 and blocks the release. That is correct behaviour, and it must be a known consequence
  rather than a surprise.

Related: [ADR-P6-01](ADR-P6-01-per-scenario-yaml-master.md), [ADR-P6-05](ADR-P6-05-class-capability-map.md)

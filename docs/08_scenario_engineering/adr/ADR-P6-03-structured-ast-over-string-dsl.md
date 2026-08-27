# ADR-P6-03 — Triggers Are Structured Predicate Trees Compiled to an AST

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

`src/pipeline/rule_engine.py` evaluates a condition **string** with regexes and `str.split`. The initial
Phase-6 framing was "extend the DSL, don't replace it". Close reading showed that framing to be
impossible and the existing parser to be wrong in four ways:

- `:164` `re.search`es for `any_of([...])` across the whole string and returns at `:167`, *before* the
  AND/OR split at `:170`/`:177`. `any_of([a, b]) AND detected(person)` silently discards the conjunct.
- AND is split before OR, giving AND lower precedence than OR — inverted from standard semantics.
- No parentheses; a leading `(` defeats every `re.match`-anchored predicate, falling through to a warning
  and `False`.
- `NOT` composes only with `detected`, so `NOT any_of(...)` and every future predicate return `False`.

The decisive constraint is different from all of these. `:91` builds `detected_names = {d.class_name ...}`
and `:107` passes only that set into the evaluator. **Bounding boxes and multiplicity are destroyed
before any predicate runs.** `near`, `overlaps`, `above`/`below`, and `count` cannot be implemented
against that signature at all. The evaluator signature must change regardless of which syntax is chosen,
which removes "extend, don't replace" from the table.

The failure mode if this is missed is silent and severe: `near(person, stove, 0.2)` implemented against
the set it was handed degenerates into `detected(person) AND detected(stove)` — semantically identical to
the `knife_near_person` rule being deleted for false positives — while *looking* implemented.

Note also that `docs/02_technical_architecture_specification/rule_engine.md:34` has documented
`near(class_a, class_b, pixel_dist)` as a V1 condition type since Phase 2. This is spec reconciliation,
not new scope.

## Decision

Scenario triggers are authored as **structured YAML predicate trees**, validated against a predicate
registry, and compiled to an AST that the runtime evaluates directly. A one-line string form is *emitted*
into the artifact as a human-readable rendering for the CSV view and for logs, and is **never parsed
back**.

`EvalContext` — detections with bounding boxes and multiplicity, event memory, measured fps, frame id,
deployment room — is defined first, in M1, before any predicate is written. Every predicate is a pure
function of it, unit-tested table-driven with no YAML in the loop.

If a string grammar is ever wanted for authoring, its parser belongs in the **compiler only**. Re-parsing
text 15 times a second on an edge device, to answer a question fully known at compile time, is
indefensible against the 5 ms rule-engine budget at
`docs/02_technical_architecture_specification/performance_budget.md:30`.

## Alternatives considered

1. **Keep the regex evaluator and add branches.** Rejected: cannot express the required predicates at
   all, and each addition is silently `False` until implemented rather than a loud error.
2. **Write a proper recursive-descent/Pratt parser and keep strings at runtime.** A well-understood,
   dependency-free ~120 lines that would fix precedence, parentheses, and generic `NOT`. Rejected only
   because it still parses at runtime and still yields column offsets instead of field paths in errors.
   This remains the fallback if a string authoring format is ever demanded.
3. **An `eval()`-based evaluator.** Rejected on security grounds and blocked mechanically — ruff's `S`
   ruleset is enabled in `pyproject.toml` and S307 fails the build.

## Consequences

- Positive: error messages become
  `SC-KIT-001.yaml: trigger.all[1].args.max_dist must be a float in (0,1], got "0.2m"` instead of a
  column offset — which is the whole ballgame for a stakeholder-authored knowledge base.
- Positive: validation is dict-walking against a registry (~50 lines) rather than parse-then-validate-
  then-map-offsets-back-to-lines.
- Positive: predicates become independently testable pure functions, matching the house style of
  `src/dataset/completeness_policies.py`.
- Constraint: authors write YAML trees rather than one-line conditions. Mitigated by the emitted string
  rendering and by per-field error messages.
- Constraint: the legacy `RuleEngine` and its string DSL remain in the tree until M8 and must not be
  extended in the meantime.

Related: [ADR-P6-04](ADR-P6-04-rule-engine-injection.md), [ADR-P6-07](ADR-P6-07-near-units-and-room-vs-zone.md)

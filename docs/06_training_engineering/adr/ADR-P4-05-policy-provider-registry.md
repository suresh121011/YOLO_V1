# ADR-P4-05 — Pluggable Completeness-Policy Providers

**Status:** Accepted (Phase 4, 2026-07)

## Context

Completeness semantics differ per source *kind*: exhaustive trust lists
(COCO/Open Images/WIDER FACE/Roboflow), verified absence of all classes
(negatives), per-session declarations (custom captures). Phase 5+ will add
dataset types with semantics we cannot predict (e.g. per-region annotations,
crowd-labeled subsets). Hardcoding modes into the generator would force core
edits — and re-review of the hard-fail guarantees — for every addition.

## Decision

Policy modes are classes behind a registry
(`src/dataset/completeness_policies.py`), mirroring the existing
split-strategy registry pattern (`src/dataset/splitting`):

- `CompletenessPolicyProvider` ABC: `resolve_policies(ctx)` → policy key →
  trusted ids; `policy_key_for_image(ctx, filename)` for multi-policy sources.
- `@register_policy_provider("mode")` registers; duplicate modes rejected.
- Built-ins: `trusted_list`, `verified_absence_all`, `per_session`.
- The generator only iterates the registry; an unknown mode in
  `completeness.policies` fails listing the registered modes.

Adding a dataset type = one provider module + one config line; the core
generator, validator, and preflight are untouched.

## Alternatives considered

1. **if/elif on mode strings in the generator** — simplest, rejected: every
   new mode edits validated core logic; providers keep the hard-fail contract
   per-mode and unit-testable in isolation.
2. **Entry-point plugins** (setuptools entry points) — over-engineered for a
   single-repo project; the decorator registry gives the same extensibility
   without packaging machinery. Revisit only if policies must ship out of
   tree.

## Consequences

- Positive: extension without core modification (user requirement);
  per-provider tests document each mode's contract; registry errors
  self-describe the valid vocabulary.
- Negative: slight indirection when reading the generator — mitigated by the
  module docstring's provider table and this ADR.

Related: [ADR-P4-03](ADR-P4-03-completeness-artifact.md).

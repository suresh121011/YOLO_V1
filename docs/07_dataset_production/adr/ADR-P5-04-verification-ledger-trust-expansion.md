# ADR-P5-04 — Verification Ledger as the Single Source of Per-Image Trust Expansion

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Phase 4's completeness artifact masks (image, class) cells whose source does not
trust that class. As humans verify candidates, masking must shrink by exactly
the verified cells — without mutating source policies or the artifact schema.

## Decision

The verification ledger (`data/annotation/verification_ledger.json`) is the
single source of per-image trust expansion. A new completeness policy mode
`trusted_list_with_ledger` (subclass of `TrustedListPolicy`), opted into
**explicitly per source** in `completeness.policies`, resolves the base source
policy PLUS one policy per distinct effective trusted-class set among
ledger-verified images (keyed `"{source}/ledger/{8-hex}"`). `PolicyContext`
gains one additive trailing field `verification_ledger: LedgerLike | None = None`
— verified back-compatible (both construction sites use keyword args). The
`LedgerLike` Protocol lives in `completeness_policies.py` itself to avoid a
dataset→annotation layering inversion. Hard-fails cover every drift: ledger image
absent from provenance, wrong source, class not in taxonomy, taxonomy
fingerprint drift, conflicting verdicts without `supersedes`, `verified_absent`
for a class with delta boxes. The ledger is a `cache: false`, git-tracked,
append-only audit trail, bootstrapped empty in M1.

## Alternatives considered

1. **Rewrite source trusted-class lists as verification grows.** Rejected:
   destroys per-source provenance and makes masking non-reproducible.
2. **A new artifact schema version.** Rejected: the change is purely additive;
   bumping the schema would break the byte-identity regression needlessly.
3. **Infer the ledger policy automatically for all sources.** Rejected: trust
   expansion must be an explicit, auditable opt-in, never a silent default.

## Consequences

- Positive: masking shrinks exactly with verified cells; empty ledger ⇒
  normalized-identical completeness (golden regression); full provenance kept.
- Constraint: policy providers coexist, so rollback is a one-line config flip
  back to `trusted_list`.

Related: [ADR-P5-03](ADR-P5-03-cvat-round-trip.md),
[ADR-P5-05](ADR-P5-05-labels-overlay-no-mutation.md),
[ADR-P4-05](../../06_training_engineering/adr/ADR-P4-05-policy-provider-registry.md)

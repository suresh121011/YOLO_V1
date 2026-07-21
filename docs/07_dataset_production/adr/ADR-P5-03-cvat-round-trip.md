# ADR-P5-03 — CVAT Round-Trip Is the Only Candidates→Labels Path

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17 (privacy finding folded in)

## Context

Candidates ([ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md)) must be
human-verified before they can influence labels. Phase 3 already built a
class-order-verified YOLO importer and dual-annotator IAA machinery for custom
captures; Phase 5 needs a verification loop that reuses, not duplicates, it.

## Decision

The only candidates→labels path is a CVAT round-trip.
`13_build_verification_batches` emits per-batch `preannotations.zip`
(`obj.names` = full ordered 23-class taxonomy, `obj_train_data/<stem>.txt` =
base labels ∪ candidates) plus a `cvat_labels.json` task spec so labels are
created in exact taxonomy order. Humans create the task, upload images + the
zip, correct, and export "YOLO 1.1". `14_import_verified_batch` **reuses**
`read_yolo_export` + `verify_class_order`, hard-fails on any byte-wise edit to a
non-target-class line, extracts deltas for target classes, and records
per-(image,class) verdicts into the ledger. 10 % of each batch is dual-annotated;
`compare_annotators` is reused with gate `min_agreement: 0.70`. Verification
batches mirror the capture-session lifecycle (`created→exported→staged→verified
→imported`). **CVAT must be self-hosted (Docker), version pinned in the runbook**
— custom-capture images never leave the machine (consent posture).

## Alternatives considered

1. **Label Studio / custom web tool.** Rejected: CVAT's YOLO 1.1 export already
   matches the Phase-3 importer; a new tool means new format glue and risk.
2. **Public cvat.ai.** Rejected: uploading custom-home images to a third-party
   host violates the consent posture; self-hosting is mandatory.
3. **Auto-accept, spot-check later.** Rejected: same R30 hallucination-flood
   risk as auto-writing labels (see ADR-P5-01).

## Consequences

- Positive: reuses proven Phase-3 machinery; class-order tamper and trusted-label
  edits are caught by hard-fails; pseudonymous verifier IDs preserve privacy.
- Constraint: a running self-hosted CVAT instance is an operational prerequisite
  for the H-C verification campaign (documented in `verification_runbook.md` §0).

Related: [ADR-P5-01](ADR-P5-01-candidate-artifact-isolation.md),
[ADR-P5-04](ADR-P5-04-verification-ledger-trust-expansion.md)

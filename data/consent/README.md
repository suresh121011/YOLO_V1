# data/consent — Local Consent Records (never committed, never pushed)

This directory holds the **local consent registry** for custom Indian-home
capture sessions. Everything here except this README is gitignored and must
**never** become a DVC output — consent artifacts must not reach git history
or the S3 remote.

## Privacy model

| Artifact | Where it lives | PII? |
|---|---|---|
| Signed consent form (paper/scan) | **Offline**, with the collection lead | Yes — the only place |
| `consent_registry.yaml` (this dir) | Collection machine only | No — pseudonymous IDs only |
| `consent_reference` string | Capture-session manifests (repo/DVC) | No — an ID |

Never write names, addresses, phone numbers or any other personal detail
into the registry. Houses are identified only by pseudonymous IDs (`h01`,
`h02`, …). See `docs/04_dataset_engineering/` §4 (Privacy, PII & DPDP).

## Registry format (`consent_registry.yaml`)

```yaml
# consent_id → record. Read by src/dataset/capture/consent.py
# and src/scenario_engine/clips.py.
CONSENT-h01-2026-001:
  house_id: h01
  granted_on: "2026-07-20"
  scope: dataset-training
  withdrawn: false
```

- `consent_id` must match `consent.reference_pattern` in
  `configs/capture_config.yaml` (default `CONSENT-h{NN}-{YYYY}-{NNN}`).
- The free-text location of the signed form may be recorded in a separate
  private note by the lead — not in the repo.

## Scope vocabulary

`scope` records **what the household agreed to**, and the two media are not
interchangeable.

| Scope | Covers | Enforced by |
|---|---|---|
| `dataset-training` | Still photographs for the training/eval datasets | Format-checked only; image ingest does not read `scope` |
| `scenario-video` | Short scenario clips (10–120 s of continuous video) | `src/scenario_engine/clips.py` — ingest **refuses** any other scope |

**Video needs its own consent and its own signed form.** A still can be
curated frame by frame before it is kept; a 30-second clip cannot, so it will
contain incidental faces, speech in the background, and whatever else was in
the room for that half-minute. A household that agreed to photographs has not
agreed to that, and `dataset-training` consent is therefore rejected at clip
ingest rather than treated as implying it.

Two consequences worth knowing before a capture session:

- A household contributing both media needs **two records** (e.g.
  `CONSENT-h01-2026-001` for images and `CONSENT-h01-2026-002` for clips).
- Clip ingest **requires the registry to be present**. Image ingest degrades to
  a format-only check when the registry is missing; scope cannot be checked
  without it, so clips are only ever ingested on the collection machine.

## Withdrawal

Set `withdrawn: true` on the record. The next
`python scripts/dataset/10_capture_progress.py` run flags every ingested
session that references it; follow the withdrawal SOP in
`docs/04_dataset_engineering/capture_annotation_runbook.md` (remove data,
cut a dataset patch release).

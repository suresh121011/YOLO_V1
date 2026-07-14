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
# consent_id → record. Read by src/dataset/capture/consent.py.
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

## Withdrawal

Set `withdrawn: true` on the record. The next
`python scripts/dataset/10_capture_progress.py` run flags every ingested
session that references it; follow the withdrawal SOP in
`docs/04_dataset_engineering/capture_annotation_runbook.md` (remove data,
cut a dataset patch release).

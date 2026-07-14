# Custom Capture Consent Form (Template)

## Purpose

Template for the signed, offline consent record required before any
Phase-3 custom Indian-home capture session. The signed form itself is
**never digitized into the repository** — only a pseudonymous reference
(`consent_id`) travels into the codebase, in the local
`data/consent/consent_registry.yaml` registry (see
`data/consent/README.md`) and in `CaptureSessionManifest.consent_reference`.

Related: [../04_dataset_engineering/capture_annotation_runbook.md](../04_dataset_engineering/capture_annotation_runbook.md) §1,
[../01_executive_implementation_plan/security_privacy.md](../01_executive_implementation_plan/security_privacy.md)

---

## Form (English)

**Elderly Assistant System — Home Photography Consent**

I understand and agree that:

1. Photos will be taken inside my home for the purpose of training an
   offline AI safety-detection model (household objects: e.g. stove, gas
   cylinder, medicines, walking aids — not people's faces beyond what is
   incidentally visible).
2. These photos will be used only for **model training and evaluation**
   for the Elderly Assistant System project, not for any other purpose,
   and not sold or shared outside this project.
3. Any document (e.g. passport) captured will have personal details
   blurred before use.
4. My home will be identified only by a code (not my name or address) in
   any project record.
5. I can withdraw this consent at any time by contacting the collection
   team; photos already collected will be removed from future dataset
   releases upon withdrawal.
6. Photos are stored without location (GPS) data.

Household code (assigned by collection team, not filled by participant): `h__`

Participant signature: ________________________  Date: __________

Collection lead signature: ________________________

*(This signed form is kept offline by the collection lead — it is never
uploaded, emailed unencrypted, or committed to the repository.)*

---

## Form (Hindi placeholder — V2)

Hindi translation of the above is planned alongside the V2 Hindi TTS/UI
work (see `docs/01_executive_implementation_plan/roadmap.md`). Until then,
collection requires a verbal explanation in the participant's preferred
language by the collection lead, confirmed before signature.

---

## After signing: registry entry

The collection lead creates a `consent_id` (format
`CONSENT-h{NN}-{YYYY}-{NNN}`) and adds a record to the **local, gitignored**
`data/consent/consent_registry.yaml` — no name, address, or contact detail,
only the pseudonymous house code and dates:

```yaml
CONSENT-h01-2026-001:
  house_id: h01
  granted_on: "2026-07-20"
  scope: dataset-training
  withdrawn: false
```

This `consent_id` is then passed to `08_ingest_capture_session.py
--consent-ref` for every session captured at that house.

---

Previous: [annotation_guide.md](./annotation_guide.md)

Related: [../04_dataset_engineering/capture_annotation_runbook.md](../04_dataset_engineering/capture_annotation_runbook.md)

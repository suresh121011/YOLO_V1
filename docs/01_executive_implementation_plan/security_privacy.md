# Security & Privacy Principles

## Purpose

Defines privacy-by-design commitments, security controls, and regulatory readiness for the Elderly Assistant System.

## Dependencies

Reads:
- product_vision.md

Used By:
- recommendations.md
- engineering_standards.md

Related:
- risk_register.md (R17)

---

## Privacy Architecture

```mermaid
graph LR
    CAM3["Camera"] --> PROC["On-Device Processing Only"]
    PROC --> ALERT2["Alert Spoken Locally"]
    PROC --> LOG2["Local SQLite Log No PII stored"]
    LOG2 -->|"User-initiated WiFi sync only"| OPT["Optional: Caregiver Dashboard"]
    
    INTERNET["No Internet Required"] -.->|"No automatic data upload"| PROC
    CLOUD["No Cloud Storage Used"]-. ->|"Zero images uploaded"| PROC

    style INTERNET fill:#e94560,stroke:#fff,color:#fff
    style CLOUD fill:#e94560,stroke:#fff,color:#fff
    style OPT fill:#533483,stroke:#e94560,color:#fff
```

## Privacy-by-Design Commitments (V1)

| Principle | Implementation | V1 Status |
|:----------|:--------------|:----------|
| **No cloud dependency** | 100% on-device inference; no API calls | Core requirement |
| **No facial recognition** | `face` class used only for scene context, never identified | By design |
| **No biometric identification** | No facial embedding, voice print, or gait analysis | By design |
| **No unnecessary data retention** | Event logs contain object classes + timestamps; no raw images | V1 design |
| **User consent** | System must be explicitly enabled by user or caregiver | Operational requirement |
| **Offline-first architecture** | System functions fully without any network | Core requirement |
| **Secure model storage** | Model weights stored in app-sandboxed directory | V1 standard |
| **Configuration protection** | Risk rules YAML read-only in production build | V1 basic |
| **Input validation** | Camera frame validation before inference | V1 standard |
| **Safe exception handling** | All failures logged; no exception exposes internal state | V1 standard |
| **Encrypted logs** | Log encryption at rest | V2 enhancement |
| **DPDP compliance readiness** | India Digital Personal Data Protection Act compliance | V2 legal review |

## Security Controls Summary

| Area | V1 Control | V2+ Enhancement |
|:-----|:-----------|:----------------|
| Data in transit | N/A (no internet) | TLS 1.3 if cloud sync added |
| Data at rest | App-sandboxed logs | AES-256 encrypted logs |
| Model integrity | File hash validation on load | Signed model artifacts |
| Config protection | Read-only YAML | Authenticated config API |
| Camera access | OS-level permission | User-revocable per session |

---

Previous: [validation_strategy.md](./validation_strategy.md)

Next: [dataset_governance.md](./dataset_governance.md)

Related: [risk_register.md](./risk_register.md)

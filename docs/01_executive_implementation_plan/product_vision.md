# Product Vision & Mission

## Purpose

Defines the mission statement, product vision, and core design principles for the Elderly Assistant System.

## Dependencies

Reads:
- SUMMARY.md

Used By:
- business_goals.md
- architecture_overview.md

Related:
- security_privacy.md

---

## Mission Statement

> *"To give every elderly person living in an Indian home a silent, always-on safety companion — one that watches, understands, and speaks — without ever invading their privacy."*

## Product Vision

```mermaid
graph LR
    A["Elderly Person Living Alone"] --> B["AI Camera Always Watching"]
    B --> C["Detects Hazards: Knife · Stove · Wire · Wet Floor"]
    C --> D["Speaks Naturally: 'Please be careful, there is a wet floor ahead'"]
    D --> E["Person is Safe. Family Has Peace of Mind."]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style D fill:#533483,stroke:#e94560,color:#fff
    style E fill:#2b2d42,stroke:#8d99ae,color:#fff
```

## Core Design Principles

| Principle | Implementation |
|:----------|:--------------|
| **Privacy by Default** | Zero data leaves device; no facial recognition; no biometric storage |
| **Safety First** | False negatives (missed hazards) are worse than false positives |
| **Accessibility** | Clear spoken guidance; no screen interaction needed |
| **Indian Context** | Trained on Indian homes, objects, lighting, and layouts |
| **Graceful Degradation** | Works in reduced capacity even if subsystems fail |
| **Non-intrusive** | Alert cooldowns prevent alert fatigue |

---

Previous: [SUMMARY.md](./SUMMARY.md)

Next: [business_goals.md](./business_goals.md)

Related: [security_privacy.md](./security_privacy.md)

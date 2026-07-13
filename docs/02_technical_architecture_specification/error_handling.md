# Error Handling & Graceful Degradation

## Purpose

Defines 5 degradation levels and error recovery actions for every component.

## Dependencies

Reads:
- threading_model.md
- performance_budget.md

Used By:
- architecture_decisions.md

Related:
- ../01_executive_implementation_plan/risk_register.md

---

## Degradation Levels

```mermaid
graph TD
    FULL["Level 0: Full Operation\nYOLO + Memory + VLM + Rules + TTS + Logging"]
    NO_VLM["Level 1: No VLM\nYOLO + Memory + Rules + TTS + Logging\n(VLM disabled or failed)"]
    NO_TTS["Level 2: No TTS\nYOLO + Memory + Rules + Beep Alert + Logging\n(TTS failed)"]
    NO_LOG["Level 3: No Logging\nYOLO + Memory + Rules + TTS\n(SQLite unavailable)"]
    MINIMAL["Level 4: Minimal\nYOLO + Rules + TTS\n(Memory failed — rules still work)"]
    FAIL["Level 5: Fail-safe\nLog error · Alert caregiver · Stop gracefully"]

    FULL -->|"VLM OOM/Timeout"| NO_VLM
    NO_VLM -->|"TTS process crash"| NO_TTS
    NO_TTS -->|"Disk full"| NO_LOG
    NO_LOG -->|"Memory error"| MINIMAL
    MINIMAL -->|"YOLO model load fail"| FAIL

    style FULL fill:#2b2d42,stroke:#8d99ae,color:#fff
    style NO_VLM fill:#0f3460,stroke:#e94560,color:#fff
    style NO_TTS fill:#16213e,stroke:#0f3460,color:#fff
    style MINIMAL fill:#533483,stroke:#e94560,color:#fff
    style FAIL fill:#e94560,stroke:#fff,color:#fff
```

## Error Handling Table

| Component | Failure Mode | Recovery Action | Degrades To |
|:----------|:-------------|:----------------|:------------|
| YOLO detector | Model file not found | Log CRITICAL, exit | Level 5 |
| YOLO inference | Runtime exception | Log ERROR, skip frame | No impact |
| Event Memory | Out of memory | Shrink window size by 50% | Level 4 |
| SmolVLM2 | OOM or timeout | Disable VLM, use flag | Level 1 |
| Rule Engine | YAML parse error | Use previous rules | No impact |
| Alert Queue | Overflow | Drop oldest INFO alerts | No impact |
| Piper TTS | Process crash | Restart process, use beep fallback | Level 2 |
| SQLite logger | Disk full | Disable logging, continue | Level 3 |
| Camera | Feed lost | Watchdog restart, TTS notify | Paused |

## Watchdog Thread

Optional 5th thread monitors health of all components every 5 seconds. Restarts crashed threads automatically (max 3 retries before escalation).

---

Previous: [threading_model.md](./threading_model.md)

Next: [data_contracts.md](./data_contracts.md)

Related: [../01_executive_implementation_plan/risk_register.md](../01_executive_implementation_plan/risk_register.md)

# System Architecture Overview

## Purpose

High-level pipeline architecture, component summary, deployment architecture, and performance budget for executive understanding.

## Dependencies

Reads:
- project_scope.md

Used By:
- implementation_phases.md
- risk_register.md

Related:
- ../02_technical_architecture_specification/system_architecture.md
- ../02_technical_architecture_specification/data_flow.md

---

## High-Level Pipeline

```mermaid
graph TD
    CAM["Camera Feed 30 FPS Input"] --> YOLO["YOLO11n Object Detection 23 classes less than 25ms"]
    YOLO --> MEM["Event Memory Temporal Context Sliding Window"]
    MEM --> VLM["SmolVLM2 Scene Understanding Every 5th frame"]
    VLM --> RULE["Rule Engine Safety Logic YAML-configured"]
    RULE --> ALERT["Alert Queue Priority + Cooldown"]
    ALERT --> TTS["Piper TTS Voice Guidance less than 500ms"]
    TTS --> USER["Elderly User Receives Guidance"]
    YOLO --> LOG["Structured Logger Events + Metrics"]
    RULE --> LOG
    LOG --> ACTIVE["Active Learning Continuous Improvement"]

    style CAM fill:#1a1a2e,stroke:#e94560,color:#fff
    style YOLO fill:#16213e,stroke:#0f3460,color:#fff
    style MEM fill:#16213e,stroke:#0f3460,color:#fff
    style VLM fill:#0f3460,stroke:#e94560,color:#fff
    style RULE fill:#0f3460,stroke:#e94560,color:#fff
    style ALERT fill:#e94560,stroke:#fff,color:#fff
    style TTS fill:#533483,stroke:#e94560,color:#fff
    style USER fill:#2b2d42,stroke:#8d99ae,color:#fff
    style LOG fill:#1a1a2e,stroke:#8d99ae,color:#fff
    style ACTIVE fill:#1a1a2e,stroke:#8d99ae,color:#fff
```

## Component Summary

| Component | Technology | Purpose | V1 Status |
|:----------|:-----------|:--------|:----------|
| Object Detection | YOLO11n (Ultralytics) | Real-time hazard detection | Core |
| Scene Analysis | SmolVLM2-256/500M | Context understanding | Optional/Core |
| Safety Logic | Rule Engine (YAML) | Decision making | Core |
| Voice Guidance | Piper TTS (neural) | Spoken alerts | Core |
| Event Memory | Python (sliding window) | Temporal context | Core |
| Structured Logging | JSON / SQLite | Audit trail | Core |
| Dataset Versioning | DVC + Git | Reproducibility | Core |

## Deployment Architecture

```mermaid
graph TD
    subgraph "Edge Device Phone or Pi or Smart Glasses"
        CAM2["Camera"] --> PRE["Preprocess 640x640"]
        PRE --> DET["YOLO11n TFLite or ONNX"]
        DET --> FILT["Confidence Filter greater than 0.25"]
        FILT --> RULES["Rule Engine"]
        RULES --> SPEECH["Piper TTS"]
        SPEECH --> SPK["Speaker Output"]
        FILT -->|"Every 5th frame"| VLM2["SmolVLM2-256M if hardware allows"]
        VLM2 --> RULES
        RULES --> SQLLOG["SQLite Log"]
    end

    subgraph "Optional WiFi Only"
        SQLLOG -->|"WiFi sync"| DASH["Caregiver Dashboard"]
        DASH --> RETRAIN["Active Learning Pipeline"]
    end
```

## Performance Budget (V1 Targets)

| Stage | Budget | Priority |
|:------|:-------|:---------|
| Camera capture | < 5 ms | Fixed |
| YOLO11n inference (CPU) | < 150 ms | Hard limit |
| YOLO11n inference (GPU/NPU) | < 25 ms | Target |
| Event Memory update | < 2 ms | Fixed |
| Rule Engine evaluation | < 5 ms | Fixed |
| SmolVLM2 inference | < 2,000 ms | Optional (5th frame) |
| Piper TTS synthesis | < 500 ms | Hard limit |
| **End-to-end (no VLM)** | **< 500 ms** | **Target** |
| **End-to-end (with VLM)** | **< 2,000 ms** | **Acceptable** |

> For detailed component specifications, see [../02_technical_architecture_specification/](../02_technical_architecture_specification/README.md)

---

Previous: [project_scope.md](./project_scope.md)

Next: [implementation_phases.md](./implementation_phases.md)

Related: [../02_technical_architecture_specification/system_architecture.md](../02_technical_architecture_specification/system_architecture.md)

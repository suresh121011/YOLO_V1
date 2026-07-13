# System Architecture

## Purpose

Project folder structure and runtime component architecture diagram.

## Dependencies

Reads: None (entry point for technical docs)

Used By:
- data_flow.md
- interfaces.md
- all component documents

Related:
- ../01_executive_implementation_plan/architecture_overview.md

---

## Project Structure

```
YOLO_V1/
├── configs/                          # All configuration files
│   ├── data.yaml                     # YOLO dataset configuration
│   ├── feature_flags.yaml            # Runtime feature toggles
│   ├── training/
│   │   ├── yolo11n_config.yaml       # Nano model hyperparameters
│   │   └── yolo11s_config.yaml       # Small model hyperparameters
│   └── deployment/
│       ├── onnx_config.yaml          # ONNX export settings
│       └── tflite_config.yaml        # TFLite quantization settings
│
├── data/                             # All dataset files (DVC tracked)
│   ├── raw/
│   │   ├── coco_filtered/
│   │   ├── openimages_filtered/
│   │   ├── roboflow_imports/
│   │   ├── wider_face/
│   │   └── custom_captures/
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── qa_reports/
│
├── scripts/
│   ├── dataset/                      # Data acquisition and processing
│   ├── qa/                           # Quality assurance checks
│   ├── training/                     # Model training and export
│   ├── inference/                    # Inference and benchmarking
│   └── utils/                        # Conversion and visualization
│
├── src/
│   ├── pipeline/                     # Core pipeline components
│   │   ├── detector.py
│   │   ├── event_memory.py
│   │   ├── scene_analyzer.py
│   │   ├── rule_engine.py
│   │   ├── alert_queue.py
│   │   ├── tts_engine.py
│   │   ├── orchestrator.py
│   │   └── confidence_fusion.py
│   ├── config/                       # Config loading and validation
│   ├── logging/                      # Event and metrics logging
│   └── plugins/                      # Plugin system
│
├── models/                           # Trained model weights and exports
├── tests/                            # Unit, integration, and performance tests
├── docs/                             # Documentation (you are here)
├── dvc.yaml                          # DVC pipeline definition
├── requirements.txt                  # Python dependencies
└── README.md
```

## Runtime Component Architecture

```mermaid
graph TD
    subgraph "Input Layer"
        CAMERA["Camera Source\n(OpenCV VideoCapture)"]
        PREPROC["Frame Preprocessor\n(Resize · Normalize · Validate)"]
    end

    subgraph "Detection Layer"
        YOLO["YOLO11n Detector\n(Ultralytics / ONNX / TFLite)"]
        FILTER["Detection Filter\n(conf > threshold · NMS)"]
    end

    subgraph "Memory Layer"
        MEM["Event Memory\n(Sliding Window · Class Tracking)"]
    end

    subgraph "Intelligence Layer"
        VLM["SmolVLM2 Analyzer\n(256M / 500M / 2.2B)\n[Feature-Flag Gated]"]
        FUSION["Confidence Fusion\n(YOLO score + VLM score → final)"]
        RULES["Rule Engine\n(YAML-driven · Stateful)"]
    end

    subgraph "Output Layer"
        QUEUE["Alert Queue\n(Priority · Cooldown · Dedup)"]
        TTS["Piper TTS Engine\n(Offline Neural Speech)"]
        SPEAKER["Audio Output"]
    end

    subgraph "Observability Layer"
        EVLOG["Event Logger\n(JSON / SQLite)"]
        ALLOG["Active Learning Logger\n(Low-confidence mining)"]
        METRICS["Metrics Collector\n(FPS · Latency · Memory)"]
    end

    CAMERA --> PREPROC --> YOLO --> FILTER --> MEM
    MEM --> VLM
    VLM --> FUSION
    FILTER --> FUSION
    FUSION --> RULES
    RULES --> QUEUE --> TTS --> SPEAKER
    FILTER --> EVLOG
    RULES --> EVLOG
    FILTER --> ALLOG
    YOLO --> METRICS

    style CAMERA fill:#1a1a2e,stroke:#e94560,color:#fff
    style YOLO fill:#16213e,stroke:#0f3460,color:#fff
    style MEM fill:#16213e,stroke:#0f3460,color:#fff
    style VLM fill:#0f3460,stroke:#e94560,color:#fff
    style RULES fill:#0f3460,stroke:#e94560,color:#fff
    style QUEUE fill:#e94560,stroke:#fff,color:#fff
    style TTS fill:#533483,stroke:#e94560,color:#fff
```

---

Previous: None (start here)

Next: [data_flow.md](./data_flow.md)

Related: [interfaces.md](./interfaces.md)

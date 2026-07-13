# Executive Summary

## Purpose

One-page summary of the Elderly Assistant System for executive stakeholders.

## Dependencies

Reads: None (entry point)

Used By:
- product_vision.md
- business_goals.md

Related:
- architecture_overview.md

---

The **Elderly Assistant System** is an offline, privacy-first AI vision platform designed to detect safety hazards in Indian homes and provide real-time spoken guidance to elderly residents living alone or semi-independently.

The system uses a **YOLO11n object detection model** to identify 23 safety-relevant objects (knives, wet floors, gas cylinders, medicines, wires, etc.) and triggers natural-language voice alerts through **Piper TTS** when risks are detected. A lightweight vision-language model (**SmolVLM2**) provides situational awareness for complex scenarios. All processing occurs **on-device** — no cloud dependency, no data leaving the home.

## Why This Matters

| Dimension | Context |
|:----------|:--------|
| **Market** | India has 140M+ elderly (60+). Solo living is rising. Home accidents cause ~50,000 fatalities annually |
| **Gap** | No affordable, privacy-respecting, Indian-home-specific AI safety assistant exists |
| **Opportunity** | First-mover advantage in AI-assisted elderly safety for Indian families |
| **Differentiator** | Works offline · Speaks naturally · Understands Indian-home objects · No biometric collection |

## V1 Scope Summary

- ✅ **23 object classes** covering all critical home safety scenarios
- ✅ **Real-time detection** at ≥15 FPS on mobile hardware
- ✅ **Voice alerts** in clear, natural English (Indian English voice)
- ✅ **Rule-based safety logic** with per-alert cooldown timers
- ✅ **Structured event logging** for caregiver review
- ✅ **100% offline operation** — works without internet

---

Previous: None (start here)

Next: [product_vision.md](./product_vision.md)

Related: [architecture_overview.md](./architecture_overview.md)

# Future Modules Reference

## Purpose

V2 and V3 planned module specifications for engineering roadmap planning.

## Dependencies

Reads:
- python_examples.md

Used By: None (terminal reference document)

Related:
- ../01_executive_implementation_plan/roadmap.md

---

## 16.1 V2 Planned Modules

Phase-6 status added in M9. Three of these six are now either **done differently** or **rejected**, and
saying so here matters more than the estimates: this table is where someone looks before starting work.

| Module | Purpose | Priority | Estimated Effort | Phase-6 status |
|:-------|:--------|:---------|:----------------|:---|
| `fall_detector.py` | Pose-based fall detection | High | 2–3 weeks | **Rejected** for these 23 static-object classes — see [../08_scenario_engineering/negative_register.md](../08_scenario_engineering/negative_register.md). Needs pose, which the taxonomy does not carry |
| `activity_classifier.py` | Sitting/standing/cooking/sleeping classification | Medium | 2 weeks | Unchanged. The VLM's `activity` string is advisory only and gates no decision |
| `hindi_tts_plugin.py` | Hindi voice support for Piper | High | 1 week | **No plugin needed.** M9 wired `runtime.tts_language` to the per-scenario `messages` map; drop in a `hi_IN` voice and set the key. The sketch below subclasses `PiperTTS`, which is no longer the shape of the problem |
| `caregiver_sync.py` | WiFi event sync to dashboard | Medium | 2 weeks | Partly built: `BaseCaregiverSink` + a local sink ship in M9 (ADR-P6-11). What remains is only the transport, and it implements that Protocol rather than `BasePlugin` — the plugin seam is still not wired |
| `multi_person_tracker.py` | Track multiple people with unique IDs | Medium | 2–3 weeks | Deliberately absent, with conditions — see [../08_scenario_engineering/integration_strategy.md](../08_scenario_engineering/integration_strategy.md) §2. A tracker arrives with the scenarios that need it and the evidence it is reliable enough |
| `medication_scheduler.py` | Cross-reference medicine detections with schedule | Low | 1–2 weeks | **Rejected.** Adherence is unobservable: seeing a medicine strip is not seeing a dose taken, and `medicine_reminder` was deleted for the double-dose path it created |

### V2 Module: `hindi_tts_plugin.py` — Quick Reference

```python
# Extension of PiperTTS — switch model based on language flag
class HindiTTSPlugin(PiperTTS):
    def __init__(self, config: dict):
        hindi_model = config["tts"]["hindi_model_path"]
        super().__init__(model_path=hindi_model, ...)

    def speak_hi(self, text_hi: str) -> None:
        """Speak Hindi text via Piper hi_IN-medium model."""
        self.speak(text_hi)
```

### V2 Module: `caregiver_sync.py` — Quick Reference

```python
# WiFi-only sync — never internet
class CaregiverSyncPlugin(BasePlugin):
    def on_alert(self, alert: Alert) -> None:
        """Push alert JSON to local WiFi caregiver dashboard."""
        if wifi_available():
            post_to_dashboard(alert.to_dict())
```

---

## 16.2 V3 Planned Modules

| Module | Purpose | Priority | Estimated Effort |
|:-------|:--------|:---------|:----------------|
| `smart_glasses_adapter.py` | Hardware abstraction for smart glasses | High | 3–4 weeks |
| `emergency_caller.py` | Auto-call caregiver on CRITICAL sustained alert | High | 2 weeks |
| `regional_tts_plugin.py` | Tamil, Telugu, Bengali, Marathi TTS | Medium | 2 weeks per language |
| `on_device_finetuner.py` | Personalization via on-device fine-tuning | Low | 4 weeks |
| `cloud_dashboard.py` | Full caregiver dashboard web app | Medium | 4–6 weeks |
| `ir_camera_adapter.py` | Infrared camera integration for night vision | Medium | 2 weeks |

### V3 Module: `smart_glasses_adapter.py` — Quick Reference

```python
# Hardware abstraction layer for smart glasses input/output
class SmartGlassesAdapter:
    """Replaces OpenCV camera source with smart glasses camera stream.
    Replaces speaker output with bone conduction audio."""

    def get_frame(self) -> np.ndarray: ...     # From glasses camera
    def speak(self, text: str) -> None: ...    # Via bone conduction
    def show_overlay(self, text: str) -> None: ...  # Via AR display
```

---

## 16.3 Architecture Extension Points

When building V2/V3 modules:

1. **New hardware source** → Implement `BaseCameraAdapter` interface, register in `orchestrator.py`
2. **New alert output** → Implement `BasePlugin.on_alert()`, register in `PluginRegistry`
3. **New safety class** → Add to taxonomy (increment major version), retrain, update rules YAML
4. **New TTS language** → Add Piper model, extend `PiperTTS` as plugin, add feature flag
5. **New rule type** → Extend `RuleEngine._evaluate_condition()` parser, add to `risk_rules.yaml`

---

*Document prepared by: Software Engineering Team*

*For executive context: see [../01_executive_implementation_plan/README.md](../01_executive_implementation_plan/README.md)*

*For architecture: see [../02_technical_architecture_specification/README.md](../02_technical_architecture_specification/README.md)*

---

Previous: [troubleshooting.md](./troubleshooting.md)

Next: None (end of engineering appendix)

Related: [../01_executive_implementation_plan/roadmap.md](../01_executive_implementation_plan/roadmap.md)

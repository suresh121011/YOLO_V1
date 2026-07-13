# Plugin Architecture

## Purpose

Extension system for adding capabilities without modifying core pipeline. V1 ships with the registry; no plugins required for core operation.

## Dependencies

Reads:
- interfaces.md
- orchestrator.md

Used By: None (extension point)

Related:
- feature_flags.md

---

**Location:** `src/plugins/`

## Plugin Interface

```python
class BasePlugin(ABC):
    plugin_name: str = "unnamed"
    plugin_version: str = "0.0.1"
    enabled: bool = True

    def on_startup(self, config: dict) -> None: ...
    def on_detection(self, detections: list[Detection], frame_id: int) -> None: ...
    def on_alert(self, alert: Alert) -> None: ...
    def on_frame(self, frame: np.ndarray, frame_id: int) -> None: ...
    def on_shutdown(self) -> None: ...
```

## Plugin Registry

```python
class PluginRegistry:
    def register(self, plugin: BasePlugin) -> None: ...
    def load_from_config(self, config_path: str) -> None: ...
    def broadcast_detection(self, detections: list[Detection]) -> None: ...
    def broadcast_alert(self, alert: Alert) -> None: ...
```

## Example V2 Plugins (Future)

| Plugin | Purpose |
|:-------|:--------|
| `CaregiverSyncPlugin` | WiFi sync of events to caregiver dashboard |
| `HindiTTSPlugin` | Hindi language fallback for TTS |
| `FallDetectionPlugin` | Pose-based fall detection |
| `MedSchedulePlugin` | Medication schedule cross-reference |

---

Previous: [orchestrator.md](./orchestrator.md)

Next: [structured_logging.md](./structured_logging.md)

Related: [feature_flags.md](./feature_flags.md)

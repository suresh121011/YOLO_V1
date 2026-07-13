# Python Implementation Examples

## Purpose

Core Python implementation snippets for all pipeline components.

## Dependencies

Reads:
- yaml_examples.md

Used By:
- dataset_templates.md
- training_scripts.md

Related:
- ../02_technical_architecture_specification/interfaces.md
- ../02_technical_architecture_specification/data_contracts.md

---

## 2.1 YOLO Detector

```python
# src/pipeline/detector.py

import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from . import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO11n object detector with class-specific confidence thresholds.

    Supports PyTorch (.pt), ONNX (.onnx), and TFLite (.tflite) formats.
    """

    DEFAULT_CONF = 0.25
    DEFAULT_IOU  = 0.45

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONF,
        class_thresholds: Optional[dict[str, float]] = None,
        expected_hash: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.class_thresholds = class_thresholds or {}

        if expected_hash:
            self._verify_hash(expected_hash)

        self.model = YOLO(str(self.model_path))
        logger.info(f"Loaded YOLO model: {self.model_path}")

    def _verify_hash(self, expected: str) -> None:
        sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if sha256 != expected:
            raise ValueError(f"Model hash mismatch: {sha256} != {expected}")

    def warmup(self, n: int = 3) -> None:
        """Run inference on dummy frames to initialize GPU/NPU."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, verbose=False)
        logger.info("YOLO warmup complete")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        """Run inference on a single BGR frame."""
        import time
        ts = time.time() * 1000

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=DEFAULT_IOU if hasattr(self, 'DEFAULT_IOU') else 0.45,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self.model.names[class_id]
                conf = float(box.conf.item())

                # Apply per-class threshold override
                min_conf = self.class_thresholds.get(class_name, self.conf_threshold)
                if conf < min_conf:
                    continue

                xyxy = box.xyxyn[0].tolist()  # Normalized [x1, y1, x2, y2]
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=BoundingBox(cx=cx, cy=cy, w=w, h=h),
                    frame_id=frame_id,
                    timestamp_ms=ts,
                ))

        return detections
```

---

## 2.2 Event Memory

```python
# src/pipeline/event_memory.py

import threading
from collections import defaultdict, deque
from typing import Optional
import time

from . import Detection, MemoryEntry, BoundingBox


class EventMemory:
    """Sliding-window temporal object tracking.

    Tracks which classes have been detected across the last N frames,
    enabling temporal rules like "stove unattended for 30 seconds".
    """

    def __init__(self, window_size: int = 150):
        self._window = window_size
        self._frames: deque[set[int]] = deque(maxlen=window_size)
        self._entries: dict[int, MemoryEntry] = {}
        self._frame_counter = 0
        self._lock = threading.Lock()

    def update(self, detections: list[Detection]) -> None:
        """Advance window with the latest frame's detections."""
        with self._lock:
            self._frame_counter += 1
            detected_ids = {d.class_id for d in detections}
            self._frames.append(detected_ids)

            for det in detections:
                if det.class_id not in self._entries:
                    self._entries[det.class_id] = MemoryEntry(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        last_seen_frame=self._frame_counter,
                        first_seen_frame=self._frame_counter,
                        detection_count=1,
                        avg_confidence=det.confidence,
                        last_bbox=det.bbox,
                    )
                else:
                    entry = self._entries[det.class_id]
                    entry.last_seen_frame = self._frame_counter
                    entry.detection_count += 1
                    alpha = 0.9
                    entry.avg_confidence = (alpha * entry.avg_confidence +
                                            (1 - alpha) * det.confidence)
                    entry.last_bbox = det.bbox

    def is_present(self, class_id: int) -> bool:
        """Was this class detected in the most recent frame?"""
        if not self._frames:
            return False
        return class_id in self._frames[-1]

    def frames_since_seen(self, class_id: int) -> int:
        """How many frames ago was this class last seen?"""
        with self._lock:
            entry = self._entries.get(class_id)
            if entry is None:
                return self._window  # Never seen
            return self._frame_counter - entry.last_seen_frame

    def seconds_since_seen(self, class_id: int, fps: float) -> float:
        frames = self.frames_since_seen(class_id)
        return frames / max(fps, 1.0)

    def is_absent_for(self, class_id: int, seconds: float, fps: float) -> bool:
        return self.seconds_since_seen(class_id, fps) >= seconds

    def get_entry(self, class_id: int) -> Optional[MemoryEntry]:
        return self._entries.get(class_id)
```

---

## 2.3 Rule Engine

```python
# src/pipeline/rule_engine.py

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from . import Alert, Detection, Severity, SceneContext
from .event_memory import EventMemory

logger = logging.getLogger(__name__)


class RuleEngine:
    """YAML-driven safety rule evaluation engine."""

    def __init__(self, rules_path: str):
        self._rules_path = Path(rules_path)
        self._lock = threading.RLock()
        self._cooldowns: dict[str, float] = {}
        self._rules: list[dict] = []
        self._load_rules()

    def _load_rules(self) -> None:
        with self._lock:
            data = yaml.safe_load(self._rules_path.read_text())
            self._rules = data.get("rules", [])
            logger.info(f"Loaded {len(self._rules)} rules from {self._rules_path}")

    def reload_rules(self) -> None:
        """Hot-reload rules without restarting the pipeline."""
        self._load_rules()

    def evaluate(
        self,
        detections: list[Detection],
        memory: EventMemory,
        context: Optional[SceneContext] = None,
        current_fps: float = 15.0,
    ) -> list[Alert]:
        """Evaluate all rules against current detections and memory state."""
        now = time.monotonic()
        detected_names = {d.class_name for d in detections}
        alerts: list[Alert] = []

        for rule in self._rules:
            rule_id = rule["id"]
            cooldown = rule.get("cooldown_seconds", 60)

            # Check cooldown
            last_fired = self._cooldowns.get(rule_id, 0.0)
            if now - last_fired < cooldown:
                continue

            if self._evaluate_condition(rule["condition"], detected_names,
                                        memory, current_fps):
                severity = Severity[rule.get("severity", "INFO")]
                alert = Alert(
                    rule_id=rule_id,
                    severity=severity,
                    message=rule["message_en"],
                    message_hi=rule.get("message_hi"),
                    triggering_detections=detections,
                    timestamp_ms=time.time() * 1000,
                    cooldown_seconds=cooldown,
                    frame_id=0,
                )
                alerts.append(alert)
                self._cooldowns[rule_id] = now
                logger.info(f"Alert fired: {rule_id} [{severity.name}]")

        return alerts

    def _evaluate_condition(
        self,
        condition: str,
        detected_names: set[str],
        memory: EventMemory,
        fps: float,
    ) -> bool:
        """Simple condition parser (production should use proper AST parser)."""
        # This is a simplified evaluator — see references for full implementation
        if "detected(" in condition:
            import re
            detected_calls = re.findall(r'detected\((\w+)\)', condition)
            absent_calls = re.findall(r'absent_for\((\w+),\s*(\d+)\)', condition)

            # All detected() calls must be satisfied
            for cls in detected_calls:
                if f"NOT detected({cls})" in condition:
                    if cls in detected_names:
                        return False
                elif cls not in detected_names:
                    return False

            for cls, seconds in absent_calls:
                if not memory.is_absent_for(cls, float(seconds), fps):
                    return False

            return True
        return False
```

---

## 2.4 Piper TTS Engine

```python
# src/pipeline/tts_engine.py

import logging
import queue
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

TTS_QUEUE_MAX = 5


class PiperTTS:
    """Non-blocking Piper neural TTS engine.

    Maintains a background thread that consumes from the alert queue
    and synthesizes speech — main inference loop is never blocked.
    """

    def __init__(self, model_path: str, config_path: str, speech_rate: float = 0.9):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.speech_rate = speech_rate

        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=TTS_QUEUE_MAX)
        self._shutdown = threading.Event()
        self._speaking = threading.Event()
        self._healthy = self._health_check()

        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts")
        self._thread.start()

    def _health_check(self) -> bool:
        try:
            result = subprocess.run(["piper", "--help"], capture_output=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Piper TTS not found — falling back to beep")
            return False

    def speak(self, text: str, priority: bool = False) -> None:
        """Enqueue text for speech (non-blocking)."""
        if not text.strip():
            return
        prio = 0 if priority else 1
        try:
            self._queue.put_nowait((prio, text))
        except queue.Full:
            logger.warning("TTS queue full, dropping message")

    def _worker(self) -> None:
        """Background thread: consume queue and speak."""
        while not self._shutdown.is_set():
            try:
                _, text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._speaking.set()
            try:
                self._synthesize_and_play(text)
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}")
                self._beep()
            finally:
                self._speaking.clear()

    def _synthesize_and_play(self, text: str) -> None:
        cmd = [
            "piper",
            "--model", str(self.model_path),
            "--config", str(self.config_path),
            "--output_file", "-",
            "--length_scale", str(1.0 / self.speech_rate),
        ]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        audio_data, stderr = process.communicate(input=text.encode("utf-8"), timeout=5)
        if process.returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr.decode()}")
        self._play_wav(audio_data)

    def _play_wav(self, wav_bytes: bytes) -> None:
        try:
            import sounddevice as sd
            import numpy as np
            import io, wave
            with wave.open(io.BytesIO(wav_bytes)) as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                sd.play(audio, wf.getframerate(), blocking=True)
        except ImportError:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes); f.flush()
                if os.name == "nt":
                    os.system(f'start /min "" "{f.name}"')
                else:
                    os.system(f'aplay "{f.name}" 2>/dev/null')

    def _beep(self) -> None:
        try:
            import sounddevice as sd, numpy as np
            t = np.linspace(0, 0.5, int(22050 * 0.5), endpoint=False)
            tone = (np.sin(2 * np.pi * 800 * t) * 32767 * 0.5).astype(np.int16)
            sd.play(tone, 22050, blocking=True)
        except Exception:
            print("\a")

    def is_speaking(self) -> bool: return self._speaking.is_set()
    def health_check(self) -> bool: return self._healthy and self._thread.is_alive()
    def shutdown(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=5)
```

---

## 2.5 Confidence Fusion

```python
# src/pipeline/confidence_fusion.py

from typing import Optional
from . import Detection, SceneContext


class ConfidenceFusion:
    """Fuse YOLO detection confidence with VLM risk assessment.

    Formula: fused = (alpha * yolo_conf) + (beta * vlm_risk_score)

    Safety constraint: fused score can only INCREASE severity,
    never decrease a rule-engine-triggered alert.
    """

    VLM_RISK_MAP = {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta

    def fuse(
        self, detections: list[Detection], context: Optional[SceneContext]
    ) -> list[Detection]:
        if context is None:
            return detections

        vlm_risks: dict[str, float] = {}
        for risk in context.risks:
            risk_type = risk.get("risk_type", "").lower()
            severity = risk.get("severity", "").lower()
            vlm_risks[risk_type] = self.VLM_RISK_MAP.get(severity, 0.0)

        fused = []
        for det in detections:
            vlm_score = vlm_risks.get(det.class_name, 0.0)
            new_conf = (self.alpha * det.confidence) + (self.beta * vlm_score)
            new_conf = max(det.confidence, min(1.0, new_conf))  # Only increase
            fused.append(Detection(
                class_id=det.class_id, class_name=det.class_name,
                confidence=new_conf, bbox=det.bbox,
                frame_id=det.frame_id, timestamp_ms=det.timestamp_ms,
            ))

        return fused
```

---

Previous: [yaml_examples.md](./yaml_examples.md)

Next: [dataset_templates.md](./dataset_templates.md)

Related: [../02_technical_architecture_specification/interfaces.md](../02_technical_architecture_specification/interfaces.md)

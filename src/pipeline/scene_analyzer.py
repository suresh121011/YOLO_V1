"""
SmolVLM2 Scene Analyzer
========================
Wraps SmolVLM2 (HuggingFace) for contextual scene understanding.
Invoked every Nth frame to reduce compute; result is cached between invocations.

Model variants:
  SmolVLM2-256M  — low-end Android / Raspberry Pi (recommended for deployment)
  SmolVLM2-500M  — flagship Android
  SmolVLM2-2.2B  — development / evaluation only

Safety constraint:
  VLM output is advisory. The Rule Engine fires on YOLO detections first.
  VLM can boost confidence via ConfidenceFusion but cannot suppress a rule alert.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from . import Detection, SceneContext

logger = logging.getLogger(__name__)

SCENE_PROMPT = """You are an elderly home safety assistant.
The camera has detected these objects: {detections}

Analyze the scene and respond ONLY with valid JSON:
{{
  "activity": "<brief description of what the person is doing>",
  "risks": [
    {{
      "risk_type": "<object or class name matching the detection>",
      "severity": "<low|medium|high|critical>",
      "description": "<one sentence about the risk>"
    }}
  ],
  "recommendations": ["<one spoken sentence guidance>"]
}}

Focus on safety risks relevant to elderly people. Keep recommendations short and clear."""


class SmolVLM2Analyzer:
    """Scene contextual analyzer using SmolVLM2.

    Runs inference every N frames (configurable). Returns None when unavailable
    so the pipeline degrades gracefully to rule-only mode.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        max_new_tokens: int = 300,
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            model_name:      HuggingFace model identifier or local path.
            max_new_tokens:  Maximum tokens in VLM response.
            timeout_seconds: Discard stale context older than this threshold.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

        self._model: Any = None
        self._processor: Any = None
        self._available = False

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load model. Sets _available=False on failure (graceful)."""
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info(f"Loading SmolVLM2: {self.model_name} ...")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()
            self._available = True
            logger.info(f"SmolVLM2 loaded on {device}")
        except Exception as e:
            logger.warning(f"SmolVLM2 not available: {e}. Pipeline runs in YOLO-only mode.")
            self._available = False

    def is_available(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._available

    def analyze(
        self,
        frame: Any,
        detections: list[Detection],
        frame_id: int = 0,
    ) -> SceneContext | None:
        """Run scene analysis on a frame.

        Args:
            frame:      BGR numpy array from OpenCV.
            detections: Current YOLO detections (used to build prompt).
            frame_id:   Current frame index for traceability.

        Returns:
            SceneContext on success; None on failure (graceful degradation).
        """
        if not self._available:
            return None

        if not detections:
            return None  # Nothing to analyze

        start_ms = time.time() * 1000

        try:
            import cv2
            from PIL import Image

            # Build detection summary for prompt
            det_summary = ", ".join(f"{d.class_name} ({d.confidence:.0%})" for d in detections)
            prompt = SCENE_PROMPT.format(detections=det_summary)

            # Convert BGR → PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inputs = self._processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )

            import torch

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            response = self._processor.decode(output_ids[0], skip_special_tokens=True)
            inference_time_ms = time.time() * 1000 - start_ms

            return self._parse_response(response, frame_id, inference_time_ms)

        except Exception as e:
            logger.warning(f"SmolVLM2 inference failed: {e}")
            return None

    def _parse_response(
        self,
        response: str,
        frame_id: int,
        inference_time_ms: float,
    ) -> SceneContext | None:
        """Parse JSON response from VLM into typed SceneContext."""
        try:
            # Extract JSON block from response
            start = response.rfind("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON block found in VLM response")

            data = json.loads(response[start:end])
            return SceneContext(
                activity=data.get("activity", "unknown"),
                risks=data.get("risks", []),
                recommendations=data.get("recommendations", []),
                raw_response=response,
                inference_time_ms=inference_time_ms,
                frame_id=frame_id,
            )
        except Exception as e:
            logger.warning(f"VLM response parse error: {e} | response: {response[:100]}")
            return None

"""
Confidence Fusion
==================
Fuses YOLO detection confidence with SmolVLM2 scene risk assessment
to reduce false positives on safety-critical classes.

Safety invariant: fused confidence can only INCREASE relative to YOLO baseline.
VLM context is advisory — it cannot cancel a rule-engine-triggered alert.

Algorithm:
    fused_conf = clip(alpha * yolo_conf + beta * vlm_risk_score, yolo_conf, 1.0)

where:
    alpha = 0.7 (YOLO weight)
    beta  = 0.3 (VLM weight)
    vlm_risk_score: 0.3 (low) | 0.5 (medium) | 0.8 (high) | 1.0 (critical)
"""

from __future__ import annotations

import logging
from typing import Optional

from . import BoundingBox, Detection, SceneContext

logger = logging.getLogger(__name__)


# Safety-critical classes eligible for confidence fusion.
# Non-safety classes are not fused (VLM rarely comments on them).
SAFETY_CLASSES = frozenset({
    "knife",
    "stove",
    "gas_cylinder",
    "wet_floor",
    "wire",
    "medicine_strip",
    "medicine_bottle",
})

# Map from VLM severity string to numeric score
VLM_RISK_SCORE: dict[str, float] = {
    "low":      0.3,
    "medium":   0.5,
    "high":     0.8,
    "critical": 1.0,
}


class ConfidenceFusion:
    """Fuse YOLO detection confidence with VLM risk assessment.

    For safety-critical detections, blends YOLO's confidence with
    any matching risk from SmolVLM2, biased toward increasing recall.

    Args:
        alpha: Weight for YOLO confidence (default 0.7).
        beta:  Weight for VLM risk score (default 0.3).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError("alpha and beta must be in [0, 1]")
        self.alpha = alpha
        self.beta = beta

    def fuse(
        self,
        detections: list[Detection],
        context: Optional[SceneContext],
    ) -> list[Detection]:
        """Return detections with fused confidence scores.

        Non-safety classes and detections without matching VLM risk are returned unchanged.
        Fused confidence is always >= original YOLO confidence (never downgrades).

        Args:
            detections: Raw YOLO detections for the current frame.
            context: SmolVLM2 scene context (may be None if VLM is disabled).

        Returns:
            List of Detection objects with updated confidence values.
        """
        if context is None:
            return detections  # Nothing to fuse

        # Build a lookup: risk_type → vlm_risk_score for this frame
        vlm_risks: dict[str, float] = {}
        for risk in context.risks:
            risk_type = risk.get("risk_type", "").lower().replace(" ", "_")
            severity_str = risk.get("severity", "").lower()
            if risk_type and severity_str in VLM_RISK_SCORE:
                vlm_risks[risk_type] = VLM_RISK_SCORE[severity_str]

        if not vlm_risks:
            return detections  # VLM found no risks to fuse

        fused: list[Detection] = []
        for det in detections:
            if det.class_name not in SAFETY_CLASSES:
                fused.append(det)
                continue

            vlm_score = vlm_risks.get(det.class_name, 0.0)
            if vlm_score == 0.0:
                fused.append(det)
                continue

            raw_fused = (self.alpha * det.confidence) + (self.beta * vlm_score)
            # Safety invariant: only increase confidence, never decrease
            new_conf = min(1.0, max(det.confidence, raw_fused))

            if new_conf != det.confidence:
                logger.debug(
                    f"Fusion: {det.class_name} conf {det.confidence:.3f} → {new_conf:.3f} "
                    f"(vlm_score={vlm_score:.2f})"
                )

            fused.append(Detection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=new_conf,
                bbox=det.bbox,
                frame_id=det.frame_id,
                timestamp_ms=det.timestamp_ms,
            ))

        return fused

"""
Unit tests for src.logging.structured_logger.

Focused on the privacy contract: person/face geometry must never reach
logs/events.jsonl. ``log_frame`` has always redacted bounding boxes, but
``log_alert`` wrote ``alert.explanation`` verbatim — which becomes a real leak
as soon as spatial predicates record coordinates there.

See docs/08_scenario_engineering/adr/ADR-P6-09-alert-contract-extension.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.logging.structured_logger import BBOX_REDACT_CLASSES, StructuredLogger
from src.pipeline import Alert, BoundingBox, Detection, Severity


def _alert(explanation: dict) -> Alert:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(cx=0.41, cy=0.62, w=0.2, h=0.5),
        frame_id=3,
        timestamp_ms=0.0,
    )
    return Alert(
        rule_id="test_rule",
        severity=Severity.HIGH,
        message="The floor near the sink looks wet.",
        message_hi=None,
        triggering_detections=[detection],
        timestamp_ms=0.0,
        cooldown_seconds=60,
        frame_id=3,
        explanation=explanation,
    )


class TestRedactExplanation:
    @pytest.mark.unit
    def test_drops_keys_naming_redacted_classes(self) -> None:
        result = StructuredLogger.redact_explanation(
            {
                "person_center": [0.41, 0.62],
                "face_bbox": [0.1, 0.2, 0.3, 0.4],
                "dist_to_stove": 0.08,
                "rule_id": "wet_floor_hazard",
            }
        )
        assert result == {"dist_to_stove": 0.08, "rule_id": "wet_floor_hazard"}

    @pytest.mark.unit
    def test_matches_on_word_boundaries_only(self) -> None:
        """`persistence` must survive; `person` and `person_center` must not."""
        result = StructuredLogger.redact_explanation(
            {"persistence": 1, "personal_best": 2, "person": 3, "last_person_bbox": 4}
        )
        assert result == {"persistence": 1, "personal_best": 2}

    @pytest.mark.unit
    def test_redacts_at_any_nesting_depth(self) -> None:
        result = StructuredLogger.redact_explanation(
            {"spatial": {"near": [{"person_bbox": [0.1], "stove_bbox": [0.2]}]}}
        )
        assert result == {"spatial": {"near": [{"stove_bbox": [0.2]}]}}

    @pytest.mark.unit
    def test_scalars_pass_through(self) -> None:
        assert StructuredLogger.redact_explanation("plain") == "plain"
        assert StructuredLogger.redact_explanation(None) is None

    @pytest.mark.unit
    def test_every_redacted_class_is_covered(self) -> None:
        payload = {f"{name}_center": [0.5, 0.5] for name in BBOX_REDACT_CLASSES}
        assert StructuredLogger.redact_explanation(payload) == {}


class TestLogAlert:
    @pytest.mark.unit
    def test_written_alert_carries_no_person_geometry(self, tmp_path: Path) -> None:
        logger = StructuredLogger(log_dir=str(tmp_path))
        logger.log_alert(_alert({"person_center": [0.41, 0.62], "condition": "detected(knife)"}))

        events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(events[-1])

        assert entry["type"] == "alert"
        assert entry["rule_id"] == "test_rule"
        assert entry["explanation"] == {"condition": "detected(knife)"}
        assert "0.41" not in json.dumps(entry)

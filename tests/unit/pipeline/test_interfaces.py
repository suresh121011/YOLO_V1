"""
Unit tests for src.pipeline data contracts (__init__.py).

Tests the BoundingBox, Detection, Alert, and other dataclasses
defined in src/pipeline/__init__.py.

These tests validate the data contract layer only — no ML model required.
"""
from __future__ import annotations

import pytest

from src.pipeline import Alert, BoundingBox, Detection, FrameResult, PipelineMetrics, Severity


class TestSeverity:
    """Test Severity enum ordering."""

    @pytest.mark.unit
    def test_severity_ordering(self) -> None:
        """CRITICAL should be greater than INFO."""
        assert Severity.CRITICAL > Severity.HIGH
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.MEDIUM > Severity.LOW
        assert Severity.LOW > Severity.INFO

    @pytest.mark.unit
    def test_severity_values(self) -> None:
        """Severity integer values should match spec."""
        assert Severity.INFO == 0
        assert Severity.LOW == 1
        assert Severity.MEDIUM == 2
        assert Severity.HIGH == 3
        assert Severity.CRITICAL == 4


class TestBoundingBox:
    """Test BoundingBox geometric operations."""

    @pytest.mark.unit
    def test_area(self) -> None:
        """Area should equal w * h."""
        bbox = BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.5)
        assert bbox.area() == pytest.approx(0.20, rel=1e-6)

    @pytest.mark.unit
    def test_to_xyxy(self) -> None:
        """Center-format bbox should convert correctly to xyxy."""
        bbox = BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.4)
        x1, y1, x2, y2 = bbox.to_xyxy()
        assert x1 == pytest.approx(0.3)
        assert y1 == pytest.approx(0.3)
        assert x2 == pytest.approx(0.7)
        assert y2 == pytest.approx(0.7)

    @pytest.mark.unit
    def test_iou_identical(self) -> None:
        """IoU of identical boxes should be 1.0."""
        bbox = BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.4)
        assert bbox.iou(bbox) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.unit
    def test_iou_no_overlap(self) -> None:
        """IoU of non-overlapping boxes should be 0.0."""
        bbox_a = BoundingBox(cx=0.1, cy=0.1, w=0.1, h=0.1)
        bbox_b = BoundingBox(cx=0.9, cy=0.9, w=0.1, h=0.1)
        assert bbox_a.iou(bbox_b) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_iou_partial_overlap(self) -> None:
        """IoU of partially overlapping boxes should be between 0 and 1."""
        bbox_a = BoundingBox(cx=0.4, cy=0.5, w=0.4, h=0.4)
        bbox_b = BoundingBox(cx=0.6, cy=0.5, w=0.4, h=0.4)
        iou = bbox_a.iou(bbox_b)
        assert 0.0 < iou < 1.0


class TestDetection:
    """Test Detection dataclass."""

    @pytest.mark.unit
    def test_detection_creation(self) -> None:
        """Detection should be constructable with required fields."""
        det = Detection(
            class_id=5,
            class_name="knife",
            confidence=0.85,
            bbox=BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.3),
            frame_id=42,
            timestamp_ms=1234567890.0,
        )
        assert det.class_name == "knife"
        assert det.confidence == pytest.approx(0.85)
        assert det.frame_id == 42

"""
Shared fixtures for unit tests.

Unit test rules:
    - No real file I/O (use tmp_path fixture or in-memory objects)
    - No real model inference (mock YOLODetector)
    - No real camera (use synthetic frames)
    - No real TTS (mock PiperTTS)
    - All tests complete in < 100ms
"""

from __future__ import annotations

import pytest

from src.pipeline import (
    BoundingBox,
    Detection,
)

# ─── Sample Data Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """A normalized bounding box in the center of the frame."""
    return BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.3)


@pytest.fixture
def detection_knife(sample_bbox: BoundingBox) -> Detection:
    """A high-confidence knife detection."""
    return Detection(
        class_id=5,
        class_name="knife",
        confidence=0.85,
        bbox=sample_bbox,
        frame_id=1,
        timestamp_ms=1000.0,
    )


@pytest.fixture
def detection_person(sample_bbox: BoundingBox) -> Detection:
    """A high-confidence person detection."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=BoundingBox(cx=0.6, cy=0.5, w=0.3, h=0.6),
        frame_id=1,
        timestamp_ms=1000.0,
    )


@pytest.fixture
def detection_stove(sample_bbox: BoundingBox) -> Detection:
    """A stove detection."""
    return Detection(
        class_id=6,
        class_name="stove",
        confidence=0.75,
        bbox=sample_bbox,
        frame_id=1,
        timestamp_ms=1000.0,
    )


@pytest.fixture
def detection_wet_floor(sample_bbox: BoundingBox) -> Detection:
    """A wet floor detection."""
    return Detection(
        class_id=20,
        class_name="wet_floor",
        confidence=0.60,
        bbox=sample_bbox,
        frame_id=1,
        timestamp_ms=1000.0,
    )

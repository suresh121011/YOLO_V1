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

import shutil
from pathlib import Path

import pytest
import yaml

from src.pipeline import (
    BoundingBox,
    Detection,
)

_SCENARIO_DIR = Path("configs/scenarios")


# ─── Scenario Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def active_scenarios(tmp_path: Path) -> Path:
    """A copy of the authored scenarios, promoted to `active` for testing only.

    Every scenario in `configs/scenarios/` ships as `draft`, and the schema
    refuses `status: active` without `reviewed_by`/`reviewed_on` — a clinical
    reviewer, not an engineer. So the runtime engine cannot be exercised against
    the repository set as it stands, which is correct and must stay correct.

    The fixture takes a *copy* into tmp_path and promotes it there. Nothing under
    `configs/` is touched, so the gate cannot be eroded by a test needing it out
    of the way, and the tests that assert the real refusal still see the real
    directory. Rejected scenarios stay rejected — promoting those would be
    inventing capability the taxonomy cannot support.
    """
    destination = tmp_path / "scenarios"
    shutil.copytree(_SCENARIO_DIR, destination)

    for path in sorted(destination.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("status") != "draft":
            continue  # leave `rejected` rows exactly as authored
        raw["status"] = "active"
        raw["reviewed_by"] = "test-fixture-not-a-clinician"
        raw["reviewed_on"] = "2026-08-11"
        path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
    return destination


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

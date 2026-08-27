"""
Unit tests for src.pipeline.detector.

Pins the two threshold/privacy defects fixed in Phase 6 M1
(docs/08_scenario_engineering/architecture_review.md §3, D2 and D5).

Follows the house pattern of faking at our own module boundary
(``YOLODetector._load_model``) rather than mocking ``ultralytics.YOLO``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.pipeline.detector import TaxonomyMismatchError, YOLODetector

# ─── Fake model ───────────────────────────────────────────────────────────────


class _FakeTensor:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeCoords:
    def __init__(self, xyxyn: list[float]) -> None:
        self._xyxyn = xyxyn

    def __getitem__(self, index: int) -> _FakeCoords:
        return self

    def tolist(self) -> list[float]:
        return self._xyxyn


class _FakeBox:
    def __init__(self, class_id: int, conf: float) -> None:
        self.cls = _FakeTensor(class_id)
        self.conf = _FakeTensor(conf)
        self.xyxyn = _FakeCoords([0.4, 0.4, 0.6, 0.6])


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    """Records the `conf` it was called with and filters like Ultralytics does."""

    def __init__(self, names: dict[int, str], boxes: list[_FakeBox]) -> None:
        self.names = names
        self._boxes = boxes
        self.last_conf: float | None = None

    def predict(self, frame: Any, conf: float, iou: float, verbose: bool) -> list[_FakeResult]:
        self.last_conf = conf
        # Ultralytics applies `conf` BEFORE the caller sees any box. That
        # pre-filter is precisely what made low per-class thresholds unreachable.
        kept = [b for b in self._boxes if b.conf.item() >= conf]
        return [_FakeResult(kept)]


_NAMES = {0: "person", 5: "knife", 8: "passport", 20: "wet_floor"}


@pytest.fixture
def model_file(tmp_path: Path) -> str:
    path = tmp_path / "fake.pt"
    path.write_bytes(b"not-a-real-model")
    return str(path)


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str,
    boxes: list[_FakeBox],
    **kwargs: Any,
) -> tuple[YOLODetector, _FakeModel]:
    fake = _FakeModel(_NAMES, boxes)
    monkeypatch.setattr(YOLODetector, "_load_model", lambda self: fake)
    detector = YOLODetector(model_path=model_file, **kwargs)
    return detector, fake


# ─── D5: the confidence floor ─────────────────────────────────────────────────


class TestPredictConfidenceFloor:
    @pytest.mark.unit
    def test_predict_uses_loosest_per_class_threshold(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        """Passing the global 0.25 made knife's 0.20 unreachable."""
        detector, fake = _detector(monkeypatch, model_file, [], class_thresholds={"knife": 0.20})
        detector.detect(object(), frame_id=1)
        assert fake.last_conf == pytest.approx(0.20)
        assert detector.conf_threshold == pytest.approx(0.25)

    @pytest.mark.unit
    def test_low_confidence_safety_detection_survives(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        """A 0.22-confidence knife must pass a 0.20 per-class threshold."""
        detector, _ = _detector(
            monkeypatch,
            model_file,
            [_FakeBox(class_id=5, conf=0.22)],
            class_thresholds={"knife": 0.20},
        )
        detections = detector.detect(object(), frame_id=1)
        assert [d.class_name for d in detections] == ["knife"]

    @pytest.mark.unit
    def test_per_class_threshold_still_filters_stricter(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        """Loosening the predict floor must not loosen the real per-class cut."""
        detector, _ = _detector(
            monkeypatch,
            model_file,
            [_FakeBox(class_id=8, conf=0.35)],
            class_thresholds={"knife": 0.20, "passport": 0.40},
        )
        assert detector.detect(object(), frame_id=1) == []

    @pytest.mark.unit
    def test_floor_never_exceeds_global_threshold(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        detector, fake = _detector(monkeypatch, model_file, [], class_thresholds={"passport": 0.9})
        detector.detect(object(), frame_id=1)
        assert fake.last_conf <= 0.25


# ─── D2: class gating is a real guarantee ─────────────────────────────────────


class TestDisabledClasses:
    @pytest.mark.unit
    def test_disabled_class_never_becomes_a_detection(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        """`passport: false  # privacy` was enforced nowhere before Phase 6."""
        detector, _ = _detector(
            monkeypatch,
            model_file,
            [_FakeBox(class_id=8, conf=0.95), _FakeBox(class_id=5, conf=0.95)],
            disabled_classes={"passport"},
        )
        detections = detector.detect(object(), frame_id=1)
        assert [d.class_name for d in detections] == ["knife"]

    @pytest.mark.unit
    def test_suppression_beats_even_a_perfect_score(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        detector, _ = _detector(
            monkeypatch,
            model_file,
            [_FakeBox(class_id=8, conf=1.0)],
            disabled_classes={"passport"},
        )
        assert detector.detect(object(), frame_id=1) == []

    @pytest.mark.unit
    def test_no_suppression_by_default(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        detector, _ = _detector(monkeypatch, model_file, [_FakeBox(class_id=8, conf=0.95)])
        assert [d.class_name for d in detector.detect(object(), frame_id=1)] == ["passport"]


# ─── Detection shape ──────────────────────────────────────────────────────────


class TestDetectionMapping:
    @pytest.mark.unit
    def test_xyxyn_maps_to_normalized_centre_format(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        detector, _ = _detector(monkeypatch, model_file, [_FakeBox(class_id=0, conf=0.9)])
        detection = detector.detect(object(), frame_id=42)[0]
        assert detection.bbox.cx == pytest.approx(0.5)
        assert detection.bbox.cy == pytest.approx(0.5)
        assert detection.bbox.w == pytest.approx(0.2)
        assert detection.bbox.h == pytest.approx(0.2)
        assert detection.frame_id == 42
        assert detection.class_id == 0


# ─── M9: the weights must carry the taxonomy the system reasons over ──────────


class TestTaxonomyVerification:
    @pytest.mark.unit
    def test_matching_taxonomy_loads(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        detector, _ = _detector(monkeypatch, model_file, [], expected_classes=dict(_NAMES))
        assert detector.class_names == _NAMES

    @pytest.mark.unit
    def test_a_renamed_class_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        expected = {**_NAMES, 5: "cutting_knife"}
        with pytest.raises(TaxonomyMismatchError, match="cutting_knife"):
            _detector(monkeypatch, model_file, [], expected_classes=expected)

    @pytest.mark.unit
    def test_renumbering_is_refused_even_with_the_same_names(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        """The R24 decision holds id 20 reserved without renumbering. A model
        that shuffles ids while keeping every name would otherwise sail through
        and mislabel every detection it makes."""
        shuffled = {0: "person", 5: "passport", 8: "knife", 20: "wet_floor"}
        with pytest.raises(TaxonomyMismatchError, match="id collisions"):
            _detector(monkeypatch, model_file, [], expected_classes=shuffled)

    @pytest.mark.unit
    def test_a_missing_class_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str
    ) -> None:
        expected = {**_NAMES, 22: "support_handle"}
        with pytest.raises(TaxonomyMismatchError, match="missing from the model"):
            _detector(monkeypatch, model_file, [], expected_classes=expected)

    @pytest.mark.unit
    def test_no_expectation_loads_but_says_so(
        self, monkeypatch: pytest.MonkeyPatch, model_file: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Permissive by default for experiments — but never silently."""
        with caplog.at_level("WARNING"):
            _detector(monkeypatch, model_file, [])
        assert "UNVERIFIED" in caplog.text

    @pytest.mark.unit
    def test_list_style_names_are_normalised(self) -> None:
        """Ultralytics hands back a dict; exported formats sometimes a list."""

        class _ListNames:
            names = ["person", "face"]

        assert YOLODetector.model_class_names(_ListNames()) == {0: "person", 1: "face"}

    @pytest.mark.unit
    def test_the_shipped_taxonomy_is_what_gets_checked(self) -> None:
        """The guard is worthless if SystemConfig hands the detector nothing."""
        from src.config.config_loader import SystemConfig

        config = SystemConfig.load()
        assert len(config.class_names) == 23
        assert config.class_names[0] == "person"
        assert config.class_names[20] == "wet_floor"

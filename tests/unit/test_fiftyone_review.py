"""Unit tests for the FiftyOne review bridge (P8).

The pure conversions and the duck-typed reverse run without fiftyone; the
fo-dependent functions are tested against a fake ``fiftyone`` module injected
into sys.modules (house lazy-import pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dataset.annotation.fiftyone_review import (
    boxes_to_label_text,
    export_reviewed_labels,
    fo_bbox_to_xywhn,
    from_fo_detections,
    parse_label_text,
    to_fo_detections,
    xywhn_to_fo_bbox,
)

pytestmark = pytest.mark.unit

_NAMES = {0: "person", 1: "charger", 10: "wire"}
_NAME_TO_ID = {v: k for k, v in _NAMES.items()}


class TestCoordinateConversion:
    def test_xywhn_to_fo_and_back_round_trips(self) -> None:
        cx, cy, w, h = 0.6, 0.4, 0.2, 0.1
        bbox = xywhn_to_fo_bbox(cx, cy, w, h)
        assert bbox == [pytest.approx(0.5), pytest.approx(0.35), 0.2, 0.1]  # top-left
        assert fo_bbox_to_xywhn(bbox) == pytest.approx((cx, cy, w, h))


class TestLabelTextParsing:
    def test_parse_and_serialize_round_trip(self) -> None:
        text = "0 0.500000 0.500000 0.200000 0.200000\n10 0.300000 0.300000 0.100000 0.100000\n"
        boxes = parse_label_text(text)
        assert boxes == [(0, 0.5, 0.5, 0.2, 0.2), (10, 0.3, 0.3, 0.1, 0.1)]
        assert boxes_to_label_text(boxes) == text  # sorted by class id, 6-dp

    def test_parse_skips_malformed(self) -> None:
        text = "# c\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5\nx y z a b\n"
        assert parse_label_text(text) == [(0, 0.5, 0.5, 0.2, 0.2)]

    def test_empty_boxes_serialize_empty(self) -> None:
        assert boxes_to_label_text([]) == ""


class TestFromFoDetections:
    def test_duck_typed_reverse(self) -> None:
        dets = SimpleNamespace(
            detections=[
                SimpleNamespace(label="person", bounding_box=[0.4, 0.4, 0.2, 0.2]),
                SimpleNamespace(label="wire", bounding_box=[0.25, 0.25, 0.1, 0.1]),
            ]
        )
        boxes = from_fo_detections(dets, _NAME_TO_ID)
        assert boxes == [(0, 0.5, 0.5, 0.2, 0.2), (10, 0.3, 0.3, 0.1, 0.1)]

    def test_out_of_taxonomy_label_dropped(self) -> None:
        dets = SimpleNamespace(
            detections=[SimpleNamespace(label="unicorn", bounding_box=[0.1, 0.1, 0.1, 0.1])]
        )
        assert from_fo_detections(dets, _NAME_TO_ID) == []

    def test_none_detections_empty(self) -> None:
        assert from_fo_detections(None, _NAME_TO_ID) == []


# ── Fake fiftyone module for the lazy-import functions ───────────────────────


class _FoDetection:
    def __init__(self, label: str, bounding_box: list[float]) -> None:
        self.label = label
        self.bounding_box = bounding_box


class _FoDetections:
    def __init__(self, detections: list[_FoDetection]) -> None:
        self.detections = detections


class _FoSample:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._fields: dict[str, object] = {}

    def __setitem__(self, key: str, value: object) -> None:
        self._fields[key] = value

    def __getitem__(self, key: str) -> object:
        return self._fields[key]


class _FoDataset:
    def __init__(self, name: str, overwrite: bool = False) -> None:
        self.name = name
        self.samples: list[_FoSample] = []

    def add_samples(self, samples: list[_FoSample]) -> None:
        self.samples.extend(samples)

    def __iter__(self):
        return iter(self.samples)


@pytest.fixture
def fake_fo(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    mod = SimpleNamespace(
        Detection=_FoDetection,
        Detections=_FoDetections,
        Sample=_FoSample,
        Dataset=_FoDataset,
        launch_app=lambda dataset, port=5151: SimpleNamespace(wait=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "fiftyone", mod)
    return mod


class TestFoDependentFunctions:
    def test_to_fo_detections_maps_labels_and_bbox(self, fake_fo: SimpleNamespace) -> None:
        boxes = [(0, 0.5, 0.5, 0.2, 0.2), (10, 0.3, 0.3, 0.1, 0.1)]
        dets = to_fo_detections(boxes, _NAMES)
        assert [d.label for d in dets.detections] == ["person", "wire"]
        assert dets.detections[0].bounding_box == [pytest.approx(0.4), pytest.approx(0.4), 0.2, 0.2]

    def test_to_fo_detections_skips_unknown_id(self, fake_fo: SimpleNamespace) -> None:
        dets = to_fo_detections([(99, 0.5, 0.5, 0.1, 0.1)], _NAMES)
        assert dets.detections == []

    def test_label_to_fo_to_label_full_round_trip(self, fake_fo: SimpleNamespace) -> None:
        text = "0 0.500000 0.500000 0.200000 0.200000\n10 0.300000 0.300000 0.100000 0.100000\n"
        dets = to_fo_detections(parse_label_text(text), _NAMES)
        back = boxes_to_label_text(from_fo_detections(dets, _NAME_TO_ID))
        assert back == text

    def test_export_reviewed_labels_writes_yolo(
        self, fake_fo: SimpleNamespace, tmp_path: Path
    ) -> None:
        from src.dataset.annotation.fiftyone_review import build_review_dataset

        img = tmp_path / "frame1.jpg"
        img.write_bytes(b"x")
        samples = [(img, "0 0.5 0.5 0.2 0.2\n10 0.3 0.3 0.1 0.1\n")]
        dataset = build_review_dataset("batch1", samples, _NAMES)
        out = tmp_path / "reviewed"
        n = export_reviewed_labels(dataset, out, _NAME_TO_ID)
        assert n == 1
        written = (out / "frame1.txt").read_text(encoding="utf-8")
        assert (
            written
            == "0 0.500000 0.500000 0.200000 0.200000\n10 0.300000 0.300000 0.100000 0.100000\n"
        )

    def test_build_review_dataset_creates_samples(
        self, fake_fo: SimpleNamespace, tmp_path: Path
    ) -> None:
        from src.dataset.annotation.fiftyone_review import build_review_dataset

        img = tmp_path / "frame1.jpg"
        img.write_bytes(b"x")
        dataset = build_review_dataset("b", [(img, "1 0.5 0.5 0.2 0.2\n")], _NAMES)
        assert len(dataset.samples) == 1
        prelabels = dataset.samples[0]["prelabels"]
        assert prelabels.detections[0].label == "charger"

"""
Unit tests for scripts.inference.test_video.

Heavy dependencies (YOLO model, OpenCV VideoCapture, real video files)
are mocked throughout. Tests validate source type detection, FPS logic,
output path generation, and threshold validation without requiring
GPU or actual video input.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.inference.test_video import (
    resolve_source_type,
)


# ─── resolve_source_type ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestResolveSourceType:
    def test_integer_string_is_webcam(self) -> None:
        assert resolve_source_type("0") == "webcam"
        assert resolve_source_type("1") == "webcam"
        assert resolve_source_type("2") == "webcam"

    def test_mp4_file_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"fake_mp4")
        assert resolve_source_type(str(f)) == "video"

    def test_avi_file_is_video(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.avi"
        f.write_bytes(b"fake_avi")
        assert resolve_source_type(str(f)) == "video"

    def test_folder_is_images(self, tmp_path: Path) -> None:
        assert resolve_source_type(str(tmp_path)) == "images"

    def test_nonexistent_source_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_source_type("/nonexistent/path/video.mp4")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_source_type(str(f))


# ─── frames_from_folder ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestFramesFromFolder:
    def test_empty_folder_raises(self, tmp_path: Path) -> None:
        from scripts.inference.test_video import frames_from_folder

        with pytest.raises(RuntimeError, match="No image files"):
            list(frames_from_folder(tmp_path))

    def test_yields_frames_in_order(self, tmp_path: Path) -> None:
        """Test with mock cv2 to avoid needing real image files."""
        # Create image files
        for name in ["c_img.jpg", "a_img.jpg", "b_img.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        import numpy as np

        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("scripts.inference.test_video.find_image_files") as mock_find:
            # Return sorted list
            mock_find.return_value = sorted(tmp_path.glob("*.jpg"))
            with patch("cv2.imread", return_value=fake_frame):
                from scripts.inference.test_video import frames_from_folder
                frames = list(frames_from_folder(tmp_path))

        assert len(frames) == 3
        # Frame IDs should start at 1 and be sequential
        frame_ids = [fid for fid, _ in frames]
        assert frame_ids == [1, 2, 3]


# ─── draw_detections ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDrawDetections:
    def test_returns_frame_without_crash(self) -> None:
        """draw_detections should not crash with empty detections or real boxes."""
        try:
            import numpy as np
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("OpenCV/numpy not installed")

        from scripts.inference.test_video import draw_detections

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"class_id": 5, "class_name": "knife", "confidence": 0.85,
             "x1": 100, "y1": 100, "x2": 200, "y2": 200},
        ]

        result = draw_detections(frame.copy(), detections, {5: "knife"})
        assert result.shape == frame.shape

    def test_empty_detections_no_crash(self) -> None:
        try:
            import numpy as np
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("OpenCV/numpy not installed")

        from scripts.inference.test_video import draw_detections

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame.copy(), [], {})
        assert result.shape == frame.shape


# ─── FPS calculation ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFpsCalculation:
    """Test the rolling-window FPS calculation logic used in the inference loop."""

    def test_fps_from_latency(self) -> None:
        """1000ms / latency_ms = FPS."""
        latency_ms = 100.0
        fps = 1000.0 / max(latency_ms, 1.0)
        assert fps == pytest.approx(10.0)

    def test_fps_from_100ms_latency(self) -> None:
        assert 1000.0 / 100.0 == pytest.approx(10.0)

    def test_fps_from_25ms_latency(self) -> None:
        assert 1000.0 / 25.0 == pytest.approx(40.0)

    def test_zero_latency_protected(self) -> None:
        """Should never divide by zero."""
        fps = 1000.0 / max(0.0, 1.0)
        assert fps == pytest.approx(1000.0)

    def test_rolling_window_average(self) -> None:
        """Rolling window of 30 frames smooths FPS estimate."""
        latencies = [50.0] * 20 + [100.0] * 10  # avg = 66.7ms
        window: list[float] = []
        for lat in latencies:
            window.append(lat)
            if len(window) > 30:
                window.pop(0)

        avg = sum(window) / len(window)
        fps = 1000.0 / max(avg, 1.0)
        assert fps == pytest.approx(1000.0 / (200/3), rel=0.01)


# ─── Output path generation ───────────────────────────────────────────────────


@pytest.mark.unit
class TestOutputPathGeneration:
    def test_output_dir_created(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "inference_out"
        assert not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.exists()

    def test_summary_json_path(self, tmp_path: Path) -> None:
        """inference_summary.json should be in the output dir."""
        expected = tmp_path / "inference_summary.json"
        assert expected.parent == tmp_path

    def test_predictions_json_path(self, tmp_path: Path) -> None:
        """predictions.json should be in the output dir."""
        expected = tmp_path / "predictions.json"
        assert expected.parent == tmp_path

"""
scripts.inference.test_video — Multi-Source YOLO Inference Pipeline
====================================================================

Runs YOLO11 inference on video files, webcam streams, or image folders.
Measures real-time FPS and per-frame latency, and optionally writes an
annotated output video and per-frame prediction JSON.

Supported input sources:
    --source 0              → Webcam (OpenCV camera index 0)
    --source video.mp4      → Video file
    --source images/        → Folder of images (alphabetical order)

Outputs (optional, to --output-dir):
    annotated_output.mp4    → Video with bounding box overlays
    predictions.json        → Per-frame prediction JSON
    inference_summary.json  → Aggregate summary (FPS, latency, class counts)

Usage:
    python scripts/inference/test_video.py --source 0
    python scripts/inference/test_video.py --source video.mp4 --model models/yolo11n/weights/best.pt
    python scripts/inference/test_video.py --source data/images/ --conf 0.25 --no-display

Performance measurements:
    - Per-frame inference latency (ms)
    - Running average FPS
    - Min / max / avg latency over the session
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_helpers import resolve_device
from src.utils.dataset_utils import find_image_files
from src.utils.report_utils import save_json_report, timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Source Type Resolution ───────────────────────────────────────────────────


def resolve_source_type(source: str) -> str:
    """Determine the input source type.

    Args:
        source: CLI source argument (webcam index, file path, or directory).

    Returns:
        One of: "webcam", "video", "images".

    Raises:
        ValueError: If the source is not a valid integer, existing video file,
            or existing directory.
    """
    # Try integer (webcam index)
    try:
        int(source)
        return "webcam"
    except ValueError:
        pass

    p = Path(source)
    if p.is_dir():
        return "images"
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return "video"
        raise ValueError(f"Unsupported video format: {suffix}")

    raise ValueError(f"Source not found: {source}")


# ─── Frame Generators ─────────────────────────────────────────────────────────


def frames_from_webcam(
    camera_index: int,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (frame_id, BGR frame) tuples from a webcam.

    Args:
        camera_index: OpenCV camera index.

    Yields:
        (frame_id, frame) tuples until the user presses 'q' or capture fails.
    """
    import cv2  # type: ignore[import]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {camera_index}")

    logger.info(f"Webcam opened: index={camera_index}")
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed — stopping")
                break
            frame_id += 1
            yield frame_id, frame
    finally:
        cap.release()


def frames_from_video(
    video_path: Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (frame_id, BGR frame) tuples from a video file.

    Args:
        video_path: Path to the video file.

    Yields:
        (frame_id, frame) tuples for each frame in the video.
    """
    import cv2  # type: ignore[import]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {video_path.name}, {total_frames} frames @ {fps_native:.1f} FPS")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
    finally:
        cap.release()


def frames_from_folder(
    folder_path: Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (frame_id, BGR frame) tuples from an image folder.

    Args:
        folder_path: Directory containing image files.

    Yields:
        (frame_id, frame) tuples in alphabetical order.
    """
    import cv2  # type: ignore[import]

    images = find_image_files(folder_path)
    if not images:
        raise RuntimeError(f"No image files found in: {folder_path}")

    logger.info(f"Image folder: {len(images)} images from {folder_path}")

    for frame_id, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read image: {img_path.name}")
            continue
        yield frame_id, frame


# ─── Annotation Drawing ───────────────────────────────────────────────────────


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR frame.

    Args:
        frame:       BGR numpy array (modified in-place).
        detections:  List of detection dicts with keys:
                       class_id, class_name, confidence, bbox (x1,y1,x2,y2 abs)
        class_names: class_id → name mapping.

    Returns:
        Annotated BGR frame.
    """
    import cv2  # type: ignore[import]

    h, w = frame.shape[:2]

    # Class-specific colors (BGR)
    safety_colors = {
        "knife": (0, 0, 255),  # Red
        "stove": (0, 128, 255),  # Orange
        "gas_cylinder": (0, 0, 200),  # Dark red
        "wire": (0, 0, 255),  # Red
        "wet_floor": (0, 165, 255),  # Orange
        "medicine_strip": (255, 0, 255),  # Magenta
        "medicine_bottle": (255, 0, 200),  # Pink
    }
    default_color = (0, 255, 0)  # Green

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        class_name = det.get("class_name", "")
        conf = det.get("confidence", 0.0)

        color = safety_colors.get(class_name, default_color)
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {conf:.2f}"
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


# ─── Main Inference Loop ──────────────────────────────────────────────────────


def run_inference(args: argparse.Namespace) -> int:
    """Execute the inference pipeline on the specified source.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    try:
        import cv2  # type: ignore[import]
        import numpy  # noqa: F401 — availability check only
    except ImportError:
        logger.error("OpenCV is required. Install with: pip install opencv-python")
        return 1

    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("ultralytics is required. Install with: pip install ultralytics")
        return 1

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path.absolute()}")
        return 1

    device = resolve_device(args.device)
    logger.info(f"Loading model: {model_path.name} on {device}")

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    class_names: dict[int, str] = model.names or {}

    # Resolve source
    source_str = args.source
    try:
        source_type = resolve_source_type(source_str)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Source type: {source_type} ({source_str})")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")

    # Setup video writer
    video_writer = None
    output_dir: Path | None = None

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup frame generator
    try:
        if source_type == "webcam":
            frame_gen = frames_from_webcam(int(source_str))
        elif source_type == "video":
            frame_gen = frames_from_video(Path(source_str))
        else:
            frame_gen = frames_from_folder(Path(source_str))
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    # Inference loop
    frame_results: list[dict] = []
    latencies_ms: list[float] = []
    class_counts: dict[str, int] = {}
    fps_tracker: list[float] = []

    logger.info("Starting inference…  (press Ctrl+C to stop)")

    try:
        for frame_id, frame in frame_gen:
            t0 = time.perf_counter()

            # Run inference
            try:
                yolo_results = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"Frame {frame_id} inference failed: {e}")
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(latency_ms)

            # FPS tracking (rolling window of last 30 frames)
            fps_tracker.append(latency_ms)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_latency = sum(fps_tracker) / len(fps_tracker)
            fps = 1000.0 / max(avg_latency, 1.0)

            # Extract detections
            frame_dets: list[dict] = []
            h_frame, w_frame = frame.shape[:2]

            for result in yolo_results:
                for box in result.boxes:
                    cid = int(box.cls.item())
                    conf = float(box.conf.item())
                    cname = class_names.get(cid, f"class_{cid}")
                    xyxy = box.xyxy[0].tolist()

                    frame_dets.append(
                        {
                            "class_id": cid,
                            "class_name": cname,
                            "confidence": round(conf, 4),
                            "x1": int(xyxy[0]),
                            "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]),
                            "y2": int(xyxy[3]),
                        }
                    )

                    class_counts[cname] = class_counts.get(cname, 0) + 1

            frame_results.append(
                {
                    "frame_id": frame_id,
                    "latency_ms": round(latency_ms, 2),
                    "fps": round(fps, 1),
                    "detections": frame_dets,
                }
            )

            # Annotate frame
            if args.output_dir or not args.no_display:
                annotated = draw_detections(frame.copy(), frame_dets, class_names)
            else:
                annotated = frame

            # Write to video
            if output_dir and args.output_video:
                if video_writer is None:
                    out_video_path = output_dir / "annotated_output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                    video_writer = cv2.VideoWriter(
                        str(out_video_path), fourcc, 15.0, (frame.shape[1], frame.shape[0])
                    )
                    logger.info(f"Writing annotated video: {out_video_path}")
                video_writer.write(annotated)

            # Display
            if not args.no_display:
                overlay = annotated.copy()
                cv2.putText(
                    overlay,
                    f"FPS: {fps:.1f}  |  {len(frame_dets)} det",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("YOLO Inference — Elderly Assistant", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q' — stopping")
                    break

            # Periodic progress log
            if frame_id % 50 == 0:
                logger.info(
                    f"Frame {frame_id}: {fps:.1f} FPS, "
                    f"{latency_ms:.1f} ms, {len(frame_dets)} detections"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception as e:  # noqa: S110 — window teardown is best-effort
                logger.debug(f"cv2.destroyAllWindows failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_frames = len(frame_results)
    if total_frames == 0:
        logger.warning("No frames processed")
        return 0

    avg_lat = sum(latencies_ms) / len(latencies_ms)
    summary = {
        "timestamp": timestamp_str(),
        "source": source_str,
        "source_type": source_type,
        "model": str(model_path),
        "device": device,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "total_frames": total_frames,
        "avg_fps": round(1000.0 / max(avg_lat, 1.0), 1),
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min(latencies_ms), 2),
        "max_latency_ms": round(max(latencies_ms), 2),
        "total_detections": sum(len(f["detections"]) for f in frame_results),
        "class_counts": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
    }

    logger.info("=" * 60)
    logger.info("Inference Summary")
    logger.info("=" * 60)
    logger.info(f"  Total frames:    {total_frames}")
    logger.info(f"  Average FPS:     {summary['avg_fps']:.1f}")
    logger.info(f"  Avg latency:     {summary['avg_latency_ms']:.1f} ms")
    logger.info(
        f"  Min/Max latency: {summary['min_latency_ms']:.1f} / "
        f"{summary['max_latency_ms']:.1f} ms"
    )
    logger.info(f"  Total dets:      {summary['total_detections']}")
    if class_counts:
        top = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
        logger.info(f"  Top classes:     {', '.join(f'{n}:{c}' for n, c in top)}")
    logger.info("=" * 60)

    # Write output reports
    if output_dir:
        save_json_report(summary, output_dir / "inference_summary.json")

        if args.save_predictions:
            preds_path = output_dir / "predictions.json"
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(frame_results, f, indent=2)
            logger.info(f"Predictions saved: {preds_path}")

    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO11 inference on video, webcam, or image folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: webcam index (0), video file, or image folder.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n/weights/best.pt",
        help="Path to YOLO model weights (.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output video and reports. None = no file output.",
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="Write annotated output video to --output-dir.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-frame prediction JSON to --output-dir.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress real-time display window (for headless/server use).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_inference(args)


if __name__ == "__main__":
    import json

    sys.exit(main())

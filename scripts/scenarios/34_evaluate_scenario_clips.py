"""
scripts.scenarios.34_evaluate_scenario_clips — Offline Scenario Clip Evaluator
===============================================================================

Runs the complete YOLO -> EvalContext -> ScenarioTrigger -> StateMachine pipeline
against each accepted clip in the dataset, comparing expected vs predicted outcomes.

Architecture:
    For each accepted clip:
        1. Load video frames via OpenCV
        2. Run YOLODetector.detect() per frame
        3. Feed detections to a FRESH EventMemory (isolation per clip)
        4. Build EvalContext from detections + memory + actual FPS
        5. Evaluate the SC-KIT-001 trigger via evaluate_trigger()
        6. Run a state machine replica (dwell, cooldown) to determine if an
           alert would fire
        7. Compare expected outcome (manifest) vs predicted outcome (engine)
        8. Classify any failures by root cause

The evaluator does NOT use ScenarioRuleEngine directly because all scenarios are
``status: draft``. Instead it loads the Scenario object, calls evaluate_trigger()
on the same trigger tree the runtime would use, and replicates the state machine.
This tests the identical predicate path without requiring clinical review status.

Usage:
    python scripts/scenarios/34_evaluate_scenario_clips.py
    python scripts/scenarios/34_evaluate_scenario_clips.py \\
        --model models/benchmarks/models/baseline_r0/weights/best.pt
    python scripts/scenarios/34_evaluate_scenario_clips.py --scenario SC-KIT-001
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.detector import YOLODetector
from src.pipeline.event_memory import EventMemory
from src.scenario_engine.clips import (
    ClipManifest,
    load_clip_manifests,
    load_clip_requirements,
)
from src.scenario_engine.compile import load_scenario_files
from src.scenario_engine.context import EvalContext
from src.scenario_engine.schema import Scenario
from src.scenario_engine.trigger import evaluate_trigger
from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path("models/benchmarks/models/baseline_r0/weights/best.pt")
DEFAULT_OUTPUT = Path("outputs/scenario_eval")


# ── State machine (mirrors runtime._EventState) ─────────────────────────────


@dataclass
class OfflineEventState:
    """Per-scenario state for offline evaluation. Same model as runtime._EventState."""

    pending_since_s: float | None = None
    active_since_s: float | None = None
    last_true_at_s: float | None = None
    last_alert_at_s: float | None = None
    repeats: int = 0
    alert_fired: bool = False
    first_alert_at_s: float | None = None

    def reset(self) -> None:
        self.pending_since_s = None
        self.active_since_s = None
        self.last_true_at_s = None
        self.repeats = 0


@dataclass
class FrameDecision:
    """Decision for a single frame."""

    frame_idx: int
    time_s: float
    condition_true: bool
    state: str  # idle, dwelling, active, alert_fired, condition_false
    detected_classes: list[str] = field(default_factory=list)
    detection_count: int = 0
    confidence_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ClipEvalResult:
    """Complete evaluation result for one clip."""

    clip_id: str
    scenario_id: str
    collection_label: str
    expected_fires: bool
    actual_fires: bool
    expected_alert: str  # "no-alert" or "fires"
    predicted_alert: str  # "no-alert" or "fires"
    first_alert_at_s: float | None
    total_frames: int
    duration_s: float
    fps: float
    result: str  # PASS, FALSE_POSITIVE, FALSE_NEGATIVE, INCONCLUSIVE
    error_type: str  # none, detector_failure, temporal_failure, etc.
    notes: str
    frames_with_stove: int = 0
    frames_with_person: int = 0
    frames_with_knife: int = 0
    frames_with_gas_cylinder: int = 0
    total_detections: int = 0
    unique_classes_detected: list[str] = field(default_factory=list)


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_clip(
    clip: ClipManifest,
    scenario: Scenario,
    detector: YOLODetector,
    clips_root: Path,
    output_dir: Path,
) -> ClipEvalResult:
    """Run the full evaluation pipeline on a single clip."""

    # Find the video file.
    video_path = None
    for ext in (".mp4", ".mov"):
        candidate = clips_root / "video" / f"{clip.clip_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        return ClipEvalResult(
            clip_id=clip.clip_id,
            scenario_id=clip.scenario_id,
            collection_label=clip.notes.split(".")[0] if clip.notes else "",
            expected_fires=clip.expected.fires,
            actual_fires=False,
            expected_alert="fires" if clip.expected.fires else "no-alert",
            predicted_alert="error",
            first_alert_at_s=None,
            total_frames=0,
            duration_s=clip.duration_s,
            fps=clip.fps,
            result="INVALID",
            error_type="data_quality_failure",
            notes=f"Video file not found for {clip.clip_id}",
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ClipEvalResult(
            clip_id=clip.clip_id,
            scenario_id=clip.scenario_id,
            collection_label=clip.notes.split(".")[0] if clip.notes else "",
            expected_fires=clip.expected.fires,
            actual_fires=False,
            expected_alert="fires" if clip.expected.fires else "no-alert",
            predicted_alert="error",
            first_alert_at_s=None,
            total_frames=0,
            duration_s=clip.duration_s,
            fps=clip.fps,
            result="INVALID",
            error_type="data_quality_failure",
            notes=f"Cannot open video: {video_path}",
        )

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or clip.fps
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fresh state per clip — CRITICAL for isolation.
    memory = EventMemory(window_size=max(150, int(actual_fps * 20)))
    state = OfflineEventState()

    decisions: list[FrameDecision] = []
    detection_log: list[dict[str, Any]] = []

    # Per-clip detection counters.
    frames_with_stove = 0
    frames_with_person = 0
    frames_with_knife = 0
    frames_with_gas_cylinder = 0
    total_detections = 0
    all_classes_seen: set[str] = set()

    frame_idx = 0
    alert_fired = False
    first_alert_time_s: float | None = None

    logger.info(
        f"  Evaluating {clip.clip_id}: {total_frame_count} frames, "
        f"{actual_fps:.1f} fps, {clip.duration_s:.1f}s"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_s = frame_idx / max(actual_fps, 1.0)

        # Run YOLO detection.
        detections = detector.detect(frame, frame_id=frame_idx)
        total_detections += len(detections)

        # Track class presence.
        detected_names = {d.class_name for d in detections}
        all_classes_seen.update(detected_names)
        if "stove" in detected_names:
            frames_with_stove += 1
        if "person" in detected_names:
            frames_with_person += 1
        if "knife" in detected_names:
            frames_with_knife += 1
        if "gas_cylinder" in detected_names:
            frames_with_gas_cylinder += 1

        # Update EventMemory.
        memory.update(detections)

        # Build EvalContext.
        # Auto-detect room from clip_id (e.g. h01_bathroom_s001_c001 -> bathroom)
        clip_parts = clip.clip_id.split("_")
        room = clip_parts[1] if len(clip_parts) >= 2 else "kitchen"

        context = EvalContext(
            detections=tuple(detections),
            memory=memory,
            fps=actual_fps,
            frame_id=frame_idx,
            room=room,
        )

        # Evaluate trigger.
        try:
            condition_true = evaluate_trigger(scenario.trigger, context)
        except Exception as exc:
            logger.warning(f"    Frame {frame_idx}: trigger error: {exc}")
            condition_true = False

        # Confidence gate (same as runtime._confident_enough).
        if condition_true:
            for class_name in scenario.required_objects:
                best = context.best(class_name)
                if best is not None and best.confidence < scenario.min_confidence:
                    condition_true = False
                    break

        # State machine (mirrors runtime._evaluate_one).
        frame_state = "condition_false"
        if condition_true:
            state.last_true_at_s = current_time_s
            if state.pending_since_s is None and state.active_since_s is None:
                state.pending_since_s = current_time_s
            if state.active_since_s is None:
                dwell = current_time_s - (state.pending_since_s or current_time_s)
                if dwell < scenario.min_dwell_seconds:
                    frame_state = "dwelling"
                else:
                    state.active_since_s = current_time_s
                    frame_state = "active"
            else:
                frame_state = "active"

            # Check if alert should fire.
            if state.active_since_s is not None and not alert_fired:
                # Check cooldown.
                can_fire = True
                if (
                    state.last_alert_at_s is not None
                    and current_time_s - state.last_alert_at_s < scenario.cooldown_seconds
                ):
                    can_fire = False
                if state.repeats >= scenario.max_repeats:
                    can_fire = False

                if can_fire:
                    alert_fired = True
                    first_alert_time_s = current_time_s
                    state.last_alert_at_s = current_time_s
                    state.repeats += 1
                    frame_state = "alert_fired"
                    logger.info(
                        f"    ALERT at frame {frame_idx} ({current_time_s:.1f}s): "
                        f"{scenario.scenario_id}"
                    )
        else:
            # Hysteresis: clear after clear_after_seconds of false.
            if state.active_since_s is not None and state.last_true_at_s is not None:
                if current_time_s - state.last_true_at_s >= scenario.clear_after_seconds:
                    state.reset()
            elif state.active_since_s is None:
                state.pending_since_s = None

        # Confidence summary for key classes.
        conf_summary: dict[str, float] = {}
        for d in detections:
            if d.class_name in ("stove", "person", "knife", "gas_cylinder"):
                if d.class_name not in conf_summary or d.confidence > conf_summary[d.class_name]:
                    conf_summary[d.class_name] = round(d.confidence, 3)

        decisions.append(FrameDecision(
            frame_idx=frame_idx,
            time_s=round(current_time_s, 3),
            condition_true=condition_true,
            state=frame_state,
            detected_classes=sorted(detected_names),
            detection_count=len(detections),
            confidence_summary=conf_summary,
        ))

        # Log every Nth frame for the detection log.
        if frame_idx % max(1, int(actual_fps)) == 0:  # ~1 per second
            detection_log.append({
                "frame": frame_idx,
                "time_s": round(current_time_s, 3),
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "bbox": [round(d.bbox.cx, 3), round(d.bbox.cy, 3),
                                 round(d.bbox.w, 3), round(d.bbox.h, 3)],
                    }
                    for d in detections
                ],
                "trigger": condition_true,
                "state": frame_state,
            })

        frame_idx += 1

    cap.release()

    # Determine result.
    expected_fires = clip.expected.fires

    # INCONCLUSIVE: model produced zero detections across all frames.
    # This means the model is blind/undertrained and the result is not
    # evidence for or against correctness. See Phase 1 diagnosis.
    if total_detections == 0:
        result = "INCONCLUSIVE"
        error_type = "model_blind"
    elif expected_fires and alert_fired:
        result = "PASS"
        error_type = "none"
    elif not expected_fires and not alert_fired:
        result = "PASS"
        error_type = "none"
    elif not expected_fires and alert_fired:
        result = "FALSE_POSITIVE"
        # Classify the failure.
        if frames_with_stove == 0:
            error_type = "detector_false_positive"
        else:
            error_type = "temporal_failure"
    elif expected_fires and not alert_fired:
        result = "FALSE_NEGATIVE"
        error_type = "temporal_failure"
    else:
        result = "INVALID"
        error_type = "unknown"

    # Build notes.
    notes_parts = [
        f"stove: {frames_with_stove}/{frame_idx} frames",
        f"person: {frames_with_person}/{frame_idx} frames",
    ]
    if frames_with_knife > 0:
        notes_parts.append(f"knife: {frames_with_knife}")
    if frames_with_gas_cylinder > 0:
        notes_parts.append(f"gas_cylinder: {frames_with_gas_cylinder}")

    eval_result = ClipEvalResult(
        clip_id=clip.clip_id,
        scenario_id=clip.scenario_id,
        collection_label=clip.notes.split(".")[0] if clip.notes else "",
        expected_fires=expected_fires,
        actual_fires=alert_fired,
        expected_alert="fires" if expected_fires else "no-alert",
        predicted_alert="fires" if alert_fired else "no-alert",
        first_alert_at_s=first_alert_time_s,
        total_frames=frame_idx,
        duration_s=clip.duration_s,
        fps=actual_fps,
        result=result,
        error_type=error_type,
        notes="; ".join(notes_parts),
        frames_with_stove=frames_with_stove,
        frames_with_person=frames_with_person,
        frames_with_knife=frames_with_knife,
        frames_with_gas_cylinder=frames_with_gas_cylinder,
        total_detections=total_detections,
        unique_classes_detected=sorted(all_classes_seen),
    )

    # Save per-clip detection log.
    clip_output = output_dir / clip.clip_id
    clip_output.mkdir(parents=True, exist_ok=True)
    (clip_output / "detection_log.json").write_text(
        json.dumps(detection_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return eval_result


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline scenario clip evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to YOLO model weights.")
    parser.add_argument("--scenario", default="all",
                        help="Scenario to evaluate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for results.")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (cpu, cuda, mps).")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clip requirements and manifests.
    try:
        requirements = load_clip_requirements()
    except Exception as exc:
        logger.error(f"Failed to load clip requirements: {exc}")
        return 1

    clips_root = requirements.clips_root
    try:
        all_clips = load_clip_manifests(clips_root)
    except Exception as exc:
        logger.error(f"Failed to load manifests: {exc}")
        return 1

    # Filter to accepted clips for the target scenario(s).
    if args.scenario == "all":
        target_clips = [c for c in all_clips if c.is_accepted]
    else:
        target_clips = [
            c for c in all_clips
            if c.is_accepted and c.scenario_id == args.scenario
        ]
    if not target_clips:
        logger.error(f"No accepted clips found for scenario {args.scenario}")
        return 1

    logger.info(f"Found {len(target_clips)} accepted clips for {args.scenario}")

    # Load the scenario definition (handles draft status).
    scenarios = load_scenario_files()
    # Build scenario lookup by ID
    scenario_map = {s.scenario_id: s for s in scenarios}

    if args.scenario != "all":
        scenario = scenario_map.get(args.scenario)
        if scenario is None:
            logger.error(f"Scenario {args.scenario} not found in configs/scenarios/")
            return 1

    if args.scenario != "all":
        logger.info(
            f"Loaded scenario {scenario.scenario_id} (status: {scenario.status.value}). "
            f"Trigger: {scenario.condition_text}. "
            f"Dwell: {scenario.min_dwell_seconds}s. "
            f"Required: {scenario.required_objects}"
        )
    else:
        scenario_ids = sorted(set(c.scenario_id for c in target_clips))
        logger.info(
            f"Multi-scenario evaluation: {len(scenario_ids)} scenarios, "
            f"{len(target_clips)} clips"
        )

    # Load YOLO model.
    logger.info(f"Loading YOLO model from {args.model}...")
    detector = YOLODetector(
        model_path=str(args.model),
        device=args.device,
    )
    detector.warmup(n=2)
    logger.info("YOLO model ready")

    # Evaluate each clip.
    results: list[ClipEvalResult] = []
    eval_start = time.time()

    for i, clip in enumerate(target_clips, 1):
        logger.info(f"[{i}/{len(target_clips)}] {clip.clip_id}")
        # Look up scenario for this clip
        clip_scenario = scenario_map.get(clip.scenario_id)
        if clip_scenario is None:
            logger.warning(f"  Skipping {clip.clip_id}: scenario {clip.scenario_id} not found")
            continue
        result = evaluate_clip(clip, clip_scenario, detector, clips_root, output_dir)
        results.append(result)
        logger.info(f"  Result: {result.result} | Detections: {result.total_detections}")

    eval_duration = time.time() - eval_start
    logger.info(f"Evaluation complete in {eval_duration:.1f}s")

    # Generate evaluation CSV.
    csv_path = output_dir / "evaluation.csv"
    csv_fields = [
        "clip_id", "scenario_id", "collection_label",
        "expected_fires", "actual_fires",
        "expected_alert", "predicted_alert",
        "first_alert_at_s", "total_frames", "duration_s", "fps",
        "result", "error_type",
        "frames_with_stove", "frames_with_person",
        "frames_with_knife", "frames_with_gas_cylinder",
        "total_detections", "unique_classes_detected", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            row = {
                "clip_id": r.clip_id,
                "scenario_id": r.scenario_id,
                "collection_label": r.collection_label,
                "expected_fires": r.expected_fires,
                "actual_fires": r.actual_fires,
                "expected_alert": r.expected_alert,
                "predicted_alert": r.predicted_alert,
                "first_alert_at_s": r.first_alert_at_s,
                "total_frames": r.total_frames,
                "duration_s": r.duration_s,
                "fps": r.fps,
                "result": r.result,
                "error_type": r.error_type,
                "frames_with_stove": r.frames_with_stove,
                "frames_with_person": r.frames_with_person,
                "frames_with_knife": r.frames_with_knife,
                "frames_with_gas_cylinder": r.frames_with_gas_cylinder,
                "total_detections": r.total_detections,
                "unique_classes_detected": ";".join(r.unique_classes_detected),
                "notes": r.notes,
            }
            writer.writerow(row)
    logger.info(f"Evaluation CSV: {csv_path}")

    # Generate metrics summary.
    total = len(results)
    passes = sum(1 for r in results if r.result == "PASS")
    false_positives = sum(1 for r in results if r.result == "FALSE_POSITIVE")
    false_negatives = sum(1 for r in results if r.result == "FALSE_NEGATIVE")
    invalid = sum(1 for r in results if r.result == "INVALID")
    inconclusive = sum(1 for r in results if r.result == "INCONCLUSIVE")

    expected_negatives = sum(1 for r in results if not r.expected_fires)
    true_negatives = sum(1 for r in results if not r.expected_fires and not r.actual_fires)
    expected_positives = sum(1 for r in results if r.expected_fires)
    true_positives = sum(1 for r in results if r.expected_fires and r.actual_fires)

    # YOLO detection stats.
    all_detected: set[str] = set()
    for r in results:
        all_detected.update(r.unique_classes_detected)

    metrics = {
        "evaluation_timestamp": timestamp_str(),
        "scenario_id": args.scenario,
        "model_path": str(args.model),
        "dataset_version": "v0.1",
        "total_clips": total,
        "total_frames_processed": sum(r.total_frames for r in results),
        "evaluation_duration_s": round(eval_duration, 1),
        "results": {
            "PASS": passes,
            "FALSE_POSITIVE": false_positives,
            "FALSE_NEGATIVE": false_negatives,
            "INCONCLUSIVE": inconclusive,
            "INVALID": invalid,
        },
        "confusion_matrix": {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "metrics": {
            "accuracy": round(passes / max(total, 1), 4),
            "false_positive_rate": round(
                false_positives / max(expected_negatives, 1), 4
            ),
            "specificity": round(
                true_negatives / max(expected_negatives, 1), 4
            ),
            "note": (
                "Recall and F1 are NOT reported because Dataset v0.1 contains "
                "zero true-positive clips (all clips are engine-negatives). "
                f"Expected positives: {expected_positives}, expected negatives: "
                f"{expected_negatives}. Sample size: n={total}."
            ),
        },
        "yolo_detection_summary": {
            "classes_detected_across_all_clips": sorted(all_detected),
            "per_clip": [
                {
                    "clip_id": r.clip_id,
                    "total_detections": r.total_detections,
                    "frames_with_stove": r.frames_with_stove,
                    "frames_with_person": r.frames_with_person,
                    "frames_with_knife": r.frames_with_knife,
                    "frames_with_gas_cylinder": r.frames_with_gas_cylinder,
                    "unique_classes": r.unique_classes_detected,
                }
                for r in results
            ],
        },
        "failure_analysis": [
            {
                "clip_id": r.clip_id,
                "result": r.result,
                "error_type": r.error_type,
                "notes": r.notes,
            }
            for r in results
            if r.result != "PASS"
        ],
        "true_positive_coverage": 0,
        "known_limitations": [
            "Dataset v0.1 contains zero true-positive clips",
            "All clips are engine-negatives (pre_dwell or absence)",
            f"Longest clip is {max(r.duration_s for r in results):.1f}s",
            "All clips are stock footage (not Indian home environment)",
            f"Sample size is n={total} -- insufficient for statistical inference",
        ],
    }

    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info(f"Metrics: {metrics_path}")

    # Summary to console.
    eval_label = args.scenario if args.scenario != "all" else "ALL SCENARIOS"
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS — {eval_label} — Dataset v0.1")
    print(f"{'='*70}")
    print(f"  Clips evaluated:     {total}")
    print(f"  PASS:                {passes}")
    print(f"  FALSE_POSITIVE:      {false_positives}")
    print(f"  FALSE_NEGATIVE:      {false_negatives}")
    print(f"  INCONCLUSIVE:        {inconclusive}")
    print(f"  INVALID:             {invalid}")
    print(f"{'─'*70}")
    print(f"  True Negatives:      {true_negatives}/{expected_negatives}")
    print(f"  False Positives:     {false_positives}/{expected_negatives}")
    print(f"  Specificity:         {true_negatives / max(expected_negatives, 1):.2%}")
    print(f"  FP Rate:             {false_positives / max(expected_negatives, 1):.2%}")
    print(f"{'─'*70}")
    print(f"  True Positives:      {true_positives} (0 expected — no genuine positives in v0.1)")
    print("  Recall:              NOT REPORTED (no true positives)")
    print("  F1:                  NOT REPORTED (no true positives)")
    if inconclusive > 0:
        print(f"{'─'*70}")
        print(f"  INCONCLUSIVE:        {inconclusive}/{total} ({inconclusive/max(total,1):.0%}) — model produced zero detections")
        print(f"  Diagnosis:           Model undertrained (3 epochs, mAP50=0.13). See Phase 1 report.")
    print(f"{'─'*70}")
    print(f"  Classes detected:    {sorted(all_detected)}")
    print(f"  Evaluation time:     {eval_duration:.1f}s")
    print(f"  Output:              {output_dir}")
    print(f"{'='*70}")

    if false_positives > 0:
        print("\n  FALSE POSITIVE CLIPS:")
        for r in results:
            if r.result == "FALSE_POSITIVE":
                print(f"    {r.clip_id}: alert at {r.first_alert_at_s:.1f}s, "
                      f"error: {r.error_type}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

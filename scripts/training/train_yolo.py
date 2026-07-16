"""
scripts.training.train_yolo — YOLO11 Training Automation
=========================================================

Full training pipeline for YOLO11 models. Reads all hyperparameters from
configs/training/yolo11n_config.yaml, with CLI overrides for any parameter.

Supports:
    - YOLO11n, YOLO11s (configurable --model-size)
    - Resume from last checkpoint (--resume)
    - Early stopping via patience from config
    - Automatic checkpointing (save_period)
    - TensorBoard support (auto-enabled by Ultralytics)
    - Optional W&B integration (wandb.enabled in training config)
    - Structured logging of training metadata
    - Optional missing-annotation mitigation (Phase-4): masked BCE cls loss
      driven by data/processed/completeness.json; preflight gates G1–G8 run
      before training; strictly opt-in (missing_annotation_mitigation.enabled
      or --mitigation on|off) — disabled keeps the stock path byte-for-byte

Training outputs (under models/yolo11n/):
    weights/best.pt           — Best checkpoint by val mAP50
    weights/last.pt           — Final checkpoint
    results/results.csv       — Per-epoch metrics
    results/metrics.json      — Final metrics summary (DVC tracked)

Usage:
    python scripts/training/train_yolo.py
    python scripts/training/train_yolo.py --config configs/training/yolo11n_config.yaml
    python scripts/training/train_yolo.py --epochs 50 --batch 8 --device cpu
    python scripts/training/train_yolo.py --resume
    python scripts/training/train_yolo.py --mitigation on

DVC integration:
    This script is invoked by the train_yolo11n DVC stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.mitigation_config import MITIGATION_SECTION, MitigationConfig
from src.utils.config_helpers import load_data_config, load_training_config, resolve_device
from src.utils.report_utils import timestamp_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── W&B Integration (optional) ──────────────────────────────────────────────


def _setup_wandb(wandb_cfg: dict, run_name: str) -> bool:
    """Initialize W&B if enabled and API key is available.

    Args:
        wandb_cfg: W&B config section from training YAML.
        run_name:  Run name for W&B dashboard.

    Returns:
        True if W&B was successfully initialized, False otherwise.
    """
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B integration disabled (wandb.enabled=false in config)")
        return False

    try:
        import wandb  # type: ignore[import]

        wandb.init(
            project=wandb_cfg.get("project", "elderly-assistant"),
            entity=wandb_cfg.get("entity") or None,
            name=run_name,
            tags=wandb_cfg.get("tags", []),
        )
        logger.info(f"W&B initialized: project='{wandb_cfg.get('project')}'")
        return True

    except ImportError:
        logger.warning("W&B not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}. Continuing without W&B.")
        return False


# ─── Metrics Extraction ───────────────────────────────────────────────────────


def extract_metrics(results) -> dict[str, float]:
    """Extract standard metrics from an Ultralytics Results object.

    Args:
        results: Ultralytics YOLO training results object.

    Returns:
        Dict with keys: precision, recall, mAP50, mAP50_95.
    """
    try:
        metrics = results.results_dict
        return {
            "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        }
    except Exception as e:
        logger.warning(f"Could not extract metrics from results object: {e}")
        return {}


def save_metrics_json(metrics: dict, output_dir: Path) -> Path:
    """Save training metrics to a JSON file for DVC tracking.

    Args:
        metrics: Metrics dict (from extract_metrics + training metadata).
        output_dir: Directory to write metrics.json.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved: {metrics_path}")
    return metrics_path


# ─── Training kwargs assembly ─────────────────────────────────────────────────


def build_train_kwargs(args: argparse.Namespace, train_cfg: dict) -> dict:
    """Assemble the kwargs passed to Ultralytics ``model.train()``.

    Extracted verbatim from the pre-Phase-4 inline assembly; a golden
    regression test (tests/unit/test_train_kwargs_regression.py) guards that
    the disabled-mitigation output stays byte-identical.

    Args:
        args:      Parsed CLI arguments (CLI overrides config values).
        train_cfg: Parsed training config dict.

    Returns:
        Flat kwargs dict for model.train(). Never contains a 'trainer' key —
        the mitigation-enabled path adds that separately in run_training().
    """
    model_section = train_cfg.get("model", {})
    training_section = train_cfg.get("training", {})
    output_section = train_cfg.get("output", {})

    epochs = args.epochs or training_section.get("epochs", 150)
    batch = args.batch or training_section.get("batch", 16)
    imgsz = args.imgsz or training_section.get("imgsz", 640)
    patience = args.patience or training_section.get("patience", 25)
    device_raw = args.device or model_section.get("device", "auto")
    device = resolve_device(device_raw)

    project = output_section.get("project", "models")
    name = args.name or output_section.get("name", "yolo11n")
    save_period = output_section.get("save_period", 10)

    train_kwargs: dict = {
        "data": str(args.data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": output_section.get("exist_ok", True),
        "save": output_section.get("save", True),
        "save_period": save_period,
        "val": output_section.get("val", True),
        "plots": output_section.get("plots", True),
        "verbose": output_section.get("verbose", True),
        "resume": args.resume,
        # Optimizer settings
        "optimizer": training_section.get("optimizer", "AdamW"),
        "lr0": training_section.get("lr0", 0.001),
        "lrf": training_section.get("lrf", 0.01),
        "momentum": training_section.get("momentum", 0.937),
        "weight_decay": training_section.get("weight_decay", 0.0005),
        "warmup_epochs": training_section.get("warmup_epochs", 5),
        "warmup_bias_lr": training_section.get("warmup_bias_lr", 0.1),
        "close_mosaic": training_section.get("close_mosaic", 15),
    }

    # Augmentation settings (from config)
    aug_section = train_cfg.get("augmentation", {})
    if aug_section:
        aug_kwargs = {
            "hsv_h": aug_section.get("hsv_h", 0.015),
            "hsv_s": aug_section.get("hsv_s", 0.5),
            "hsv_v": aug_section.get("hsv_v", 0.3),
            "degrees": aug_section.get("degrees", 5.0),
            "translate": aug_section.get("translate", 0.1),
            "scale": aug_section.get("scale", 0.4),
            "flipud": aug_section.get("flipud", 0.0),
            "fliplr": aug_section.get("fliplr", 0.5),
            "mosaic": aug_section.get("mosaic", 0.8),
            "mixup": aug_section.get("mixup", 0.1),
            "copy_paste": aug_section.get("copy_paste", 0.1),
        }
        train_kwargs.update(aug_kwargs)

    return train_kwargs


# ─── Training Pipeline ────────────────────────────────────────────────────────


def run_training(args: argparse.Namespace) -> int:
    """Execute the full YOLO training pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    # Load configurations
    try:
        train_cfg = load_training_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Training config not found: {e}")
        return 1

    try:
        load_data_config(args.data)  # validation only — Ultralytics reads the YAML itself
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data config error: {e}")
        return 1

    # Missing-annotation mitigation (Phase-4). Disabled config ⇒ the stock
    # training path below runs completely unchanged.
    try:
        mitigation = MitigationConfig.from_training_config(train_cfg)
        if args.mitigation is not None:
            mitigation = mitigation.with_overrides(enabled=(args.mitigation == "on"))
    except ValueError as e:
        logger.error(f"Invalid {MITIGATION_SECTION} config: {e}")
        return 1

    if mitigation.enabled:
        # Fail early, before any Ultralytics/W&B work.
        from src.training.preflight import run_preflight

        report = run_preflight(mitigation, data_yaml_path=Path(args.data), train_cfg=train_cfg)
        for gate in report.results:
            line = gate.format_line()
            if gate.status == "fail":
                logger.error(line)
            elif gate.status == "warn":
                logger.warning(line)
            else:
                logger.info(line)
        if report.verdict == "FAIL":
            logger.error("Mitigation preflight FAILED — training aborted before start.")
            return 1

    # Resolve settings (CLI args override config values)
    model_section = train_cfg.get("model", {})
    wandb_cfg = train_cfg.get("wandb", {})

    model_base = args.model or model_section.get("base", "yolo11n.pt")

    # Assemble training kwargs from config + CLI (extracted verbatim;
    # guarded by the golden regression test).
    train_kwargs = build_train_kwargs(args, train_cfg)
    project = train_kwargs["project"]
    name = train_kwargs["name"]

    run_name = f"{name}_{timestamp_str().replace(':', '-')}"

    logger.info("=" * 60)
    logger.info("YOLO Training Pipeline — Elderly Assistant System")
    logger.info("=" * 60)
    logger.info(f"Model:    {model_base}")
    logger.info(f"Data:     {args.data}")
    logger.info(f"Epochs:   {train_kwargs['epochs']}")
    logger.info(f"Batch:    {train_kwargs['batch']}")
    logger.info(f"Image sz: {train_kwargs['imgsz']}")
    logger.info(f"Device:   {train_kwargs['device']}")
    logger.info(f"Patience: {train_kwargs['patience']}")
    logger.info(f"Output:   {project}/{name}")
    logger.info(f"Resume:   {args.resume}")

    # W&B setup (optional)
    _setup_wandb(wandb_cfg, run_name)

    # Load YOLO model
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return 1

    if args.resume:
        # Resume from last checkpoint
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        if not last_ckpt.exists():
            logger.error(f"No checkpoint found to resume: {last_ckpt}")
            return 1
        logger.info(f"Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(model_base)

    mitigation_metrics: dict | None = None
    if mitigation.enabled:
        # Inject the masked-loss trainer. On resume this must be passed again
        # (the trainer class is never stored in checkpoints).
        from src.training.completeness_lookup import CompletenessLookup
        from src.training.trainer import build_masked_trainer

        try:
            lookup = CompletenessLookup.load(mitigation.completeness_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Completeness artifact unusable: {e}")
            return 1
        train_kwargs["trainer"] = build_masked_trainer(mitigation, lookup)
        mitigation_metrics = {
            "enabled": True,
            "completeness_path": mitigation.completeness_path.as_posix(),
            "taxonomy_fingerprint": lookup.fingerprint,
            "images_covered": len(lookup),
        }
        logger.info(
            f"Mitigation: ENABLED — masked BCE trainer injected "
            f"({len(lookup)} images, artifact {mitigation.completeness_path})"
        )

    # Run training
    logger.info("Starting training…")
    start_time = time.time()

    try:
        results = model.train(**train_kwargs)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 3600:.2f} hours")

    # Extract and save metrics
    metrics = extract_metrics(results)
    metrics.update(
        {
            "timestamp": timestamp_str(),
            "model_base": model_base,
            "epochs_trained": train_kwargs["epochs"],
            "batch_size": train_kwargs["batch"],
            "imgsz": train_kwargs["imgsz"],
            "device": train_kwargs["device"],
            "training_time_hours": round(elapsed / 3600, 3),
            "run_name": run_name,
        }
    )
    if mitigation_metrics is not None:
        # Only added when enabled — the disabled metrics.json stays unchanged.
        metrics["mitigation"] = mitigation_metrics

    results_dir = Path(project) / name / "results"
    save_metrics_json(metrics, results_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for key in ["precision", "recall", "mAP50", "mAP50_95"]:
        if key in metrics:
            logger.info(f"  {key:<15}: {metrics[key]:.4f}")

    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        logger.info(f"  Best weights : {best_weights.absolute()}")
    logger.info("=" * 60)

    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 model on the Elderly Assistant dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/yolo11n_config.yaml"),
        help="Training config YAML.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Dataset config YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model base (e.g., yolo11n.pt, yolo11s.pt).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override output run name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override training image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (auto, cpu, cuda, mps, 0, 1, ...).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt checkpoint.",
    )
    parser.add_argument(
        "--mitigation",
        choices=("on", "off"),
        default=None,
        help=(
            "Override missing_annotation_mitigation.enabled from the training "
            "config (Phase-4 masked BCE loss)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())

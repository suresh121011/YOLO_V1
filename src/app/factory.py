"""
src.app.factory — Pipeline Assembly
====================================

Builds a running pipeline with the scenario engine injected.

    from src.app import build_pipeline

    pipeline = build_pipeline(room="kitchen")

That is the whole public surface. Everything it does is wiring; the behaviour
lives in the components it assembles.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.config_loader import SystemConfig
from src.pipeline.orchestrator import ElderlyAssistantPipeline
from src.scenario_engine.runtime import ScenarioRuleEngine

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_FLAGS = Path("configs/feature_flags.yaml")
DEFAULT_THRESHOLDS = Path("configs/class_thresholds.yaml")


def build_rule_engine(
    scenario_dir: str | Path = DEFAULT_SCENARIO_DIR,
    fps: float = 15.0,
    room: str | None = None,
    flags_path: str | Path = DEFAULT_FLAGS,
    thresholds_path: str | Path = DEFAULT_THRESHOLDS,
) -> ScenarioRuleEngine:
    """Build the scenario engine with feature-flag gating applied.

    Args:
        scenario_dir:    Authored masters (``configs/scenarios/``).
        fps:             Fallback loop rate.
        room:            Deployment room for this camera. Room-scoped scenarios
                         never fire without it — a pipeline with no room set is
                         a pipeline where ``SC-KIT-001`` is inert.
        flags_path:      ``configs/feature_flags.yaml``.
        thresholds_path: ``configs/class_thresholds.yaml``.

    Raises:
        ScenarioRuntimeError: If no scenario is active. Expected before clinical
            review — every scenario ships as ``draft`` on purpose.
    """
    config = SystemConfig.load(str(flags_path), str(thresholds_path))
    return ScenarioRuleEngine(
        scenario_dir=scenario_dir,
        fps=fps,
        room=room,
        scenario_enabled=config.is_rule_enabled,
    )


def build_pipeline(
    room: str | None = None,
    scenario_dir: str | Path = DEFAULT_SCENARIO_DIR,
    model_path: str = "models/yolo11n/weights/best.pt",
    flags_path: str | Path = DEFAULT_FLAGS,
    thresholds_path: str | Path = DEFAULT_THRESHOLDS,
    target_fps: float | None = None,
    **pipeline_kwargs: object,
) -> ElderlyAssistantPipeline:
    """Assemble the full pipeline with the scenario engine injected.

    Args:
        room:            Deployment room for this camera. See
                         :func:`build_rule_engine`.
        scenario_dir:    Authored scenario masters.
        model_path:      YOLO weights.
        flags_path:      Feature flags.
        thresholds_path: Per-class confidence thresholds.
        target_fps:      Overrides the configured target.
        **pipeline_kwargs: Passed through to
            :class:`~src.pipeline.orchestrator.ElderlyAssistantPipeline`.

    Raises:
        ScenarioRuntimeError: If no scenario is active (see above).
    """
    config = SystemConfig.load(str(flags_path), str(thresholds_path))
    fps = target_fps if target_fps is not None else float(config.get_runtime("target_fps", 15))

    engine = build_rule_engine(
        scenario_dir=scenario_dir,
        fps=fps,
        room=room,
        flags_path=flags_path,
        thresholds_path=thresholds_path,
    )
    logger.info(
        f"Pipeline assembled with {len(engine.scenarios)} active scenario(s), "
        f"room={room or 'unset'}"
    )
    return ElderlyAssistantPipeline(
        model_path=model_path,
        scenario_dir=str(scenario_dir),
        flags_path=str(flags_path),
        thresholds_path=str(thresholds_path),
        target_fps=fps,
        room=room,
        rule_engine=engine,
        **pipeline_kwargs,  # type: ignore[arg-type]
    )

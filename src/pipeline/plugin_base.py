"""
src.pipeline.plugin_base — Extensibility Plugin Base Class
==========================================================

STAGE 1: Full implementation (30-line abstract base class, pure Python).

Provides the base class for all V2+ analysis plugins. No plugins are registered
in V1 — the for-loop over an empty list costs zero runtime overhead.

Planned V2+ plugins:
    - FallDetectionPlugin      — Pose estimation for fall events
    - MedicationOCRPlugin      — OCR-based medication label reading
    - ActivityRecognitionPlugin — Cooking / sleeping / exercise classification

Plugin registration (V2):
    pipeline = ElderlyAssistantPipeline(config_path="...")
    pipeline.register_plugin(FallDetectionPlugin())

See also:
    - src/pipeline/orchestrator.py for plugin lifecycle management
    - docs/03_engineering_appendix/future_modules.md for planned plugins
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from . import Alert, Detection
    from .event_memory import EventMemory

logger = logging.getLogger(__name__)


class AnalysisPlugin(ABC):
    """Abstract base class for all extensible analysis plugins.

    Implementing this interface guarantees the plugin can be safely registered
    with the pipeline orchestrator without code changes to the core pipeline.

    V1 Behavior:
        No plugins are instantiated or registered. The orchestrator maintains
        an empty list: self.plugins: list[AnalysisPlugin] = []
        An empty for-loop has zero runtime overhead.

    Lifecycle:
        1. __init__()      — plugin loads resources
        2. health_check()  — called before every analyze() call
        3. analyze()       — called once per frame (or N frames if plugin throttles)
        4. cleanup()       — called on pipeline shutdown
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (e.g., 'FallDetectionPlugin')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string (e.g., '1.0.0')."""
        ...

    @abstractmethod
    def analyze(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        memory: EventMemory,
    ) -> list[Alert]:
        """Analyze the current frame and return any triggered alerts.

        Args:
            frame:      BGR image array (H × W × 3, uint8)
            detections: YOLO detections from this frame (post confidence filter)
            memory:     Current event memory state (read-only access recommended)

        Returns:
            List of Alert objects. Return empty list if no alerts triggered.
            Alerts will be merged with Rule Engine alerts in the orchestrator.
        """
        ...

    def health_check(self) -> bool:
        """Return True if plugin is healthy and ready to analyze.

        Called by the orchestrator before each analyze() call. If this returns
        False, the plugin is skipped for this frame (logged as a warning).

        Default implementation always returns True.
        Override to check GPU memory, model availability, etc.
        """
        return True

    def cleanup(self) -> None:  # noqa: B027 — optional hook; no-op default is intentional
        """Release plugin resources on pipeline shutdown.

        Called by orchestrator.shutdown(). Override to close file handles,
        release GPU memory, stop background threads, etc.
        Default implementation is a no-op.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"

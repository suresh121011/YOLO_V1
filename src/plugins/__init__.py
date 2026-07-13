"""
src.plugins — Plugin System
============================

V1: No plugins registered. This package provides the namespace for V2+ plugins.

All plugin implementations must subclass:
    src.pipeline.plugin_base.AnalysisPlugin

Planned V2+ plugins:
    - FallDetectionPlugin      (fall_detection.py)
    - MedicationOCRPlugin      (medication_ocr.py)
    - ActivityRecognitionPlugin (activity_recognition.py)

Registration pattern (V2):
    from src.plugins.fall_detection import FallDetectionPlugin
    pipeline.register_plugin(FallDetectionPlugin())
"""

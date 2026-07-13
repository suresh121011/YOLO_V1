"""
src.config — Configuration Loading and Validation
=================================================

Provides centralized configuration management for the Elderly Assistant System.

All runtime parameters are controlled through YAML files in configs/.
No hardcoded paths or parameters exist anywhere in src/.

Configuration hierarchy:
    configs/feature_flags.yaml      → SystemConfig (components, classes, rules, runtime)
    configs/class_thresholds.yaml   → ClassThresholdConfig
    configs/risk_rules.yaml         → Loaded by RuleEngine directly
    configs/data.yaml               → Loaded by Ultralytics training scripts
    configs/tts_config.yaml         → Loaded by PiperTTS
"""

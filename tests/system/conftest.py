"""
System tests conftest.

System tests:
    - Test full end-to-end pipeline scenarios
    - Require YOLO model weights to be present
    - Require camera access OR pre-recorded test videos
    - Validate the 10 defined V1 field test scenarios

V1 System Test Scenarios (from docs/01_executive_implementation_plan/validation_strategy.md):
    1. Knife visible near person → HIGH alert within 2s
    2. Stove visible, person leaves → CRITICAL alert after 30s
    3. Wet floor detected near person → HIGH alert within 2s
    4. Wire on floor with person nearby → HIGH alert within 2s
    5. Medicine strip visible → INFO reminder within 5s
    6. Gas cylinder visible, no stove → INFO check within 5s
    7. No hazards in frame → No alerts (false positive check)
    8. Person present, multiple safe objects → No false alerts
    9. SmolVLM2 disabled mode → YOLO-only mode, all rules fire correctly
    10. TTS failure → Silent mode, no crash, alerts still logged

Mark slow tests with @pytest.mark.slow.
"""
from __future__ import annotations

import pytest

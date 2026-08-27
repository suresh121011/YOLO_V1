"""
Fixtures for pipeline-level tests.

The `active_scenarios` fixture these tests rely on lives one level up, in
`tests/unit/conftest.py`, because `tests/unit/test_model_landing_check.py`
needs it too and one promoted copy is better than two that can drift.
"""

from __future__ import annotations

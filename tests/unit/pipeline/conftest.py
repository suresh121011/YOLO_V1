"""
Fixtures for pipeline-level tests.

The `active_scenarios` fixture exists because of a real and deliberate
constraint: **every scenario in `configs/scenarios/` ships as `draft`**, and the
schema refuses `status: active` without `reviewed_by`/`reviewed_on` — a clinical
reviewer, not an engineer. So the runtime engine cannot be exercised against the
repository set as it stands, which is correct and must stay correct.

The fixture takes a *copy* into tmp_path and promotes it there. Nothing under
`configs/` is touched, so the gate cannot be eroded by a test needing it out of
the way, and `test_it_refuses_to_start_while_every_scenario_is_draft` still
asserts the real behaviour against the real directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

_SCENARIO_DIR = Path("configs/scenarios")


@pytest.fixture
def active_scenarios(tmp_path: Path) -> Path:
    """A copy of the authored scenarios, promoted to `active` for testing only.

    Rejected scenarios stay rejected — promoting those would be inventing
    capability the taxonomy cannot support.
    """
    destination = tmp_path / "scenarios"
    shutil.copytree(_SCENARIO_DIR, destination)

    for path in sorted(destination.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("status") != "draft":
            continue  # leave `rejected` rows exactly as authored
        raw["status"] = "active"
        raw["reviewed_by"] = "test-fixture-not-a-clinician"
        raw["reviewed_on"] = "2026-08-11"
        path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
    return destination

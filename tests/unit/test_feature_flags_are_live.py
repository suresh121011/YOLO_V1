"""
Every feature flag either does something or says it doesn't.

`configs/feature_flags.yaml` opens with "All runtime behavior is controlled
here. No code changes required." When Phase-6 M9 checked, eight of the ten
component toggles were read by nothing at all — an operator could set
`tts_output: false` and the device would keep speaking.

Phase-6 M1 fixed the *loading* of that file (the orchestrator was reading a
`feature_flags:` root key that does not exist, so every flag was permanently
dead). Nobody then checked that each individual flag was *consumed*, which is
how a control plane comes to be 20% real while reading as 100%.

This test closes that gap in the only way that stays closed: a flag must be
referenced somewhere in `src/`, or be listed here as NOT WIRED. Adding a flag
without wiring it fails. Wiring one and forgetting to unlabel it also fails, so
the labels cannot rot into pessimism either.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import yaml

_FLAGS_PATH = Path("configs/feature_flags.yaml")
_SRC = Path("src")

#: Flags deliberately not wired, each with the reason recorded in the YAML
#: itself. Shrinking this set is progress; growing it needs a reason in review.
UNWIRED: dict[str, str] = {
    "yolo_detection": "structural — without the detector there is no pipeline",
    "active_learning_logging": "uncertain-band capture is V2",
    "thermal_monitoring": "Phase 7 field testing",
    "rule_hot_reload": "reload_rules() works; nothing calls it on a timer",
    "caregiver_sync": "governs remote delivery; the local sink is always on",
    "hindi_tts": "superseded by runtime.tts_language plus per-scenario messages",
    "debug_overlay": "no render surface in the headless pipeline",
    "tts_volume": "the OS mixer is the volume control",
    "camera_index": "the camera loop belongs to the caller, not to the pipeline",
    "log_retention_days": "log rotation is the deployment's job",
    "active_learning_conf_min": "uncertain-band capture is V2",
    "active_learning_conf_max": "uncertain-band capture is V2",
}


def _flag_names() -> list[str]:
    payload = yaml.safe_load(_FLAGS_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for section in ("components", "runtime"):
        names.extend(str(key) for key in (payload.get(section) or {}))
    return names


def _without_the_defaults_block(text: str) -> str:
    """Drop ``_DEFAULT_RUNTIME``: it *declares* every key and consumes none.

    Without this the check passes vacuously — the fallback dict alone mentions
    every runtime key by name, so a flag nothing honours would still look wired.
    """
    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith("_DEFAULT_RUNTIME"):
            skipping = True
        elif skipping and line.startswith("}"):
            skipping = False
            continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


@lru_cache(maxsize=1)
def _source_text() -> str:
    parts = []
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == "config_loader.py":
            text = _without_the_defaults_block(text)
        parts.append(text)
    return "\n".join(parts)


@pytest.mark.unit
@pytest.mark.parametrize("flag", _flag_names())
def test_flag_is_wired_or_declared_unwired(flag: str) -> None:
    referenced = f'"{flag}"' in _source_text() or f"'{flag}'" in _source_text()

    if flag in UNWIRED:
        assert not referenced, (
            f"{flag!r} is listed as NOT WIRED but src/ references it. If it is now "
            f"honoured, remove it from UNWIRED here and from the label in "
            f"{_FLAGS_PATH}."
        )
        return

    assert referenced, (
        f"{flag!r} is in {_FLAGS_PATH} but no code in src/ reads it. Wire it, or "
        f"label it NOT WIRED in the YAML and add it to UNWIRED here with the reason. "
        f"A flag that silently does nothing is worse than a missing feature: the "
        f"operator believes they have configured something."
    )


@pytest.mark.unit
def test_the_unwired_list_has_no_ghosts() -> None:
    """Every excuse must correspond to a flag that still exists."""
    ghosts = sorted(set(UNWIRED) - set(_flag_names()))
    assert not ghosts, f"UNWIRED names flags that are no longer in the config: {ghosts}"


@pytest.mark.unit
@pytest.mark.parametrize("flag", sorted(UNWIRED))
def test_an_unwired_flag_says_so_where_an_operator_reads_it(flag: str) -> None:
    """The label has to be in the YAML, not only in this test.

    Nobody configuring a device reads the test suite. A flag that is inert in
    practice and honest only in `UNWIRED` is still a flag that lies to the person
    setting it.
    """
    for line in _FLAGS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{flag}:"):
            assert "NOT WIRED" in line, (
                f"{flag!r} is inert but its line in {_FLAGS_PATH} does not say so. "
                f"Add a `# NOT WIRED — <reason>` comment."
            )
            return
    pytest.fail(f"{flag!r} was not found as a key in {_FLAGS_PATH}")


@pytest.mark.unit
def test_the_per_class_and_per_rule_sections_are_not_checked_here() -> None:
    """Documentation, in test form.

    `classes:` is enforced in the detector by name and `rules:` in the scenario
    engine by scenario id, both dynamically. Neither appears as a literal in
    src/, so scanning them the same way would produce false alarms.
    """
    payload = yaml.safe_load(_FLAGS_PATH.read_text(encoding="utf-8"))
    assert payload["classes"]["passport"] is False
    assert payload["rules"], "the per-rule kill switches must still exist"

"""
Layering guard for src.scenario_engine.

ADR-P6-04 makes the scenario engine a leaf package injected into the
orchestrator, rather than something the pipeline reaches into. That only holds
if the dependency arrow stays one-way, so this test enforces it statically.

It mirrors the existing `src/dataset` must-not-import-`src/training` rule.
Statically scanning imports rather than checking `sys.modules` means the guard
also catches a forbidden import that happens to be lazy or conditional.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SCENARIO_ENGINE = Path("src/scenario_engine")

#: src/scenario_engine must never depend on these. Heavy runtime components and
#: the dataset/training stacks are all the wrong direction.
_FORBIDDEN_FOR_SCENARIO_ENGINE = (
    "src.pipeline.orchestrator",
    "src.pipeline.detector",
    "src.pipeline.scene_analyzer",
    "src.pipeline.tts_engine",
    "src.pipeline.confidence_fusion",
    "src.dataset",
    "src.training",
)


def _imported_modules(path: Path) -> set[str]:
    """Every module name a file imports, with relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = path.parent.as_posix().replace("/", ".").split(".")
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                module = ".".join([*base, node.module]) if node.module else ".".join(base)
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.unit
@pytest.mark.parametrize("path", _python_files(_SCENARIO_ENGINE), ids=lambda p: p.name)
def test_scenario_engine_imports_stay_within_its_layer(path: Path) -> None:
    imported = _imported_modules(path)
    violations = sorted(
        module
        for module in imported
        for forbidden in _FORBIDDEN_FOR_SCENARIO_ENGINE
        if module == forbidden or module.startswith(f"{forbidden}.")
    )
    assert not violations, (
        f"{path} imports {violations}. src/scenario_engine is a leaf package: it may "
        f"import src.pipeline data contracts, src.pipeline.event_memory and src.utils, "
        f"and nothing else from src. See ADR-P6-04."
    )


@pytest.mark.unit
def test_pipeline_never_imports_the_scenario_engine() -> None:
    """The orchestrator injects the engine; the pipeline must not reach for it.

    A cycle here happens to work today only because src/pipeline/__init__.py
    imports no submodules, and would detonate the moment one is re-exported.
    """
    offenders = [
        path
        for path in _python_files(Path("src/pipeline"))
        if any(m.startswith("src.scenario_engine") for m in _imported_modules(path))
    ]
    assert not offenders, (
        f"{offenders} import src.scenario_engine, creating a package cycle "
        f"(src.pipeline -> src.scenario_engine -> src.pipeline). The orchestrator "
        f"chooses the engine via constructor injection instead. See ADR-P6-04."
    )


@pytest.mark.unit
def test_guard_detects_a_planted_violation(tmp_path: Path) -> None:
    """The guard must actually catch something — including a lazy import."""
    planted = tmp_path / "bad.py"
    planted.write_text(
        "def f():\n    from src.dataset.merge import thing\n    return thing\n",
        encoding="utf-8",
    )
    assert "src.dataset.merge" in _imported_modules(planted)

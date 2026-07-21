"""
src.dataset.annotation.registry — Auto-Annotator Registry
=========================================================

Decorator-based backend registry, third instance of the house pattern
(src/dataset/completeness_policies.py, src/dataset/splitting/registry.py).
Adding a backend: subclass :class:`AutoAnnotator`, decorate with
``@register_annotator("my_backend")``, add a ``my_backend:`` section under
``auto_annotation.backends`` in configs/annotation.yaml — the generator core
needs no changes (ADR-P5-01/02).
"""

from __future__ import annotations

from collections.abc import Callable

from src.dataset.annotation.base import AnnotationError, AutoAnnotator

_ANNOTATORS: dict[str, type[AutoAnnotator]] = {}


def register_annotator(name: str) -> Callable[[type[AutoAnnotator]], type[AutoAnnotator]]:
    """Class decorator registering a backend under an annotator name.

    Args:
        name: The ``auto_annotation.backends`` config key for this backend.

    Raises:
        ValueError: If the name is already registered.
    """

    def _register(cls: type[AutoAnnotator]) -> type[AutoAnnotator]:
        if name in _ANNOTATORS:
            raise ValueError(
                f"Annotator '{name}' already registered by {_ANNOTATORS[name].__name__}"
            )
        cls.name = name
        _ANNOTATORS[name] = cls
        return cls

    return _register


def available_annotators() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_ANNOTATORS)


def get_annotator(name: str) -> AutoAnnotator:
    """Instantiate the backend registered under a name.

    Args:
        name: Backend name from ``auto_annotation.backends``.

    Raises:
        AnnotationError: If the name is unknown, listing registered backends.
    """
    if name not in _ANNOTATORS:
        raise AnnotationError(
            f"Unknown annotator '{name}'. Registered backends: {available_annotators()}. "
            f"Register one via @register_annotator or fix configs/annotation.yaml."
        )
    return _ANNOTATORS[name]()

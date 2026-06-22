"""Backend registry — selection is data-driven by name.

``get_recognizer(name)`` constructs the backend, which may raise
``BadInputError`` (via the constructor) for backends whose optional
dependency / API key isn't available — callers should let that propagate, it
is already an actionable error.
"""

from __future__ import annotations

from typing import Callable

from ..errors import BadInputError
from .ace_step import AceStepRecognizer
from .clap import ClapRecognizer
from .heuristic import HeuristicRecognizer
from .types import Recognizer

# name -> zero-arg constructor. Constructors for opt-in backends raise
# BadInputError if their dependency/key isn't available (lazy-imported inside
# __init__, never at module load time).
_BACKENDS: dict[str, Callable[[], Recognizer]] = {
    "heuristic": HeuristicRecognizer,
    "clap": ClapRecognizer,
    "ace-step": AceStepRecognizer,
}


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_recognizer(name: str) -> Recognizer:
    try:
        constructor = _BACKENDS[name]
    except KeyError:
        raise BadInputError(
            f"unknown recognizer backend '{name}' (available: {', '.join(list_backends())})"
        )
    return constructor()

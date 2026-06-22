"""Pluggable content-based recognition backends.

Given a batch of files (already probed for duration/kind via a ``FileProbe``), a
backend listens to the audio and returns a coarse ``category`` plus a
multi-valued ``instruments`` list. Recognizers are batch-first because model
backends (CLAP, ACE-Step) amortize load over a folder.

Three backends, selected by name (see :func:`list_backends`):

  * ``heuristic``  — local, zero extra deps, spectral-feature rules.
  * ``clap``       — local zero-shot audio-text embedding (extra: ``timbre[clap]``).
  * ``ace-step``   — local ACE-Step captioner (extra: ``timbre[ace-step]``).

Most callers want the higher-level :func:`timbre.classify` instead, which fuses
a recognizer with the filename pass. See ``recognize.types`` for the
``Recognizer``/``FileProbe``/``Recognition`` shapes.
"""

from .registry import get_recognizer, list_backends
from .types import FileProbe, Recognition, Recognizer

__all__ = ["FileProbe", "Recognition", "Recognizer", "get_recognizer", "list_backends"]

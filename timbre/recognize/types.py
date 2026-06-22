"""Shared shapes for the recognizer seam: ``FileProbe`` in, ``Recognition`` out.

Two-axis taxonomy on top of the existing ``kind`` (one-shot / loop / unknown):

  * ``category``    — single coarse role, vocabulary depends on ``kind``
                       (see ``ONESHOT_CATEGORIES`` / ``LOOP_CATEGORIES``).
  * ``instruments`` — 0..N instrument tags from ``INSTRUMENT_VOCAB``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

# Coarse category vocabulary, split by `kind` — a recognizer picks from the
# set matching the file's already-known `kind` (one-shot vs loop). `unknown`
# kind files are treated as one-shots for category purposes (the safer,
# smaller vocabulary), since a recognizer's `category` is only used on the
# fallback path anyway (see `library._fuse_category`).
ONESHOT_CATEGORIES: tuple[str, ...] = (
    "kick", "snare", "clap", "hat", "tom", "crash", "ride", "rim", "perc",
    "bass", "808", "stab", "vocal", "fx", "melody",
)
LOOP_CATEGORIES: tuple[str, ...] = (
    "drum", "perc", "bass", "melodic", "chord", "vocal", "fx", "full",
)

# Multi-valued instrument vocabulary — shared across one-shots and loops.
# Drum-kit and hand-percussion labels come first so percussion-heavy packs get
# specific instrument tags (kick/snare/conga/shaker/...) instead of collapsing
# to a bare "drums"; melodic instruments follow for non-percussion libraries.
INSTRUMENT_VOCAB: tuple[str, ...] = (
    # drum kit
    "kick", "snare", "clap", "snap", "rimshot", "hihat", "open hat",
    "crash", "ride", "cymbal", "tom",
    # hand / world percussion
    "conga", "bongo", "shaker", "tambourine", "cowbell", "woodblock",
    "clave", "triangle", "djembe", "timbale", "agogo", "cabasa",
    # generic fallbacks
    "drums", "percussion",
    # melodic / other
    "bass", "808", "piano", "keys", "guitar", "strings", "brass", "synth",
    "vocal", "fx",
)


@dataclass(frozen=True)
class FileProbe:
    """Already-computed per-file facts handed to a recognizer — backends must
    not re-probe duration/kind themselves."""

    path: Path
    filename: str
    duration: float | None
    kind: str  # "one-shot" | "loop" | "unknown"


@dataclass(frozen=True)
class Recognition:
    """One backend's verdict for a single file."""

    category: str
    instruments: list[str]
    source: str  # "heuristic" | "clap" | "ace-step"
    confidence: float  # 0..1
    caption: str | None = None  # free-text description, when a backend produces one (e.g. ace-step)


# Optional streaming sink: a backend may call this as each file's verdict is
# ready (in any order), letting the caller checkpoint progress for a long scan
# so a crash can resume instead of restarting. Purely advisory — the full list
# is still returned from ``recognize``.
ResultSink = Callable[[FileProbe, "Recognition | None"], None]


class Recognizer(Protocol):
    """A pluggable content-based recognition backend.

    Batch-first: ``recognize`` takes the whole folder's probes at once so
    model backends can amortize load/round-trips. Returning ``None`` for an
    item means "defer to the filename guess" (e.g. the backend couldn't form
    an opinion for that file).

    A backend may accept an optional ``on_result`` sink and invoke it per file
    as verdicts land (used for incremental checkpointing); backends that don't
    support streaming simply omit the parameter.
    """

    name: str

    def recognize(
        self, items: list[FileProbe], on_result: "ResultSink | None" = None
    ) -> list[Recognition | None]: ...

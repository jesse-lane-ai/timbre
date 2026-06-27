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

# The three real kinds (plus "unknown"). A `recording` is a long-form capture —
# a field recording, jam, take, or voice memo — that is neither a seamlessly
# loopable phrase nor a single hit.
KINDS: tuple[str, ...] = ("one-shot", "loop", "recording", "unknown")

# Coarse category vocabulary, split by `kind` — a recognizer picks from the set
# matching the file's already-known `kind`. `unknown` kind files are treated as
# one-shots for category purposes (the safer, smaller vocabulary). Use
# `categories_for_kind()` rather than indexing these directly.
ONESHOT_CATEGORIES: tuple[str, ...] = (
    # drums
    "kick", "snare", "clap", "snap", "hat", "tom", "crash", "ride", "rim", "perc",
    # bass
    "bass", "808", "sub", "reese",
    # melodic / synth
    "stab", "melody", "lead", "pad", "pluck", "arp", "chord", "keys", "piano",
    # acoustic
    "guitar", "strings", "brass",
    # vocal
    "vocal",
    # sound design / fx
    "fx", "riser", "sweep", "impact", "drone", "texture", "ambience", "foley", "noise",
)
LOOP_CATEGORIES: tuple[str, ...] = (
    # drums / bass
    "drum", "perc", "bass", "sub", "808",
    # melodic / synth
    "melodic", "chord", "lead", "pad", "arp", "synth", "keys", "piano",
    # acoustic
    "guitar", "strings", "brass",
    # vocal
    "vocal",
    # sound design / fx
    "fx", "riser", "texture", "ambience", "foley",
    # whole-mix
    "full",
)
# Long-form captures: coarse roles for what the whole recording mostly is.
RECORDING_CATEGORIES: tuple[str, ...] = (
    "full", "vocal", "instrument", "drum", "melodic", "chord",
    "ambience", "field", "foley", "fx",
)


def categories_for_kind(kind: str) -> tuple[str, ...]:
    """Return the category vocabulary a recognizer should choose from for a
    file of the given ``kind``. Unknown kinds fall back to the one-shot vocab."""
    if kind == "loop":
        return LOOP_CATEGORIES
    if kind == "recording":
        return RECORDING_CATEGORIES
    return ONESHOT_CATEGORIES


# Multi-valued instrument vocabulary — shared across all kinds.
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
    # bass
    "bass", "808", "sub", "reese",
    # melodic / synth
    "lead", "pad", "pluck", "arp", "synth", "keys", "piano", "organ", "bell",
    # acoustic
    "guitar", "strings", "violin", "cello", "brass", "trumpet", "sax", "flute", "choir",
    # plucked world / folk strings
    "koto", "guzheng", "zither", "sitar", "harp", "banjo", "ukulele",
    # mallets / pitched percussion
    "marimba", "kalimba", "vibraphone", "xylophone", "glockenspiel",
    # other keyboard
    "harpsichord", "accordion",
    # other
    "vocal",
    # sound design / fx
    "fx", "sound design", "noise",
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

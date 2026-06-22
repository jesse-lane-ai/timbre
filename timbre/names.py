"""Filename + folder-path classifier for sample files.

``classify_from_names(file_path)`` derives kind/category/key/scale/bpm/instruments
from the file's name AND every parent-folder component — no audio loading required.
It is the cheap first pass in the library indexing pipeline; audio content
(AceStep, heuristic) fills in what names can't determine.

Returns a dict with keys:
  kind        — "loop" | "one-shot" | None
  category    — coarse role string or None
  key         — root note string ("C", "F#", ...) or None
  scale       — "major" | "minor" | None
  bpm         — float or None
  instruments — list[str] (may be empty)
  confidence  — float 0..1 (mean of per-field confidences that fired)
  source      — always "name"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Vocabularies (mirrors library._CATEGORY_KEYWORDS + recognize.types)
# ---------------------------------------------------------------------------

# (keywords, category) — first match wins; checked against lowercased tokens.
_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    # drums — specific before generic
    (("kick", "bd", "bassdrum", "bass_drum", "bass-drum", "kik", "808"), "kick"),
    (("rim", "rimshot", "sidestick", "side_stick", "side-stick"), "rim"),
    (("snare", "sd", "snr"), "snare"),
    (("clap", "clp"), "clap"),
    (("closedhat", "closed_hat", "closed-hat", "hatclosed", "hat_closed", "chh"), "hat"),
    (("openhat", "open_hat", "open-hat", "hatopen", "hat_open", "ohh"), "hat"),
    (("hat", "hh", "hihat", "hi-hat", "hi_hat"), "hat"),
    (("tom",), "tom"),
    (("crash",), "crash"),
    (("ride",), "ride"),
    (("perc", "shaker", "tamb", "cowbell", "conga", "bongo", "clave"), "perc"),
    # drum category rollups
    (("drum", "drums"), "drums"),
    # tonal
    (("bass",), "bass"),
    (("lead",), "lead"),
    (("chord", "chords"), "chord"),
    (("synth",), "synth"),
    (("pad",), "pad"),
    (("melody", "melodic", "mel"), "melody"),
    (("keys", "piano", "key"), "keys"),
    (("vocal", "vox", "voice", "acap", "sing"), "vocal"),
    (("fx", "riser", "impact", "transition", "sweep", "noise"), "fx"),
    (("arp",), "arp"),
    (("stab",), "stab"),
]

# Sub-category labels
_DRUM_CATEGORIES = {"kick", "snare", "clap", "hat", "tom", "crash", "ride", "rim", "perc", "drums"}
_MELODIC_CATEGORIES = {"lead", "synth", "pad", "chord", "melody", "keys", "arp"}

# ---------------------------------------------------------------------------
# BPM regexes
# ---------------------------------------------------------------------------

# Matches: "124bpm", "bpm124", "124 bpm", "bpm 124", "124_bpm", "bpm_124"
_BPM_RE = re.compile(
    r"(?:(\d{2,3}(?:\.\d+)?)\s*[-_]?\s*bpm|bpm\s*[-_]?\s*(\d{2,3}(?:\.\d+)?))",
    re.I,
)
# A lone 2-3-digit number surrounded by non-digits (e.g. "_128_", " 90 ")
_BPM_BARE_RE = re.compile(r"(?<![0-9])(\d{2,3})(?![0-9])")

# ---------------------------------------------------------------------------
# Key / scale regexes
# ---------------------------------------------------------------------------

_NOTE = r"([A-G][b#]?)"
_SCALE_KEYWORD = r"(?:[-_ ]?(major|maj\d*|minor|min(?:or)?|m(?=\b)))"

_KEY_RE = re.compile(
    rf"(?<![A-Za-z]){_NOTE}{_SCALE_KEYWORD}?(?![A-Za-z0-9])",
)

_ENHARMONIC: dict[str, str] = {
    "ab": "G#", "bb": "A#", "cb": "B", "db": "C#",
    "eb": "D#", "fb": "E", "gb": "F#",
}

# ---------------------------------------------------------------------------
# Kind keywords
# ---------------------------------------------------------------------------

_LOOP_RE = re.compile(r"loop", re.I)
# "oneshots" (plural folder name) matches as well as "oneshot"
_ONESHOT_RE = re.compile(r"(?<![a-z])(?:one[\s_-]?shots?|oneshots?|hit|stab)(?![a-z])", re.I)


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------

def _parse_bpm(text: str) -> float | None:
    """Extract BPM from a text fragment (filename stem or folder name)."""
    m = _BPM_RE.search(text)
    if m:
        val = float(m.group(1) or m.group(2))
        if 40 <= val <= 220:
            return val
    # Bare number fallback — only for short fragment-like strings
    m = _BPM_BARE_RE.search(text)
    if m:
        val = float(m.group(1))
        if 60 <= val <= 200:
            return val
    return None


def _parse_key_scale(text: str) -> tuple[str | None, str | None, float]:
    """Extract (root, scale, confidence) from a text fragment."""
    for m in _KEY_RE.finditer(text):
        note_raw = m.group(1)
        scale_raw = (m.group(2) or "").lower().rstrip("0123456789")

        note = note_raw[0].upper() + (note_raw[1:] if len(note_raw) > 1 else "")
        norm_key = _ENHARMONIC.get(note.lower())
        if norm_key:
            note = norm_key

        if scale_raw in ("minor", "min", "m"):
            scale = "minor"
            conf = 0.9
        elif scale_raw in ("major", "maj") or (scale_raw.startswith("maj") and len(scale_raw) > 3):
            scale = "major"
            conf = 0.9
        elif scale_raw == "":
            scale = None
            conf = 0.6
        else:
            continue

        return note, scale, conf

    return None, None, 0.0


def _parse_kind(text: str) -> str | None:
    tl = text.lower()
    if _LOOP_RE.search(tl):
        return "loop"
    if _ONESHOT_RE.search(tl):
        return "one-shot"
    return None


def _parse_category(text: str) -> str | None:
    tl = text.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(kw in tl for kw in keywords):
            return category
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_from_names(file_path: str | Path) -> dict[str, Any]:
    """Derive sample metadata from the file path (filename + parent folders).

    Parses every component of the path from root down to filename stem.
    More-specific components (closer to the file) win ties.

    Returns::

        {
            "kind":        "loop" | "one-shot" | None,
            "category":    str | None,
            "key":         str | None,       # e.g. "C", "F#"
            "scale":       "major" | "minor" | None,
            "bpm":         float | None,
            "instruments": list[str],
            "confidence":  float,            # 0..1
            "source":      "name",
        }
    """
    p = Path(file_path)
    stem = p.stem
    fragments: list[str] = [part for part in p.parts[:-1]]  # parent folders
    fragments.append(stem)                                    # filename stem last

    # --- kind ---
    kind: str | None = None
    kind_conf = 0.0
    for frag in reversed(fragments):
        k = _parse_kind(frag)
        if k is not None:
            kind = k
            kind_conf = 0.9
            break

    # --- category ---
    category: str | None = None
    cat_conf = 0.0
    for frag in reversed(fragments):
        c = _parse_category(frag)
        if c is not None:
            category = c
            cat_conf = 0.85
            break

    # --- BPM ---
    bpm: float | None = None
    bpm_conf = 0.0
    for frag in reversed(fragments):
        b = _parse_bpm(frag)
        if b is not None:
            bpm = b
            bpm_conf = 0.9
            break

    # --- Key / Scale ---
    key: str | None = None
    scale: str | None = None
    key_conf = 0.0
    for frag in reversed(fragments):
        k, s, c = _parse_key_scale(frag)
        if c > key_conf:
            key, scale, key_conf = k, s, c
        if key_conf >= 0.9:
            break

    # --- Instruments ---
    instruments: list[str] = []
    if category in _DRUM_CATEGORIES:
        if category not in ("drums", "perc"):
            instruments = sorted({category, "drums"})
        else:
            instruments = ["drums"]
    elif category in _MELODIC_CATEGORIES:
        instruments = [category]
    elif category == "vocal":
        instruments = ["vocal"]
    elif category == "fx":
        instruments = ["fx"]
    elif category == "bass":
        instruments = ["bass"]

    # --- Overall confidence ---
    fired = [c for c in (kind_conf, cat_conf, bpm_conf, key_conf) if c > 0]
    confidence = round(sum(fired) / len(fired), 3) if fired else 0.0

    return {
        "kind": kind,
        "category": category,
        "key": key,
        "scale": scale,
        "bpm": bpm,
        "instruments": instruments,
        "confidence": confidence,
        "source": "name",
    }

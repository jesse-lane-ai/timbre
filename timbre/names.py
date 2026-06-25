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
# More specific rules come before generic rollups.
_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    # drums — specific before generic
    (("snap",), "snap"),
    (("rim", "rimshot", "sidestick", "side_stick", "side-stick"), "rim"),
    (("kick", "bd", "bassdrum", "bass_drum", "bass-drum", "kik"), "kick"),
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
    # bass — specific before generic
    (("subbass", "sub_bass", "sub-bass", "sub"), "sub"),
    (("reese",), "reese"),
    (("808", "eight08"), "808"),
    (("bass",), "bass"),
    # sound design / fx — specific before the generic fx bucket
    (("riser", "uplifter", "uplift"), "riser"),
    (("sweep", "downlifter", "downshifter"), "sweep"),
    (("impact", "boom", "slam"), "impact"),
    (("drone",), "drone"),
    (("texture", "tex"), "texture"),
    (("ambience", "ambient", "atmos", "atmosphere"), "ambience"),
    (("foley",), "foley"),
    (("field", "fieldrec"), "field"),
    (("fx", "sfx", "transition", "noise"), "fx"),
    # tonal / synth
    (("pluck",), "pluck"),
    (("lead",), "lead"),
    (("arp",), "arp"),
    (("chord", "chords", "chd"), "chord"),
    (("synth",), "synth"),
    (("pad",), "pad"),
    (("stab",), "stab"),
    (("melody", "melodic", "mel"), "melody"),
    (("piano", "rhodes", "wurli", "wurlitzer"), "piano"),
    (("keys", "key"), "keys"),
    # acoustic instruments
    (("guitar", "gtr"), "guitar"),
    (("strings", "violin", "cello", "viola"), "strings"),
    (("brass", "trumpet", "sax", "trombone", "horn"), "brass"),
    # vocal
    (("vocal", "vox", "voice", "acap", "sing"), "vocal"),
]

# Sub-category labels, used to derive instrument tags from a category.
_DRUM_CATEGORIES = {"kick", "snare", "clap", "snap", "hat", "tom", "crash", "ride", "rim", "perc", "drums"}
_MELODIC_CATEGORIES = {"lead", "synth", "pad", "pluck", "chord", "melody", "keys", "piano", "arp", "stab"}
# Categories that ARE a single instrument tag (the tag == the category name).
_INSTRUMENT_CATEGORIES = {"bass", "sub", "reese", "808", "guitar", "strings", "brass"}
# Sound-design categories that all roll up to the "fx" instrument tag.
_FX_CATEGORIES = {"fx", "riser", "sweep", "impact", "drone", "texture", "ambience", "foley", "field", "noise"}

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

# Some packs (Loopmasters/Splice "<bpm>_<key>_<name>" convention) spell sharps
# with a trailing "s" — "Ds" = D#, "As" = A# — because "#" is filesystem-hostile.
# Recognized only as an isolated, delimited token with an uppercase note + lower
# "s", so it won't fire on words like "As"/"Gs" mid-name.
_KEY_S_RE = re.compile(
    rf"(?<![A-Za-z0-9])([A-G])s{_SCALE_KEYWORD}?(?![A-Za-z0-9])",
)
# Theoretical sharps that name a natural — normalize to the common spelling.
_SHARP_NORMALIZE = {"E#": "F", "B#": "C"}

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
# Long-form captures: field recordings, jams, takes, voice memos, full mixes.
_RECORDING_RE = re.compile(
    r"(?<![a-z])(?:recording|rec|field[\s_-]?rec(?:ording)?|voice[\s_-]?memo|jam|take\d*|bounce|full[\s_-]?mix|full[\s_-]?song|master)(?![a-z])",
    re.I,
)


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
    for regex, is_sharp_s in ((_KEY_RE, False), (_KEY_S_RE, True)):
        for m in regex.finditer(text):
            note_raw = m.group(1)
            scale_raw = (m.group(2) or "").lower().rstrip("0123456789")

            if is_sharp_s:
                note = note_raw.upper() + "#"
                note = _SHARP_NORMALIZE.get(note, note)
            else:
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
    if _RECORDING_RE.search(tl):
        return "recording"
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
    elif category in _INSTRUMENT_CATEGORIES:
        instruments = [category]
    elif category in _FX_CATEGORIES:
        instruments = ["fx"]
    elif category == "vocal":
        instruments = ["vocal"]

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

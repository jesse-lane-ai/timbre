"""``heuristic`` recognizer — local, zero new dependencies, deterministic.

Reuses the spectral/temporal features already computed by
``clips.audio_analysis._AnalysisCache`` (shared with ``--analyze``'s BPM pass
so audio is loaded at most once per file):

  * Loops: relabel ``detect_warp_via_analysis``'s beats/melodic/harmonic/
    vocal/complex into the loop category vocabulary (drum/melodic/chord/
    vocal/full).
  * One-shots: classify with spectral centroid / rolloff / zero-crossing rate
    / log-attack-time / percussive ratio into the one-shot vocabulary
    (kick/snare/hat/clap/tom/crash/ride/perc/bass/vocal/melody/...).

Leaves ``instruments`` empty — the heuristic backend has no basis for
multi-label instrument tagging; that's left to the embedding/generative
backends. Confidence is a fixed per-rule constant (module-level tuning
knobs below), not a continuous score.
"""

from __future__ import annotations

from typing import Callable

from .. import audio_analysis
from .types import FileProbe, Recognition

NAME = "heuristic"

# --- one-shot spectral thresholds (module constants, tunable) --------------

# Centroid below this (Hz) reads as "low/boomy" — kick/808/bass territory.
LOW_CENTROID_HZ = 400.0
# Centroid above this (Hz) reads as "bright" — hats/cymbals territory.
HIGH_CENTROID_HZ = 4000.0

# Rolloff above this (Hz) confirms a bright/noisy spectrum (cymbals vs. snare/clap).
HIGH_ROLLOFF_HZ = 7000.0

# Zero-crossing rate above this is "noisy" (snare/clap/hat); below is "tonal".
HIGH_ZCR = 0.15

# log10(seconds) attack time at/under this is "instant" (drum hits).
FAST_ATTACK_LOG = -1.5  # ~32ms

# Percussive-ratio above this confirms a percussive (non-tonal) one-shot.
HIGH_PERCUSSIVE_RATIO = 0.6

# Confidence assigned to every heuristic verdict — fixed, not continuous;
# tuned low enough that filename keywords (category_source="filename") and
# higher-confidence model backends both win over it by default.
CONFIDENCE = 0.55

# Relabel detect_warp_via_analysis's clip-warp modes into loop categories.
_WARP_TO_LOOP_CATEGORY: dict[str, str] = {
    "beats": "drum",
    "melodic": "melodic",
    "harmonic": "chord",
    "vocal": "vocal",
    "complex": "full",
}


def _classify_oneshot(cache: audio_analysis._AnalysisCache) -> str:
    centroid = cache.spectral_centroid()
    rolloff = cache.spectral_rolloff()
    zcr = cache.zero_crossing_rate()
    attack = cache.log_attack_time()
    percussive_ratio = cache.percussive_ratio()
    voiced_ratio = cache.voiced_ratio()
    pitch_stability = cache.pitch_stability()

    fast_attack = attack <= FAST_ATTACK_LOG
    is_percussive = percussive_ratio >= HIGH_PERCUSSIVE_RATIO

    # Bright + noisy + fast attack -> cymbals/hats.
    if centroid >= HIGH_CENTROID_HZ and rolloff >= HIGH_ROLLOFF_HZ and zcr >= HIGH_ZCR:
        return "hat"

    # Low/boomy + tonal + percussive -> kick / 808.
    if centroid <= LOW_CENTROID_HZ and zcr < HIGH_ZCR and fast_attack:
        if voiced_ratio > 0.4 and pitch_stability < 0.05:
            return "808"
        return "kick"

    # Mid-bright + noisy + percussive -> snare/clap.
    if is_percussive and zcr >= HIGH_ZCR and fast_attack:
        if centroid >= HIGH_CENTROID_HZ * 0.5:
            return "clap"
        return "snare"

    # Tonal, stable pitch, voiced -> melodic stab or vocal one-shot.
    if voiced_ratio > 0.5:
        if cache.formant_strength() > 0.6:
            return "vocal"
        if pitch_stability < 0.05:
            return "stab" if fast_attack else "melody"

    # Low/boomy but not fast-attack -> sustained bass.
    if centroid <= LOW_CENTROID_HZ:
        return "bass"

    # Percussive but didn't match a specific drum shape -> generic perc.
    if is_percussive:
        return "perc"

    return "fx"


def _classify_loop(cache: audio_analysis._AnalysisCache) -> str:
    warp = audio_analysis.detect_warp_via_analysis(cache.path, cache)
    return _WARP_TO_LOOP_CATEGORY.get(warp, "full")


class HeuristicRecognizer:
    """Local, deterministic, zero-extra-deps recognizer.

    ``cache_provider``, if given, is called with a file path and should return
    (and memoize) an ``_AnalysisCache`` for it — lets ``library._index_folder``
    share one cache per file between BPM/warp analysis and recognition, so
    audio is loaded at most once. Defaults to a fresh cache per file.
    """

    name = NAME

    def __init__(self, cache_provider: Callable[[str], audio_analysis._AnalysisCache] | None = None):
        self._cache_provider = cache_provider or audio_analysis._AnalysisCache

    def recognize(self, items: list[FileProbe]) -> list[Recognition | None]:
        results: list[Recognition | None] = []
        for item in items:
            try:
                cache = self._cache_provider(str(item.path))
                if item.kind == "loop":
                    category = _classify_loop(cache)
                else:
                    category = _classify_oneshot(cache)
            except Exception:
                # Unreadable/corrupt audio — defer to the filename guess.
                results.append(None)
                continue
            results.append(
                Recognition(category=category, instruments=[], source=NAME, confidence=CONFIDENCE)
            )
        return results

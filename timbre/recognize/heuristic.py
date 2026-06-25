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

Derives ``instruments`` only where the category itself names a concrete
instrument it's reliable on (drums → ``drums`` + the specific hit, bass
family, vocal, and the named acoustic/keyboard categories). The abstract
synth-role categories (``stab``/``melody``/``pad``/``lead``/``arp``/...) carry
no instrument tag — that's left to the embedding/generative backends.
Confidence is a fixed per-rule constant (module-level tuning knobs below),
not a continuous score.
"""

from __future__ import annotations

from typing import Callable

from .. import audio_analysis
from .types import FileProbe, Recognition

NAME = "heuristic"

# Categories the heuristic is reliable enough on to imply an instrument tag.
# Drum hits also carry the generic "drums" tag; "perc"/"drum" stay bare.
_DRUM_INSTRUMENT_CATEGORIES = {
    "kick", "snare", "clap", "snap", "hat", "tom", "crash", "ride", "rim",
    "perc", "drum",
}
# Categories whose name *is* the instrument tag (one-shot + loop vocab).
_SELF_INSTRUMENT_CATEGORIES = {
    "bass", "808", "sub", "reese", "guitar", "strings", "brass",
    "piano", "keys", "vocal",
}

# --- one-shot spectral thresholds (module constants, tunable) --------------

# Centroid below this (Hz) reads as "low/boomy" — kick/808/bass territory.
LOW_CENTROID_HZ = 400.0
# Centroid above this (Hz) reads as "bright" — hats/cymbals territory.
HIGH_CENTROID_HZ = 4000.0

# Rolloff above this (Hz) confirms a bright/noisy spectrum (cymbals vs. snare/clap).
HIGH_ROLLOFF_HZ = 7000.0

# Zero-crossing rate above this is "noisy" (snare/clap/hat); below is "tonal".
HIGH_ZCR = 0.15

# ZCR below this reads as a tonal (non-noisy) hit — the reliable kick/bass
# discriminator. Centroid alone is unreliable here: a click transient pushes a
# kick's centroid up past LOW_CENTROID_HZ, but its ZCR stays near zero.
TONAL_ZCR = 0.03
# Upper centroid bound for the tonal-low (kick/bass) band; brighter tonal hits
# (bells, plucks) fall through to be handled elsewhere / escalated.
LOWMID_CENTROID_HZ = 2500.0

# Vocal one-shot gate. A voiced tone only reads as `vocal` when it sits in the
# formant-relevant mid band AND its pitch actually moves (real vocals have
# vibrato/inflection) — a flat, perfectly-stable synth tone is a stab/melody,
# not a vocal. (The old `formant_strength` proxy was non-discriminating — every
# sound scored ~12-21 against a 0.6 threshold — so it labeled everything voiced
# as vocal; it's no longer used here.)
VOCAL_CENTROID_MIN = 300.0
VOCAL_CENTROID_MAX = 3500.0
VOCAL_PITCH_MOVEMENT = 0.1

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

    # Tonal (low ZCR) in the low/low-mid band -> kick / bass. ZCR, not centroid,
    # gates this: a kick's click transient can push its centroid well above
    # LOW_CENTROID_HZ while its ZCR stays near zero.
    if zcr < TONAL_ZCR and centroid <= LOWMID_CENTROID_HZ:
        if fast_attack or is_percussive:
            return "kick"
        return "bass"

    # Mid-bright + noisy + percussive -> snare/clap.
    if is_percussive and zcr >= HIGH_ZCR and fast_attack:
        if centroid >= HIGH_CENTROID_HZ * 0.5:
            return "clap"
        return "snare"

    # Vocal one-shot: voiced, in the formant-range mid band, with real pitch
    # movement, and not a percussive hit. (See VOCAL_* constants — the formant
    # proxy used before didn't discriminate, so this leans on band + inflection.)
    if (
        voiced_ratio > 0.5
        and VOCAL_CENTROID_MIN <= centroid <= VOCAL_CENTROID_MAX
        and pitch_stability >= VOCAL_PITCH_MOVEMENT
        and not is_percussive
    ):
        return "vocal"

    # Tonal, stable single pitch -> melodic stab (fast) or sustained melody.
    if voiced_ratio > 0.5 and pitch_stability < 0.05:
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

    @staticmethod
    def _instruments_for_category(category: str) -> list[str]:
        """Concrete instrument tags implied by a reliable category verdict.

        Drum categories yield ``drums`` plus the specific hit (``perc``/``drum``
        stay the bare ``drums``); bass-family, vocal, and the named
        acoustic/keyboard categories map to themselves. Abstract synth-role
        categories get nothing — the heuristic can't name their instrument."""
        if category in _DRUM_INSTRUMENT_CATEGORIES:
            if category in ("perc", "drum"):
                return ["drums"]
            return sorted({category, "drums"})
        if category in _SELF_INSTRUMENT_CATEGORIES:
            return [category]
        return []

    def recognize(self, items: list[FileProbe], on_result=None) -> list[Recognition | None]:
        results: list[Recognition | None] = []
        for item in items:
            try:
                cache = self._cache_provider(str(item.path))
                # Loops and long-form recordings are both whole-context audio —
                # classify them by warp character; only single hits go the
                # spectral-one-shot route.
                if item.kind in ("loop", "recording"):
                    category = _classify_loop(cache)
                else:
                    category = _classify_oneshot(cache)
            except Exception:
                # Unreadable/corrupt audio — defer to the filename guess.
                results.append(None)
                if on_result is not None:
                    try:
                        on_result(item, None)
                    except Exception:
                        pass
                continue
            rec = Recognition(
                category=category,
                instruments=self._instruments_for_category(category),
                source=NAME,
                confidence=CONFIDENCE,
            )
            results.append(rec)
            if on_result is not None:
                try:
                    on_result(item, rec)
                except Exception:
                    pass
        return results

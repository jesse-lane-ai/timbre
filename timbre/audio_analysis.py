"""Native-BPM and warp-mode auto-detection for imported audio clips.

Two-stage pipeline per SPEC.md: instant filename-keyword matching first, then
signal analysis (cached — runs once per import) as a fallback. Detection is
implemented with `librosa` alone; `aubio` is listed in the spec's tech-stack
table but is functionally redundant with librosa for tempo/onset estimation
and requires native build toolchains that aren't reliably available, so this
implementation standardizes on librosa for both BPM and warp detection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

WARP_MODES = ("beats", "melodic", "harmonic", "vocal", "complex")

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles — relative perceived stability of each
# pitch class within a major/minor tonal context, starting from the tonic.
# Estimating a key means rotating these to all 12 roots and correlating each
# against the clip's chroma-energy distribution; the best match wins.
_KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_FILENAME_WARP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "beats": ("drum", "loop", "beat", "perc", "hat", "kick", "snare"),
    "melodic": ("bass", "lead", "melody", "arp", "mono"),
    "harmonic": ("pad", "chord", "keys", "synth", "harm", "atmo"),
    "vocal": ("vox", "vocal", "voice", "acap", "sing"),
}

# e.g. "loop_135bpm.wav", "drums-128-bpm.wav", "perc 90.wav"
_BPM_FILENAME_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*[-_]?\s*bpm|bpm\s*[-_]?\s*(\d{2,3}(?:\.\d+)?)", re.I)
_BARE_NUMBER_RE = re.compile(r"(?<![\d.])(\d{2,3})(?![\d.])")

# Loop vs one-shot ("kind") from an explicit filename keyword. "loop" is matched
# as a substring (it shows up glued into compounds like "drumloop"); the one-shot
# tokens use word boundaries so "white" doesn't read as "hit". Bare "shot" is
# deliberately excluded — it appears in phrase names like "BIG SHOT" that are
# actually loops, so only the "one shot"/"oneshot" form counts as one-shot.
_ONESHOT_FILENAME_RE = re.compile(r"(?<![a-z])(?:one[\s_-]?shot|hit|stab)(?![a-z])")


def detect_warp_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for mode, keywords in _FILENAME_WARP_KEYWORDS.items():
        if any(kw in stem for kw in keywords):
            return mode
    return None


def detect_bpm_from_filename(filename: str) -> float | None:
    stem = Path(filename).stem.lower()
    m = _BPM_FILENAME_RE.search(stem)
    if m:
        value = m.group(1) or m.group(2)
        return float(value)
    # Fall back to a bare 2-3 digit number in plausible BPM range.
    m = _BARE_NUMBER_RE.search(stem)
    if m:
        value = float(m.group(1))
        if 40.0 <= value <= 220.0:
            return value
    return None


def detect_kind_from_filename(filename: str) -> str | None:
    """Loop vs one-shot from an explicit filename keyword, or None when absent.

    Only an unambiguous keyword counts here — duration/bar-alignment fill in the
    rest (see ``library._detect_kind``). "loop" anywhere wins; a bounded
    one-shot token (oneshot/one-shot/hit/stab/shot) is the negative signal.
    """
    stem = Path(filename).stem.lower()
    if "loop" in stem:
        return "loop"
    if _ONESHOT_FILENAME_RE.search(stem):
        return "one-shot"
    return None


def probe_duration_seconds(path: str) -> float | None:
    """Duration in seconds read from the audio *header* (no full decode), or
    None when the header can't be read — a corrupt/placeholder file or a format
    libsndfile can't open. Cheap enough to run on every indexed file."""
    try:
        info = sf.info(path)
    except Exception:
        return None
    if not info.samplerate:
        return None
    return info.frames / float(info.samplerate)


class _AnalysisCache:
    """Loads and analyzes the audio signal once; subsequent lookups are free."""

    def __init__(self, path: str):
        self.path = path
        self._y: np.ndarray | None = None
        self._sr: int | None = None
        self._tempo: float | None = None
        self._onset_env: np.ndarray | None = None
        self._harmonic: np.ndarray | None = None
        self._percussive: np.ndarray | None = None
        self._f0: np.ndarray | None = None
        self._voiced_ratio: float | None = None
        self._chroma_energy: np.ndarray | None = None
        self._spectral_centroid: float | None = None
        self._spectral_rolloff: float | None = None
        self._zcr: float | None = None
        self._log_attack_time: float | None = None

    def _load(self):
        if self._y is None:
            self._y, self._sr = librosa.load(self.path, sr=None, mono=True)

    @property
    def y(self) -> np.ndarray:
        self._load()
        return self._y

    @property
    def sr(self) -> int:
        self._load()
        return self._sr

    def tempo(self) -> float:
        if self._tempo is None:
            onset_env = self._onset_envelope()
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)
            self._tempo = float(np.atleast_1d(tempo)[0])
        return self._tempo

    def _onset_envelope(self) -> np.ndarray:
        if self._onset_env is None:
            self._onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        return self._onset_env

    def _hpss(self) -> tuple[np.ndarray, np.ndarray]:
        if self._harmonic is None:
            self._harmonic, self._percussive = librosa.effects.hpss(self.y)
        return self._harmonic, self._percussive

    def transient_density(self) -> float:
        """Onsets per second — high for drum loops/percussion."""
        onset_env = self._onset_envelope()
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        duration = len(self.y) / self.sr
        return len(onsets) / duration if duration > 0 else 0.0

    def percussive_ratio(self) -> float:
        """Fraction of total energy that is percussive (vs. harmonic)."""
        harmonic, percussive = self._hpss()
        h_energy = float(np.sum(harmonic ** 2))
        p_energy = float(np.sum(percussive ** 2))
        total = h_energy + p_energy
        return p_energy / total if total > 0 else 0.0

    def _pitch_track(self):
        if self._f0 is None:
            f0, voiced_flag, _ = librosa.pyin(
                self.y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
            )
            self._f0 = f0
            voiced = voiced_flag.astype(bool) if voiced_flag is not None else np.zeros_like(f0, dtype=bool)
            self._voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
        return self._f0

    def voiced_ratio(self) -> float:
        self._pitch_track()
        return self._voiced_ratio or 0.0

    def pitch_stability(self) -> float:
        """Coefficient of variation of the voiced f0 track — low means stable
        single pitch (melodic), high means many concurrent/changing pitches."""
        f0 = self._pitch_track()
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 2:
            return 1.0
        mean = float(np.mean(voiced))
        std = float(np.std(voiced))
        return std / mean if mean > 0 else 1.0

    def chroma_energy(self) -> np.ndarray:
        """Mean energy per pitch class (C, C#, D, ...) across the clip — the
        input to Krumhansl-Schmuckler key estimation."""
        if self._chroma_energy is None:
            chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
            self._chroma_energy = chroma.mean(axis=1)
        return self._chroma_energy

    def formant_strength(self) -> float:
        """Crude formant-structure proxy: spectral-contrast variance in the
        speech-relevant bands. Higher implies stronger formant structure."""
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        return float(np.mean(np.var(contrast, axis=1)))

    def spectral_centroid(self) -> float:
        """Mean spectral centroid in Hz — "brightness". Low for kicks/bass,
        high for hats/cymbals."""
        if self._spectral_centroid is None:
            centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
            self._spectral_centroid = float(np.mean(centroid))
        return self._spectral_centroid

    def spectral_rolloff(self) -> float:
        """Mean spectral rolloff in Hz (frequency below which 85% of the energy
        is concentrated) — another brightness proxy, robust to single peaks."""
        if self._spectral_rolloff is None:
            rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
            self._spectral_rolloff = float(np.mean(rolloff))
        return self._spectral_rolloff

    def zero_crossing_rate(self) -> float:
        """Mean zero-crossing rate — near-zero for low tonal sounds (kick/bass),
        high for noisy/broadband sounds (hats/snares/claps)."""
        if self._zcr is None:
            zcr = librosa.feature.zero_crossing_rate(self.y)
            self._zcr = float(np.mean(zcr))
        return self._zcr

    def log_attack_time(self) -> float:
        """log10 of the time (seconds) from onset to the signal's peak amplitude
        — short for percussive hits, longer for swelling/sustained sounds."""
        if self._log_attack_time is None:
            y = self.y
            if len(y) == 0:
                self._log_attack_time = -3.0
            else:
                peak_idx = int(np.argmax(np.abs(y)))
                attack_seconds = max(peak_idx / float(self.sr), 1e-4)
                self._log_attack_time = float(np.log10(attack_seconds))
        return self._log_attack_time


def detect_bpm_via_analysis(path: str, cache: _AnalysisCache | None = None) -> float:
    cache = cache or _AnalysisCache(path)
    return round(cache.tempo(), 2)


def detect_key_via_analysis(path: str, cache: _AnalysisCache | None = None) -> tuple[str, str]:
    """Krumhansl-Schmuckler key estimation: rotate the major/minor profiles to
    all 12 roots, correlate each against the clip's chroma-energy distribution,
    and return the (root, scale) pair with the strongest correlation."""
    cache = cache or _AnalysisCache(path)
    energy = cache.chroma_energy()

    best = ("C", "major", -np.inf)
    for shift in range(12):
        for scale, profile in (("major", _KS_MAJOR_PROFILE), ("minor", _KS_MINOR_PROFILE)):
            score = float(np.corrcoef(energy, np.roll(profile, shift))[0, 1])
            if score > best[2]:
                best = (PITCH_CLASS_NAMES[shift], scale, score)

    return best[0], best[1]


def detect_warp_via_analysis(path: str, cache: _AnalysisCache | None = None) -> str:
    cache = cache or _AnalysisCache(path)

    transient_density = cache.transient_density()
    percussive_ratio = cache.percussive_ratio()
    voiced_ratio = cache.voiced_ratio()
    pitch_stability = cache.pitch_stability()
    formant_strength = cache.formant_strength()

    # Stage-2 heuristics, applied in the order described in SPEC.md.
    if transient_density > 2.5 and percussive_ratio > 0.5:
        return "beats"
    if voiced_ratio > 0.5 and formant_strength > 0.6:
        return "vocal"
    if voiced_ratio > 0.4 and pitch_stability < 0.05:
        return "melodic"
    if voiced_ratio > 0.3 and pitch_stability >= 0.05:
        return "harmonic"
    return "complex"


def analyze_clip(path: str, filename: str) -> dict[str, Any]:
    """Run the full detection pipeline for native BPM, warp mode, and musical
    key. BPM/warp use the two-stage filename-then-signal-analysis pipeline
    from SPEC.md; key has no reliable filename heuristic, so it always comes
    from chroma analysis (Krumhansl-Schmuckler key-finding).

    Returns {"native_bpm", "warp", "source", "key", "scale"} where `source`
    records which stage produced the *warp* result (bpm uses the same
    precedence but isn't separately reported, per the example in SPEC.md).
    """
    warp = detect_warp_from_filename(filename)
    bpm = detect_bpm_from_filename(filename)
    source = "filename" if warp is not None else None

    cache = None
    if warp is None or bpm is None:
        cache = _AnalysisCache(path)
        if warp is None:
            warp = detect_warp_via_analysis(path, cache)
            source = "tempo_analysis"
        if bpm is None:
            bpm = detect_bpm_via_analysis(path, cache)
            source = source or "tempo_analysis"

    key, scale = detect_key_via_analysis(path, cache)

    return {"native_bpm": bpm, "warp": warp, "source": source or "filename", "key": key, "scale": scale}

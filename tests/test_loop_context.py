"""Loop-context disambiguation: a short, *unpitched* one-shot the model captions
as tonal gets re-captioned as a four-on-the-floor loop and, if the loop reads as
a drum, overridden. Pitched one-shots are left alone.

Uses real synthesized audio (the gate depends on voiced_ratio, computed from the
signal) with a captioner stub that returns a tonal caption for the raw hit and a
drum caption for the constructed loop (identified by its `_4otf` suffix).
"""

import numpy as np
import soundfile as sf
import pytest

from timbre.recognize.ace_step import AceStepRecognizer, _build_four_on_floor
from timbre.recognize.types import FileProbe


SR = 44100


def _write(path, y):
    sf.write(str(path), y.astype(np.float32), SR)
    return str(path)


def _noise_hit(path):
    # A short, percussive, *unpitched* burst -> low voiced_ratio (drum-like).
    n = int(0.3 * SR)
    env = np.exp(-np.linspace(0, 12, n))
    return _write(path, np.random.default_rng(0).standard_normal(n) * env)


def _tonal_hit(path):
    # A sustained pure tone -> high voiced_ratio (a real note).
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    return _write(path, 0.5 * np.sin(2 * np.pi * 220.0 * t))


class _StubCaptioner:
    """Tonal caption for raw hits; drum caption for the `_4otf` loop file."""

    def __init__(self):
        self._model = object()

    def _load(self):
        return self._model, None

    def _load_mode(self):
        return "full"

    def caption(self, path):
        return self.caption_batch([path])[0]

    def caption_batch(self, paths):
        out = []
        for p in paths:
            if "_4otf" in str(p):
                out.append("a dry punchy kick drum in a four on the floor pattern")
            else:
                out.append("a single sustained chord on a pipe organ")
        return out


def _recognizer():
    rec = AceStepRecognizer.__new__(AceStepRecognizer)  # skip dependency load
    rec._captioner = _StubCaptioner()
    return rec


def test_build_four_on_floor_length(tmp_path):
    src = _noise_hit(tmp_path / "hit.wav")
    loop = _build_four_on_floor(src)
    y, sr = sf.read(loop)
    # 2 bars at 120 BPM = 4.0 s of grid, plus the tail of the last hit.
    assert sr == SR
    assert 4.0 <= len(y) / sr <= 4.5


class _LowVoicedCache:
    """Stub cache forcing an unpitched (drum-like) voiced_ratio, so the gate
    doesn't hinge on pyin's read of a synthetic signal."""

    def __init__(self, path):
        self.path = path

    def voiced_ratio(self):
        return 0.2


def test_unpitched_tonal_caption_is_overridden_to_drum(tmp_path, monkeypatch):
    monkeypatch.setattr("timbre.audio_analysis._AnalysisCache", _LowVoicedCache)
    src = _noise_hit(tmp_path / "mystery_hit.wav")
    rec = _recognizer()
    out = rec.recognize([FileProbe(path=src, filename="mystery_hit.wav",
                                   duration=0.3, kind="one-shot")])
    # Raw caption ("organ chord") would be `chord`; loop context flips it.
    assert out[0].category in ("kick", "snare", "perc")


def test_pitched_one_shot_is_not_looped(tmp_path):
    src = _tonal_hit(tmp_path / "note.wav")
    rec = _recognizer()
    out = rec.recognize([FileProbe(path=src, filename="note.wav",
                                   duration=0.8, kind="one-shot")])
    # Voiced tone stays with its raw (tonal) verdict — no drum override.
    assert out[0].category not in ("kick", "snare", "perc")


def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_LOOP_CONTEXT", "0")
    src = _noise_hit(tmp_path / "mystery_hit.wav")
    rec = _recognizer()
    out = rec.recognize([FileProbe(path=src, filename="mystery_hit.wav",
                                   duration=0.3, kind="one-shot")])
    # With the pass off, the raw tonal verdict stands.
    assert out[0].category == "chord"

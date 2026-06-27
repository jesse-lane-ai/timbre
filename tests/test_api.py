"""Tests for the high-level fused API (`timbre.classify`)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

import timbre


def _kick(path) -> None:
    sr = 44100
    t = np.linspace(0, 0.3, int(sr * 0.3), False)
    env = np.exp(-t * 25)
    freq = 120 * np.exp(-t * 8)
    x = np.sin(2 * np.pi * np.cumsum(freq) / sr) * env
    sf.write(str(path), x.astype("float32"), sr)


def test_classify_path_fuses_name_and_content(tmp_path):
    p = tmp_path / "kick_120bpm_Cmin.wav"
    _kick(p)
    tags = timbre.classify(str(p), backend="heuristic")
    assert tags.category == "kick"
    # "kick" restates the category, so it's deduped out; the family tag remains.
    assert "kick" not in tags.instruments
    assert "drums" in tags.instruments
    assert tags.bpm == 120.0
    assert tags.key == "C" and tags.scale == "minor"
    assert tags.duration is not None
    assert tags.backend == "heuristic"
    assert tags.path == str(p)


def test_classify_bytes_uses_filename_hint(tmp_path):
    p = tmp_path / "src.wav"
    _kick(p)
    raw = p.read_bytes()
    tags = timbre.classify(raw, filename="snare_90bpm.wav", backend="heuristic")
    # name pass drives category/bpm from the hint; temp path is not leaked.
    assert tags.category == "snare"
    assert tags.bpm == 90.0
    assert tags.path is None


def test_classify_many_batches(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"kick_{i}.wav"
        _kick(p)
        paths.append(str(p))
    out = timbre.classify_many(paths, backend="heuristic")
    assert len(out) == 3
    assert all(t.category == "kick" for t in out)

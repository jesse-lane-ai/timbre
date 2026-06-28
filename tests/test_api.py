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


# --- auto backend selection + multi-backend merge ----------------------------

def test_select_recognizers_auto_adapts_to_availability(monkeypatch):
    import timbre.api as api
    from timbre.errors import BadInputError

    avail = {"clap", "ace-step"}

    class _Fake:
        def __init__(self, name): self.name = name

    def fake_get(name):
        if name in avail:
            return _Fake(name)
        raise BadInputError("not installed")

    monkeypatch.setattr(api, "get_recognizer", fake_get)

    avail = {"clap", "ace-step"}                       # both -> union, no heuristic
    assert [n for n, _ in api._select_recognizers("auto")] == ["clap", "ace-step"]

    avail = {"clap"}                                   # one -> that + heuristic
    assert [n for n, _ in api._select_recognizers("auto")] == ["clap", "heuristic"]

    avail = {"ace-step"}
    assert [n for n, _ in api._select_recognizers("auto")] == ["ace-step", "heuristic"]

    avail = set()                                      # none -> heuristic only
    assert [n for n, _ in api._select_recognizers("auto")] == ["heuristic"]


def test_merge_recognitions_unions_tags():
    from timbre.api import _merge_recognitions
    from timbre.recognize.types import Recognition, genre_score

    a = Recognition("drum", ["kick"], "clap", 0.4,
                    genres=[genre_score("house", 0.5, "clap")])
    b = Recognition("drum", ["snare"], "ace-step", 0.7, caption="a drum loop",
                    genres=[genre_score("hip hop", 0.6, "ace-step"),
                            genre_score("house", 0.2, "ace-step")])
    m = _merge_recognitions([a, b])
    assert m.category == "drum"                        # agreement
    assert m.instruments == ["kick", "snare"]          # union
    assert m.caption == "a drum loop"
    assert m.confidence == 0.7
    assert m.source == "ace-step+clap"
    assert {g["genre"] for g in m.genres} == {"house", "hip hop"}
    house = next(g for g in m.genres if g["genre"] == "house")
    assert house["score"] == 0.5                       # higher-scoring dup kept


def test_merge_recognitions_disagreement_uses_confidence():
    from timbre.api import _merge_recognitions
    from timbre.recognize.types import Recognition

    # Two backends, no majority -> highest confidence wins the category.
    a = Recognition("kick", [], "clap", 0.4)
    b = Recognition("perc", [], "ace-step", 0.7)
    assert _merge_recognitions([a, b]).category == "perc"


def test_merge_recognitions_all_none():
    from timbre.api import _merge_recognitions
    assert _merge_recognitions([None, None]) is None

"""Unit tests for the pluggable recognition backends (`timbre.recognize`)."""

from pathlib import Path

import pytest

from timbre.errors import BadInputError
from timbre.recognize import get_recognizer, list_backends
from timbre.recognize.heuristic import HeuristicRecognizer, _classify_loop, _classify_oneshot
from timbre.recognize.types import FileProbe, Recognition


class _FakeCache:
    """Duck-typed stand-in for `_AnalysisCache` with fixed feature values —
    deterministic, no audio I/O."""

    def __init__(self, path="fake.wav", **features):
        self.path = path
        self._features = {
            "spectral_centroid": 1000.0,
            "spectral_rolloff": 2000.0,
            "zero_crossing_rate": 0.05,
            "log_attack_time": -1.0,
            "percussive_ratio": 0.3,
            "voiced_ratio": 0.0,
            "pitch_stability": 1.0,
            "formant_strength": 0.0,
        }
        self._features.update(features)

    def spectral_centroid(self):
        return self._features["spectral_centroid"]

    def spectral_rolloff(self):
        return self._features["spectral_rolloff"]

    def zero_crossing_rate(self):
        return self._features["zero_crossing_rate"]

    def log_attack_time(self):
        return self._features["log_attack_time"]

    def percussive_ratio(self):
        return self._features["percussive_ratio"]

    def voiced_ratio(self):
        return self._features["voiced_ratio"]

    def pitch_stability(self):
        return self._features["pitch_stability"]

    def formant_strength(self):
        return self._features["formant_strength"]


# --- one-shot classification -------------------------------------------------

def test_classify_oneshot_kick():
    cache = _FakeCache(spectral_centroid=150.0, zero_crossing_rate=0.02, log_attack_time=-2.0)
    assert _classify_oneshot(cache) == "kick"


def test_classify_oneshot_808_when_pitched():
    cache = _FakeCache(
        spectral_centroid=150.0, zero_crossing_rate=0.02, log_attack_time=-2.0,
        voiced_ratio=0.6, pitch_stability=0.02,
    )
    assert _classify_oneshot(cache) == "808"


def test_classify_oneshot_hat():
    cache = _FakeCache(spectral_centroid=6000.0, spectral_rolloff=9000.0, zero_crossing_rate=0.3, log_attack_time=-2.0)
    assert _classify_oneshot(cache) == "hat"


def test_classify_oneshot_snare():
    cache = _FakeCache(
        spectral_centroid=1500.0, zero_crossing_rate=0.3, log_attack_time=-2.0,
        percussive_ratio=0.8,
    )
    assert _classify_oneshot(cache) == "snare"


def test_classify_oneshot_clap():
    cache = _FakeCache(
        spectral_centroid=2200.0, zero_crossing_rate=0.3, log_attack_time=-2.0,
        percussive_ratio=0.8,
    )
    assert _classify_oneshot(cache) == "clap"


def test_classify_oneshot_vocal():
    cache = _FakeCache(voiced_ratio=0.7, pitch_stability=1.0, formant_strength=0.8)
    assert _classify_oneshot(cache) == "vocal"


def test_classify_oneshot_melody_stab():
    cache = _FakeCache(voiced_ratio=0.6, pitch_stability=0.01, formant_strength=0.1, log_attack_time=-2.0)
    assert _classify_oneshot(cache) == "stab"


def test_classify_oneshot_melody_sustained():
    cache = _FakeCache(voiced_ratio=0.6, pitch_stability=0.01, formant_strength=0.1, log_attack_time=0.0)
    assert _classify_oneshot(cache) == "melody"


def test_classify_oneshot_bass():
    cache = _FakeCache(spectral_centroid=200.0, zero_crossing_rate=0.05, log_attack_time=0.0)
    assert _classify_oneshot(cache) == "bass"


def test_classify_oneshot_perc():
    cache = _FakeCache(spectral_centroid=1500.0, zero_crossing_rate=0.05, percussive_ratio=0.7, log_attack_time=0.0)
    assert _classify_oneshot(cache) == "perc"


def test_classify_oneshot_fx_fallback():
    cache = _FakeCache(spectral_centroid=1500.0, zero_crossing_rate=0.05, percussive_ratio=0.1, log_attack_time=0.0)
    assert _classify_oneshot(cache) == "fx"


# --- loop classification (relabel detect_warp_via_analysis) ------------------

@pytest.mark.parametrize("warp,expected", [
    ("beats", "drum"),
    ("melodic", "melodic"),
    ("harmonic", "chord"),
    ("vocal", "vocal"),
    ("complex", "full"),
])
def test_classify_loop_relabels_warp(monkeypatch, warp, expected):
    monkeypatch.setattr("timbre.recognize.heuristic.audio_analysis.detect_warp_via_analysis", lambda path, cache=None: warp)
    cache = _FakeCache()
    assert _classify_loop(cache) == expected


# --- HeuristicRecognizer batch API --------------------------------------------

def test_heuristic_recognizer_batch(monkeypatch):
    fakes = {
        "kick.wav": _FakeCache("kick.wav", spectral_centroid=150.0, zero_crossing_rate=0.02, log_attack_time=-2.0),
        "hat.wav": _FakeCache("hat.wav", spectral_centroid=6000.0, spectral_rolloff=9000.0, zero_crossing_rate=0.3, log_attack_time=-2.0),
    }
    recognizer = HeuristicRecognizer(cache_provider=lambda p: fakes[p])
    items = [
        FileProbe(path=Path("kick.wav"), filename="kick.wav", duration=0.2, kind="one-shot"),
        FileProbe(path=Path("hat.wav"), filename="hat.wav", duration=0.1, kind="one-shot"),
    ]
    results = recognizer.recognize(items)
    assert results[0] == Recognition(category="kick", instruments=[], source="heuristic", confidence=0.55)
    assert results[1] == Recognition(category="hat", instruments=[], source="heuristic", confidence=0.55)


def test_heuristic_recognizer_defers_on_unreadable_file():
    recognizer = HeuristicRecognizer(cache_provider=lambda p: (_ for _ in ()).throw(OSError("bad file")))
    items = [FileProbe(path=Path("broken.wav"), filename="broken.wav", duration=None, kind="one-shot")]
    assert recognizer.recognize(items) == [None]


# --- registry -------------------------------------------------------------

def test_list_backends():
    assert list_backends() == ["ace-step", "clap", "heuristic"]


def test_get_recognizer_unknown_backend_raises():
    with pytest.raises(BadInputError):
        get_recognizer("not-a-backend")


def test_get_recognizer_heuristic():
    assert isinstance(get_recognizer("heuristic"), HeuristicRecognizer)


def test_clap_backend_missing_dependency_raises_actionable_error():
    with pytest.raises(BadInputError, match=r"pip install 'timbre\[clap\]'"):
        get_recognizer("clap")


def test_ace_step_backend_missing_dependency_raises_actionable_error(monkeypatch):
    # Simulate `transformers` being unavailable: the captioner's dependency
    # check should surface an actionable install hint at selection time.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'", name="transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BadInputError, match=r"pip install transformers"):
        get_recognizer("ace-step")


def test_captioner_load_mode_validates(monkeypatch):
    from timbre.recognize.captioner import AceCaptioner

    cap = AceCaptioner()
    for mode in ("full", "8bit", "4bit"):
        monkeypatch.setenv("ACESTEP_CAPTIONER_LOAD", mode)
        assert cap._load_mode() == mode
    monkeypatch.setenv("ACESTEP_CAPTIONER_LOAD", "nope")
    with pytest.raises(BadInputError, match=r"ACESTEP_CAPTIONER_LOAD"):
        cap._load_mode()


def test_captioner_full_mode_needs_no_quant_config():
    from timbre.recognize.captioner import AceCaptioner

    # `full` must not require bitsandbytes.
    assert AceCaptioner()._quant_config("full") is None


def test_captioner_unload_clears_model():
    from timbre.recognize.captioner import AceCaptioner

    cap = AceCaptioner()
    cap.unload()  # no-op when nothing loaded
    assert cap._model is None and cap._processor is None
    # Simulate a loaded model; unload must drop both references.
    cap._model = object()
    cap._processor = object()
    cap.unload()
    assert cap._model is None and cap._processor is None


def test_captioner_max_audio_seconds_parsing(monkeypatch):
    from timbre.recognize.captioner import AceCaptioner

    cap = AceCaptioner()
    monkeypatch.delenv("ACESTEP_CAPTIONER_AUDIO_SECONDS", raising=False)
    assert cap._max_audio_seconds() == 30.0  # default
    monkeypatch.setenv("ACESTEP_CAPTIONER_AUDIO_SECONDS", "10")
    assert cap._max_audio_seconds() == 10.0
    monkeypatch.setenv("ACESTEP_CAPTIONER_AUDIO_SECONDS", "0")
    assert cap._max_audio_seconds() == 0.5  # floored
    monkeypatch.setenv("ACESTEP_CAPTIONER_AUDIO_SECONDS", "nope")
    assert cap._max_audio_seconds() == 30.0  # bad value falls back


def test_batch_size_env_parsing(monkeypatch):
    from timbre.recognize import ace_step

    monkeypatch.delenv("ACESTEP_CAPTIONER_BATCH", raising=False)
    assert ace_step._batch_size() == 8  # default
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "4")
    assert ace_step._batch_size() == 4
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "0")
    assert ace_step._batch_size() == 1  # clamped to >= 1
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "nope")
    assert ace_step._batch_size() == 8  # non-integer falls back to default


class _FakeCaptioner:
    """Records how files were grouped into generate() calls so we can assert
    the recognizer batches correctly, and can simulate a whole-batch failure."""

    def __init__(self, captions, fail_batch=False):
        self._captions = captions  # filename -> caption (or Exception to raise)
        self.fail_batch = fail_batch
        self.batch_calls = []
        self.single_calls = []
        self._model = object()  # already-loaded sentinel

    def _load(self):
        return self._model, None

    def _load_mode(self):
        return "full"

    def _one(self, path):
        val = self._captions[Path(path).name]
        if isinstance(val, Exception):
            raise val
        return val

    def caption(self, path):
        self.single_calls.append(Path(path).name)
        return self._one(path)

    def caption_batch(self, paths):
        self.batch_calls.append([Path(p).name for p in paths])
        if self.fail_batch:
            raise RuntimeError("batch blew up")
        return [self._one(p) for p in paths]


def _ace_recognizer(captioner):
    from timbre.recognize.ace_step import AceStepRecognizer

    rec = AceStepRecognizer.__new__(AceStepRecognizer)  # skip dependency load
    rec._captioner = captioner
    return rec


def _probe(name, kind="one-shot", duration=0.5):
    return FileProbe(path=Path(f"/tmp/{name}"), filename=name, duration=duration, kind=kind)


def test_ace_recognizer_batches_files(monkeypatch):
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "2")
    cap = _FakeCaptioner({
        "kick.wav": "a deep kick drum",
        "snare.wav": "a snappy snare",
        "hat.wav": "a closed hi-hat",
    })
    rec = _ace_recognizer(cap)
    out = rec.recognize([_probe("kick.wav"), _probe("snare.wav"), _probe("hat.wav")])

    # Grouped into chunks of 2 (2 + 1): the full pair goes through one
    # batched generate(), the trailing single-file chunk uses caption().
    assert cap.batch_calls == [["kick.wav", "snare.wav"]]
    assert cap.single_calls == ["hat.wav"]
    assert [r.category for r in out] == ["kick", "snare", "hat"]
    assert out[0].caption == "a deep kick drum"
    assert all(r.source == "ace-step" for r in out)


def test_ace_recognizer_buckets_by_length(monkeypatch):
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "2")
    # Interleaved short/long input; batching by length should group the two
    # long loops together and the two short hits together, regardless of input
    # order — so no batch mixes a long loop with a short one-shot.
    cap = _FakeCaptioner({
        "hit1.wav": "a short kick",
        "loop1.wav": "a long drum loop",
        "hit2.wav": "a short snare",
        "loop2.wav": "a long bass loop",
    })
    rec = _ace_recognizer(cap)
    probes = [
        _probe("hit1.wav", duration=0.5),
        _probe("loop1.wav", kind="loop", duration=8.0),
        _probe("hit2.wav", duration=0.4),
        _probe("loop2.wav", kind="loop", duration=9.0),
    ]
    out = rec.recognize(probes)

    # Each batch holds only same-scale clips.
    assert ["hit2.wav", "hit1.wav"] in cap.batch_calls  # shorts together
    assert ["loop1.wav", "loop2.wav"] in cap.batch_calls  # longs together
    for batch in cap.batch_calls:
        is_long = ["loop" in n for n in batch]
        assert all(is_long) or not any(is_long), batch
    # Output order matches input order, not processing order.
    assert [r.caption for r in out] == [
        "a short kick", "a long drum loop", "a short snare", "a long bass loop"
    ]


def test_plan_batches_sorts_by_duration():
    from timbre.recognize.ace_step import _plan_batches

    items = [
        _probe("a.wav", duration=0.3),
        _probe("loop1.wav", kind="loop", duration=9.0),
        _probe("b.wav", duration=0.3),
        _probe("loop2.wav", kind="loop", duration=9.0),
        _probe("c.wav", duration=0.3),
    ]
    batches = _plan_batches(items, batch_size=3)
    # Sorted by duration, chunked by batch_size: the three shorts ride together,
    # the two loops form the next batch.
    assert [len(b) for b in batches] == [3, 2]
    assert all(items[i].duration < 1.0 for i in batches[0])
    assert all(items[i].duration == 9.0 for i in batches[1])
    # Single batch when everything fits — a no-op grouping for small folders.
    assert len(_plan_batches(items, batch_size=8)) == 1
    # Every index appears exactly once.
    assert sorted(i for b in batches for i in b) == list(range(len(items)))


def test_ace_recognizer_isolates_bad_file_in_batch(monkeypatch):
    monkeypatch.setenv("ACESTEP_CAPTIONER_BATCH", "3")
    cap = _FakeCaptioner(
        {
            "kick.wav": "a deep kick drum",
            "bad.wav": RuntimeError("decode failed"),
            "snare.wav": "a snappy snare",
        },
        fail_batch=True,
    )
    rec = _ace_recognizer(cap)
    out = rec.recognize([_probe("kick.wav"), _probe("bad.wav"), _probe("snare.wav")])

    # Whole-batch failure retried file-by-file; the bad file defers (None),
    # the others still produce recognitions.
    assert cap.single_calls == ["kick.wav", "bad.wav", "snare.wav"]
    assert out[0].category == "kick"
    assert out[1] is None
    assert out[2].category == "snare"


def test_score_categories_ranks_by_emphasis_not_vocab_order():
    from timbre.recognize.ace_step import _score_categories
    from timbre.recognize.types import LOOP_CATEGORIES

    # The caption leans on "bass" (x3) but also mentions "drum" once. The old
    # mapping took the first vocab entry found — and `drum` precedes `bass` in
    # LOOP_CATEGORIES — so it would have mis-bucketed this as `drum`. Scoring by
    # occurrence count picks `bass`.
    caption = "a deep bass, warm sub bass, bass-forward groove over a light drum"
    ranked = _score_categories(caption, LOOP_CATEGORIES)
    assert ranked[0][0] == "bass"
    assert [c[0] for c in ranked] == ["bass", "drum"]


def test_score_categories_tie_breaks_on_earliest_mention():
    from timbre.recognize.ace_step import _score_categories
    from timbre.recognize.types import LOOP_CATEGORIES

    # Equal counts (1 each) -> the category mentioned first wins, regardless of
    # its position in the vocabulary (melodic sits after drum in LOOP_CATEGORIES).
    ranked = _score_categories("a melodic line over a drum beat", LOOP_CATEGORIES)
    assert ranked[0][0] == "melodic"


def test_score_categories_empty_when_no_match():
    from timbre.recognize.ace_step import _score_categories
    from timbre.recognize.types import LOOP_CATEGORIES

    assert _score_categories("an indescribable noise", LOOP_CATEGORIES) == []

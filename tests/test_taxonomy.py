"""The expanded taxonomy: the `recording` kind and the new category vocab."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

import timbre
from timbre.names import classify_from_names as classify
from timbre.recognize.types import (
    KINDS,
    RECORDING_CATEGORIES,
    categories_for_kind,
)


@pytest.mark.parametrize(
    "name",
    ["jam_session.wav", "field_recording.wav", "voice_memo_01.wav", "band_take2.wav", "full_mix_master.wav"],
)
def test_recording_kind_from_name(name):
    assert classify(name)["kind"] == "recording"


@pytest.mark.parametrize(
    "name,category",
    [
        ("lead_synth_C.wav", "lead"),
        ("pluck_one.wav", "pluck"),
        ("riser_fx.wav", "riser"),
        ("sub_bass_90.wav", "sub"),
        ("808_boom.wav", "808"),          # 808 is its own category now, not "kick"
        ("guitar_strum.wav", "guitar"),
        ("strings_swell.wav", "strings"),
        ("brass_hit.wav", "brass"),
        ("foley_steps.wav", "foley"),
    ],
)
def test_new_categories_from_name(name, category):
    assert classify(name)["category"] == category


def test_808_is_bass_instrument_not_drum():
    r = classify("808_sub.wav")
    # whichever of 808/sub wins, the instrument tag is not a drum rollup
    assert r["category"] in {"808", "sub"}
    assert "drums" not in r["instruments"]


def test_categories_for_kind_routing():
    assert categories_for_kind("recording") == RECORDING_CATEGORIES
    assert categories_for_kind("loop") != categories_for_kind("one-shot")
    # unknown falls back to the one-shot vocab
    assert categories_for_kind("unknown") == categories_for_kind("xyz")
    assert "recording" in KINDS


def test_long_unnamed_file_infers_recording_kind(tmp_path):
    # 35s of audio, name carries no loop/one-shot/recording cue -> recording.
    sr = 22050
    p = tmp_path / "untitled.wav"
    sf.write(str(p), np.zeros(int(sr * 35), dtype="float32"), sr)
    tags = timbre.classify(str(p), backend="heuristic")
    assert tags.kind == "recording"


def test_short_unnamed_file_stays_unknown(tmp_path):
    sr = 22050
    p = tmp_path / "untitled.wav"
    sf.write(str(p), np.zeros(int(sr * 2), dtype="float32"), sr)
    tags = timbre.classify(str(p), backend="heuristic")
    assert tags.kind == "unknown"

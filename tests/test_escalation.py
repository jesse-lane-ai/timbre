"""Unit tests for the scan escalation predicate (`cli._needs_escalation`).

The predicate decides which heuristic verdicts are worth a model second opinion:
empty/catch-all always, audio-*guessed* pitched categories yes, but
name-confirmed or instrument-tagged verdicts and confident drum hits stay local.
"""

from timbre.cli import _needs_escalation
from timbre.api import Tags


def _tags(category, instruments=None, path="x.wav"):
    return Tags(filename=path, kind="unknown", category=category,
                instruments=instruments or [], path=path)


def test_escalate_when_no_category():
    assert _needs_escalation(_tags(None)) is True


def test_escalate_catchall_buckets():
    assert _needs_escalation(_tags("fx")) is True
    assert _needs_escalation(_tags("full")) is True


def test_escalate_audio_guessed_pitched_category():
    # A pitched verdict whose filename yields no category cue is an audio guess.
    assert _needs_escalation(_tags("bass", path="Combo-70s.wav")) is True
    assert _needs_escalation(_tags("vocal", path="spanish triumphant1.wav")) is True


def test_no_escalate_name_confirmed_pitched_category():
    # The filename itself names the category -> trust it, don't escalate.
    assert _needs_escalation(_tags("bass", path="deep_bass_hit.wav")) is False
    assert _needs_escalation(_tags("melodic", path="grand_piano.wav")) is False


def test_no_escalate_drum_categories():
    # The heuristic is reliable on percussion; keep it local regardless of name.
    for cat in ("kick", "snare", "hat", "clap", "perc"):
        assert _needs_escalation(_tags(cat, path="anon.wav")) is False


def test_no_escalate_when_instruments_present():
    # Any instrument tag is enough signal — never escalate.
    assert _needs_escalation(_tags("bass", instruments=["bass"], path="Combo.wav")) is False
    assert _needs_escalation(_tags("fx", instruments=["fx"], path="Combo.wav")) is False

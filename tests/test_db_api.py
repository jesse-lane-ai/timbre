"""Store query/update/delete, the top-level python API, and config resolution."""

from __future__ import annotations

import numpy as np
import soundfile as sf

import timbre
from timbre import config, store
from timbre.api import Tags


def _wav(path) -> None:
    sr = 44100
    t = np.linspace(0, 0.2, int(sr * 0.2), False)
    sf.write(str(path), (np.sin(2 * np.pi * 120 * t) * np.exp(-t * 20)).astype("float32"), sr)


def _seed(con):
    store.update(con, "/a/kick.wav", {"category": "kick", "kind": "one-shot", "bpm": 90, "instruments": ["kick", "drums"]})
    store.update(con, "/a/snare.wav", {"category": "snare", "kind": "one-shot", "bpm": 140})
    store.update(con, "/a/bass.wav", {"category": "bass", "kind": "loop", "bpm": 90})


def test_store_query_filters(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    _seed(con)
    assert [t.category for t in store.query(con, category="kick")] == ["kick"]
    assert {t.category for t in store.query(con, bpm_min=100)} == {"snare"}
    assert {t.category for t in store.query(con, bpm_max=90)} == {"kick", "bass"}
    assert [t.category for t in store.query(con, instrument="drums")] == ["kick"]
    assert {t.category for t in store.query(con, kind="loop")} == {"bass"}
    assert len(store.query(con, limit=2)) == 2


def test_store_update_creates_then_edits_and_marks_edited(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    t = store.update(con, "/x/y.wav", {"category": "kick"})
    assert t.category == "kick" and t.filename == "y.wav"
    # editing flips category and the row is now flagged edited
    store.update(con, "/x/y.wav", {"category": "snare"})
    assert [r.category for r in store.query(con, edited=True)] == ["snare"]


def test_store_delete(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    store.update(con, "/x/y.wav", {"category": "kick"})
    assert store.delete(con, "/x/y.wav") is True
    assert store.delete(con, "/x/y.wav") is False
    assert store.get(con, "/x/y.wav") is None


def test_get_fresh_edited_survives_but_rescan_path_overrides(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    store.upsert(con, "/p.wav", 100.0, Tags(filename="p.wav", kind="one-shot", category="kick", backend="heuristic"))
    # not edited + changed mtime -> miss (would re-classify)
    assert store.get_fresh(con, "/p.wav", 200.0, "heuristic") is None
    # mark edited; now even a changed mtime is served (edits survive normal scans)
    store.update(con, "/p.wav", {"category": "snare"})
    hit = store.get_fresh(con, "/p.wav", 200.0, "heuristic")
    assert hit is not None and hit.category == "snare"


def test_top_level_api_uses_configured_db(tmp_path, monkeypatch):
    db = tmp_path / "tags.db"
    monkeypatch.setenv("TIMBRE_DB", str(db))
    monkeypatch.setenv("TIMBRE_CONFIG", str(tmp_path / "config.toml"))
    f = tmp_path / "kick_90.wav"
    _wav(f)
    timbre.update(str(f), {"category": "clap"})
    assert timbre.get(str(f)).category == "clap"
    assert [t.category for t in timbre.query(category="clap")] == ["clap"]
    assert timbre.delete(str(f)) is True
    assert timbre.get(str(f)) is None


def test_config_env_and_set(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.toml"
    monkeypatch.setenv("TIMBRE_CONFIG", str(cfgfile))
    monkeypatch.delenv("TIMBRE_DB", raising=False)
    monkeypatch.delenv("TIMBRE_DB_ENABLED", raising=False)
    # defaults
    assert config.db_enabled() is True
    # persisted set
    config.set_value("db.enabled", "false")
    assert config.db_enabled() is False
    config.set_value("db.path", str(tmp_path / "custom.db"))
    assert config.db_path() == tmp_path / "custom.db"
    # env overrides file
    monkeypatch.setenv("TIMBRE_DB_ENABLED", "1")
    assert config.db_enabled() is True

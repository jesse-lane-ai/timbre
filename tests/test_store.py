"""Tests for the optional sqlite persistence layer and `scan --db` caching."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from timbre import store
from timbre.api import Tags
from timbre.cli import cli


def _wav(path) -> None:
    sr = 44100
    t = np.linspace(0, 0.2, int(sr * 0.2), False)
    x = (np.sin(2 * np.pi * 120 * t) * np.exp(-t * 20)).astype("float32")
    sf.write(str(path), x, sr)


def test_upsert_and_get_fresh_roundtrip(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    tags = Tags(filename="k.wav", kind="one-shot", category="kick", instruments=["kick", "drums"], bpm=120.0)
    store.upsert(con, "/abs/k.wav", 111.0, tags)
    con.commit()

    # same mtime + backend -> hit
    hit = store.get_fresh(con, "/abs/k.wav", 111.0, "heuristic")
    assert hit is not None and hit.category == "kick" and hit.instruments == ["kick", "drums"]
    # changed mtime -> miss
    assert store.get_fresh(con, "/abs/k.wav", 222.0, "heuristic") is None
    # different backend -> miss
    assert store.get_fresh(con, "/abs/k.wav", 111.0, "ace-step") is None
    # unknown path -> miss
    assert store.get_fresh(con, "/abs/other.wav", 111.0, "heuristic") is None


def test_genres_roundtrip_and_filter(tmp_path):
    con = store.open_db(tmp_path / "db.sqlite")
    house = Tags(filename="loop.wav", kind="loop", category="drum",
                 genres=[{"genre": "house", "score": 0.82}, {"genre": "techno", "score": 0.21}])
    plain = Tags(filename="hit.wav", kind="one-shot", category="kick")
    store.upsert(con, "/abs/loop.wav", 1.0, house)
    store.upsert(con, "/abs/hit.wav", 1.0, plain)
    con.commit()

    got = store.get(con, "/abs/loop.wav")
    assert got.genres == [{"genre": "house", "score": 0.82}, {"genre": "techno", "score": 0.21}]
    assert store.get(con, "/abs/hit.wav").genres == []

    # genre filter matches the JSON-stored list
    paths = [t.filename for t in store.query(con, genre="house")]
    assert paths == ["loop.wav"]
    assert store.query(con, genre="dubstep") == []


def test_genres_column_migration(tmp_path):
    # An older DB without the genres column still opens and reads back empty.
    import sqlite3

    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(str(p))
    con.execute("CREATE TABLE tags (path TEXT PRIMARY KEY, filename TEXT, kind TEXT, "
                "category TEXT, instruments TEXT, key TEXT, scale TEXT, bpm REAL, "
                "duration REAL, confidence REAL, caption TEXT, backend TEXT, "
                "mtime REAL, scanned_at REAL)")
    con.execute("INSERT INTO tags (path, filename, kind, category) VALUES "
                "('/abs/x.wav', 'x.wav', 'loop', 'drum')")
    con.commit()
    con.close()

    con = store.open_db(p)  # runs the migration
    row = store.get(con, "/abs/x.wav")
    assert row is not None and row.genres == []
    # and new writes carry genres on the migrated DB
    store.update(con, "/abs/x.wav", {"genres": [{"genre": "techno", "score": 0.9}]})
    assert store.get(con, "/abs/x.wav").genres == [{"genre": "techno", "score": 0.9}]


def test_scan_db_caches_unchanged_files(tmp_path):
    folder = tmp_path / "kicks"
    folder.mkdir()
    for i in range(3):
        _wav(folder / f"kick_{i}.wav")
    db = str(tmp_path / "tags.db")
    runner = CliRunner()

    r1 = runner.invoke(cli, ["scan", str(folder), "--db", db])
    assert r1.exit_code == 0
    assert "3 classified, 0 from cache" in r1.output

    r2 = runner.invoke(cli, ["scan", str(folder), "--db", db])
    assert r2.exit_code == 0
    assert "0 classified, 3 from cache" in r2.output

    r3 = runner.invoke(cli, ["scan", str(folder), "--db", db, "--rescan"])
    assert "3 classified, 0 from cache" in r3.output

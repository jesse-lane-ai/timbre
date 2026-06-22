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

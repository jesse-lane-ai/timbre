"""Optional sqlite persistence for scans — a cache, not a source of truth.

timbre's classifiers stay stateless; this module is the opt-in storage layer the
CLI's ``--db`` flag uses. Rows are keyed by absolute path and carry the file's
mtime + the backend used, so a re-scan skips files that haven't changed and were
last classified with the same backend (switching backends re-runs them).

The table mirrors the :class:`timbre.api.Tags` fields. ``instruments`` is stored
comma-joined; everything else maps directly.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .api import Tags

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tags (
    path        TEXT PRIMARY KEY,
    filename    TEXT,
    kind        TEXT,
    category    TEXT,
    instruments TEXT,
    key         TEXT,
    scale       TEXT,
    bpm         REAL,
    duration    REAL,
    confidence  REAL,
    caption     TEXT,
    backend     TEXT,
    mtime       REAL,
    scanned_at  REAL
);
"""


def open_db(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    con.commit()
    return con


def get_fresh(con: sqlite3.Connection, abspath: str, mtime: float | None, backend: str) -> Tags | None:
    """Return the cached Tags for ``abspath`` iff it's still valid — same mtime
    and same backend — else None (caller should (re)classify)."""
    row = con.execute("SELECT * FROM tags WHERE path = ?", (abspath,)).fetchone()
    if row is None or mtime is None:
        return None
    if row["backend"] != backend or row["mtime"] != mtime:
        return None
    return _row_to_tags(row)


def upsert(con: sqlite3.Connection, abspath: str, mtime: float | None, tags: Tags) -> None:
    con.execute(
        """INSERT INTO tags
           (path, filename, kind, category, instruments, key, scale, bpm,
            duration, confidence, caption, backend, mtime, scanned_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(path) DO UPDATE SET
            filename=excluded.filename, kind=excluded.kind, category=excluded.category,
            instruments=excluded.instruments, key=excluded.key, scale=excluded.scale,
            bpm=excluded.bpm, duration=excluded.duration, confidence=excluded.confidence,
            caption=excluded.caption, backend=excluded.backend, mtime=excluded.mtime,
            scanned_at=excluded.scanned_at""",
        (
            abspath, tags.filename, tags.kind, tags.category,
            ",".join(tags.instruments), tags.key, tags.scale, tags.bpm,
            tags.duration, tags.confidence, tags.caption, tags.backend,
            mtime, time.time(),
        ),
    )


def _row_to_tags(row: sqlite3.Row) -> Tags:
    return Tags(
        filename=row["filename"],
        kind=row["kind"],
        category=row["category"],
        instruments=[t for t in (row["instruments"] or "").split(",") if t],
        key=row["key"],
        scale=row["scale"],
        bpm=row["bpm"],
        duration=row["duration"],
        confidence=row["confidence"],
        caption=row["caption"],
        backend=row["backend"],
        path=row["path"],
    )

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
    scanned_at  REAL,
    edited      INTEGER DEFAULT 0
);
"""

# Columns a caller may write via update() — path/mtime/scanned_at/edited are managed.
_EDITABLE = ("filename", "kind", "category", "instruments", "key", "scale", "bpm", "duration", "confidence", "caption", "backend")
_ORDERABLE = {"path", "filename", "category", "kind", "bpm", "confidence", "scanned_at"}


def open_db(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    # Migrate older DBs that predate the `edited` column.
    cols = {r["name"] for r in con.execute("PRAGMA table_info(tags)")}
    if "edited" not in cols:
        con.execute("ALTER TABLE tags ADD COLUMN edited INTEGER DEFAULT 0")
    con.commit()
    return con


def get_fresh(con: sqlite3.Connection, abspath: str, mtime: float | None, backend: str) -> Tags | None:
    """Return the cached Tags for ``abspath`` iff it should NOT be re-classified:
    either it was manually edited (edits survive normal scans; only --rescan
    overwrites), or its mtime and backend are unchanged. Else None."""
    row = con.execute("SELECT * FROM tags WHERE path = ?", (abspath,)).fetchone()
    if row is None:
        return None
    if row["edited"]:
        return _row_to_tags(row)
    if mtime is None or row["backend"] != backend or row["mtime"] != mtime:
        return None
    return _row_to_tags(row)


def get(con: sqlite3.Connection, abspath: str) -> Tags | None:
    row = con.execute("SELECT * FROM tags WHERE path = ?", (abspath,)).fetchone()
    return _row_to_tags(row) if row is not None else None


def query(
    con: sqlite3.Connection,
    *,
    category: str | None = None,
    kind: str | None = None,
    key: str | None = None,
    scale: str | None = None,
    backend: str | None = None,
    instrument: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    path_like: str | None = None,
    edited: bool | None = None,
    order: str = "path",
    limit: int | None = None,
) -> list[Tags]:
    """Filtered read over the DB. All filters are ANDed; None means 'any'."""
    clauses: list[str] = []
    params: list[object] = []
    for col, val in (("category", category), ("kind", kind), ("key", key), ("scale", scale), ("backend", backend)):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    if instrument is not None:
        clauses.append("(',' || instruments || ',') LIKE ?")
        params.append(f"%,{instrument},%")
    if bpm_min is not None:
        clauses.append("bpm >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        clauses.append("bpm <= ?")
        params.append(bpm_max)
    if path_like is not None:
        clauses.append("path LIKE ?")
        params.append(f"%{path_like}%")
    if edited is not None:
        clauses.append("edited = ?")
        params.append(1 if edited else 0)

    if order not in _ORDERABLE:
        from .errors import BadInputError

        raise BadInputError(f"cannot order by '{order}' (allowed: {', '.join(sorted(_ORDERABLE))})")

    sql = "SELECT * FROM tags"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {order}"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [_row_to_tags(r) for r in con.execute(sql, params)]


def update(con: sqlite3.Connection, abspath: str, fields: dict[str, object]) -> Tags:
    """Create-or-update a row's tag fields (marks it edited=1 so it survives
    normal re-scans). ``fields`` keys must be in :data:`_EDITABLE`;
    ``instruments`` accepts a list or a comma-joined string."""
    from .errors import BadInputError

    if not fields:
        raise BadInputError("no fields to update")
    clean: dict[str, object] = {}
    for k, v in fields.items():
        if k not in _EDITABLE:
            raise BadInputError(f"field '{k}' is not editable (editable: {', '.join(_EDITABLE)})")
        if k == "instruments" and isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        clean[k] = v

    existing = con.execute("SELECT 1 FROM tags WHERE path = ?", (abspath,)).fetchone()
    if existing is None:
        # default filename to the basename if the caller didn't set one
        clean.setdefault("filename", Path(abspath).name)
        cols = ["path", "edited", "scanned_at", *clean.keys()]
        vals = [abspath, 1, time.time(), *clean.values()]
        con.execute(f"INSERT INTO tags ({', '.join(cols)}) VALUES ({', '.join('?' * len(vals))})", vals)
    else:
        sets = ", ".join(f"{k} = ?" for k in clean)
        con.execute(
            f"UPDATE tags SET {sets}, edited = 1, scanned_at = ? WHERE path = ?",
            [*clean.values(), time.time(), abspath],
        )
    con.commit()
    return get(con, abspath)  # type: ignore[return-value]


def delete(con: sqlite3.Connection, abspath: str) -> bool:
    cur = con.execute("DELETE FROM tags WHERE path = ?", (abspath,))
    con.commit()
    return cur.rowcount > 0


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

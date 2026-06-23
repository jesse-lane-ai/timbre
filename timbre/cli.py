"""``timbre`` CLI — thin wrapper over :mod:`timbre.api`.

Single-shot, scriptable, structured output. Every command takes ``--json`` and
prints a ``{"ok": true, "data": ...}`` / ``{"ok": false, "error": ...}`` envelope.

    timbre probe kick.wav
    timbre probe kick.wav --backend ace-step --json
    timbre scan ./packs --backend heuristic --json
    timbre backends
    timbre serve --port 8765
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import classify, classify_many, list_backends
from .errors import TimbreError

_AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}


def _emit(data, human: str, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps({"ok": True, "data": data}))
    else:
        click.echo(human)


def _fail(err: TimbreError, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps({"ok": False, "error": err.message, "code": err.code}))
    else:
        click.echo(f"error: {err.message}", err=True)
    sys.exit(err.code)


json_option = click.option("--json", "as_json", is_flag=True, default=False, help="Emit a JSON envelope.")
backend_option = click.option(
    "--backend", default="heuristic", show_default=True, help="Recognizer backend (see `timbre backends`)."
)


@click.group()
@click.version_option(package_name="timbre")
def cli():
    """timbre — classify audio samples by content and filename."""


@cli.command("backends")
@json_option
def backends(as_json: bool):
    """List available recognizer backends."""
    names = list_backends()
    _emit(names, "available backends:\n  " + "\n  ".join(names), as_json)


@cli.command("probe")
@click.argument("path")
@backend_option
@json_option
def probe(path: str, backend: str, as_json: bool):
    """Classify a single audio file at PATH."""
    try:
        tags = classify(path, backend=backend)
    except TimbreError as e:
        _fail(e, as_json)
        return
    d = tags.to_dict()
    human = "\n".join(
        [
            f"file:        {tags.filename}",
            f"kind:        {tags.kind}",
            f"category:    {tags.category or '—'}",
            f"instruments: {', '.join(tags.instruments) or '—'}",
            f"key:         {tags.key or '—'}  scale: {tags.scale or '—'}",
            f"bpm:         {tags.bpm or '—'}",
            f"duration:    {tags.duration or '—'}",
            f"confidence:  {tags.confidence}",
            f"backend:     {tags.backend}",
        ]
        + ([f"caption:     {tags.caption}"] if tags.caption else [])
    )
    _emit(d, human, as_json)


@cli.command("scan")
@click.argument("folder")
@backend_option
@click.option("--db", "db_path", default=None, help="Override the DB path for this scan (default: configured DB).")
@click.option("--no-db", is_flag=True, default=False, help="Don't persist this scan (DB is on by default).")
@click.option("--rescan", is_flag=True, default=False, help="Re-classify every file, ignoring cached rows.")
@json_option
def scan(folder: str, backend: str, db_path: str | None, no_db: bool, rescan: bool, as_json: bool):
    """Recursively classify every audio file under FOLDER.

    Results persist to the configured sqlite DB by default; on a later scan,
    files whose mtime is unchanged (and were last classified with the same
    backend) are served from the cache instead of re-analyzed. --rescan forces a
    full re-run; --no-db disables persistence for this scan.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        _fail(TimbreError(f"not a directory: {root}"), as_json)
        return
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in _AUDIO_EXT and not p.name.startswith("._")
    )

    from . import config, store

    con = None
    effective_db: str | None = None
    if not no_db:
        effective_db = db_path or (str(config.db_path()) if config.db_enabled() else None)
        if effective_db:
            con = store.open_db(effective_db)

    try:
        results: dict[str, object] = {}  # abspath -> Tags
        to_classify: list[Path] = []
        cached_n = 0
        for p in files:
            ap = str(p.resolve())
            if con is not None and not rescan:
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = None
                hit = store.get_fresh(con, ap, mtime, backend)
                if hit is not None:
                    results[ap] = hit
                    cached_n += 1
                    continue
            to_classify.append(p)

        if to_classify:
            fresh = classify_many([str(p) for p in to_classify], backend=backend)
            for p, t in zip(to_classify, fresh):
                ap = str(p.resolve())
                results[ap] = t
                if con is not None:
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        mtime = None
                    store.upsert(con, ap, mtime, t)
            if con is not None:
                con.commit()
    except TimbreError as e:
        _fail(e, as_json)
        return
    finally:
        if con is not None:
            con.close()

    tags = [results[str(p.resolve())] for p in files]
    data = [t.to_dict() for t in tags]
    human = "\n".join(f"{t.category or '?':<10} {t.kind:<9} {t.filename}" for t in tags) or "(no audio files)"
    if effective_db:
        human += f"\n\n{len(to_classify)} classified, {cached_n} from cache → {effective_db}"
    _emit(data, human, as_json)


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
def serve(host: str, port: int):
    """Run the HTTP classification server (POST /classify)."""
    from .server import run

    run(host, port)


db_path_option = click.option("--db", "db_path", default=None, help="DB path (default: configured DB).")


def _tags_table(tags) -> str:
    if not tags:
        return "(no matching entries)"
    return "\n".join(
        f"{(t.category or '?'):<10} {t.kind:<9} {str(t.bpm or '—'):<6} {t.filename}" for t in tags
    )


@cli.group("db")
def db():
    """Read and write the persistent tag store."""


@db.command("find")
@click.option("--category")
@click.option("--kind")
@click.option("--key")
@click.option("--scale")
@click.option("--backend")
@click.option("--instrument", help="Match an entry carrying this instrument tag.")
@click.option("--bpm-min", type=float)
@click.option("--bpm-max", type=float)
@click.option("--path", "path_like", help="Substring match on the stored file path.")
@click.option("--edited/--not-edited", "edited", default=None, help="Only manually-edited (or not) entries.")
@click.option("--order", default="path", show_default=True)
@click.option("--limit", type=int)
@db_path_option
@json_option
def db_find(category, kind, key, scale, backend, instrument, bpm_min, bpm_max, path_like, edited, order, limit, db_path, as_json):
    """Query the store with filters (all ANDed)."""
    from . import query as _query

    try:
        tags = _query(
            db=db_path, category=category, kind=kind, key=key, scale=scale, backend=backend,
            instrument=instrument, bpm_min=bpm_min, bpm_max=bpm_max, path_like=path_like,
            edited=edited, order=order, limit=limit,
        )
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit([t.to_dict() for t in tags], _tags_table(tags), as_json)


@db.command("get")
@click.argument("path")
@db_path_option
@json_option
def db_get(path, db_path, as_json):
    """Fetch one stored entry by file PATH."""
    from . import get as _get

    tags = _get(path, db=db_path)
    if tags is None:
        _fail(TimbreError(f"no stored entry for: {path}"), as_json)
        return
    _emit(tags.to_dict(), _tags_table([tags]), as_json)


@db.command("set")
@click.argument("path")
@click.option("--category")
@click.option("--kind")
@click.option("--key")
@click.option("--scale")
@click.option("--bpm", type=float)
@click.option("--instruments", help="Comma-separated instrument tags (replaces existing).")
@click.option("--caption")
@click.option("--backend")
@db_path_option
@json_option
def db_set(path, category, kind, key, scale, bpm, instruments, caption, backend, db_path, as_json):
    """Create or update an entry's tags (marks it edited; survives re-scans)."""
    from . import update as _update

    fields = {}
    for name, val in (("category", category), ("kind", kind), ("key", key), ("scale", scale),
                      ("bpm", bpm), ("caption", caption), ("backend", backend)):
        if val is not None:
            fields[name] = val
    if instruments is not None:
        fields["instruments"] = [s.strip() for s in instruments.split(",") if s.strip()]
    if not fields:
        _fail(TimbreError("nothing to set — pass at least one field (e.g. --category kick)"), as_json)
        return
    try:
        tags = _update(path, fields, db=db_path)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit(tags.to_dict(), _tags_table([tags]), as_json)


@db.command("rm")
@click.argument("path")
@db_path_option
@json_option
def db_rm(path, db_path, as_json):
    """Delete a stored entry by file PATH."""
    from . import delete as _delete

    removed = _delete(path, db=db_path)
    if not removed:
        _fail(TimbreError(f"no stored entry for: {path}"), as_json)
        return
    _emit({"deleted": path}, f"deleted {path}", as_json)


@cli.group("config")
def config_grp():
    """Show or change timbre's configuration."""


@config_grp.command("show")
@json_option
def config_show(as_json):
    """Show the effective configuration (env + file + defaults)."""
    from . import config as cfg

    eff = cfg.effective()
    human = "\n".join(
        ["[db]", f"  enabled = {eff['db']['enabled']}", f"  path    = {eff['db']['path']}", "", f"config file: {cfg.config_path()}"]
    )
    _emit(eff, human, as_json)


@config_grp.command("path")
def config_path_cmd():
    """Print the config file path."""
    from . import config as cfg

    click.echo(str(cfg.config_path()))


@config_grp.command("set")
@click.argument("key")
@click.argument("value")
@json_option
def config_set(key, value, as_json):
    """Set a config KEY (e.g. db.enabled, db.path) to VALUE."""
    from . import config as cfg

    try:
        stored = cfg.set_value(key, value)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit({key: stored}, f"{key} = {stored}", as_json)


if __name__ == "__main__":
    cli()

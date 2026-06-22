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
@click.option("--db", "db_path", default=None, help="Persist results to a sqlite DB and skip unchanged files.")
@click.option("--rescan", is_flag=True, default=False, help="Re-classify every file, ignoring cached rows.")
@json_option
def scan(folder: str, backend: str, db_path: str | None, rescan: bool, as_json: bool):
    """Recursively classify every audio file under FOLDER.

    With --db, results are written to a sqlite DB; on a later scan, files whose
    mtime is unchanged (and were last classified with the same backend) are
    served from the cache instead of re-analyzed. --rescan forces a full re-run.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        _fail(TimbreError(f"not a directory: {root}"), as_json)
        return
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_EXT)

    con = None
    if db_path:
        from . import store

        con = store.open_db(db_path)

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
    if db_path:
        human += f"\n\n{len(to_classify)} classified, {cached_n} from cache → {db_path}"
    _emit(data, human, as_json)


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
def serve(host: str, port: int):
    """Run the HTTP classification server (POST /classify)."""
    from .server import run

    run(host, port)


if __name__ == "__main__":
    cli()

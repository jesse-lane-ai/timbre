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
    genres_str = ", ".join("{} ({})".format(g["genre"], g["score"]) for g in tags.genres)
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
        + ([f"genres:      {genres_str}"] if tags.genres else [])
        + ([f"caption:     {tags.caption}"] if tags.caption else [])
    )
    _emit(d, human, as_json)


# Categories the heuristic falls back to when it can't actually place a file:
# the one-shot ("fx") and loop ("full") catch-all buckets. A file in one of
# these with no instruments is effectively unclassified, so it's worth the model.
_CATCHALL_CATEGORIES = {"fx", "full"}

# Pitched / melodic / full-context categories the audio heuristic is unreliable
# at — its spectral features are tuned for percussion, so a *guessed* (not
# name-confirmed) verdict here is worth a model second opinion. Drum categories
# (kick/snare/hat/clap/perc/...) are deliberately excluded: the heuristic nails
# those, so they stay local and fast.
_PITCHED_CATEGORIES = {
    # bass family + the single coarse melodic/tonal bucket + vocal. (All the
    # former pitched roles — lead/pad/pluck/piano/guitar/… — now collapse into
    # `melodic`, so naming them here is no longer necessary.)
    "bass", "sub", "808", "reese",
    "melodic", "vocal",
}


def _needs_escalation(tags) -> bool:
    """A file the primary backend couldn't reliably place — worth spending the
    heavier escalation model on. True when, with no instrument tags to go on:
      * there's no category or only a catch-all bucket ("fx"/"full"), or
      * the category is a pitched/melodic one the audio heuristic merely
        *guessed* — i.e. the filename itself yields no category, so the verdict
        came from the (percussion-tuned, unreliable-on-pitched) audio pass
        rather than a name cue we'd trust.
    """
    if getattr(tags, "instruments", None):
        return False
    category = getattr(tags, "category", None)
    if not category or category in _CATCHALL_CATEGORIES:
        return True
    if category in _PITCHED_CATEGORIES:
        # Trust a name-derived pitched category; escalate an audio-guessed one.
        # The name pass is cheap (no audio) and the fuser lets a name category
        # win, so "the name yields a category" == "this verdict is name-sourced".
        from .names import classify_from_names

        ni = classify_from_names(getattr(tags, "path", None) or getattr(tags, "filename", "") or "")
        return ni.get("category") is None
    return False


@cli.command("scan")
@click.argument("folder")
@click.option("--backend", default="auto", show_default=True,
              help="Recognizer backend, or 'auto' (default): heuristic plus "
                   "whichever of clap/ace-step are installed, unioning their "
                   "verdicts. See `timbre backends`.")
@click.option("--escalate", default=None, metavar="BACKEND",
              help="Progressive scan: after the primary backend, re-classify only the "
                   "files it left with no category/instruments using this heavier backend "
                   "(e.g. --escalate ace-step).")
@click.option("--db", "db_path", default=None, help="Override the DB path for this scan (default: configured DB).")
@click.option("--no-db", is_flag=True, default=False, help="Don't persist this scan (DB is on by default).")
@click.option("--rescan", is_flag=True, default=False, help="Re-classify every file, ignoring cached rows.")
@json_option
def scan(folder: str, backend: str, escalate: str | None, db_path: str | None,
         no_db: bool, rescan: bool, as_json: bool):
    """Recursively classify every audio file under FOLDER.

    Results persist to the configured sqlite DB by default; on a later scan,
    files whose mtime is unchanged (and were last classified with the same
    backend) are served from the cache instead of re-analyzed. --rescan forces a
    full re-run; --no-db disables persistence for this scan.

    With --escalate the scan is progressive: every file is classified by the
    primary --backend first (cheap), then only those it couldn't place (no
    category and no instruments) are re-run through the heavier escalation
    backend. The expensive model is spent solely on the gaps. Both tiers' rows
    are cached, so a re-scan re-runs neither.
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

    # A cached row is fresh for a progressive scan if it came from either tier.
    accept = {backend, escalate} if escalate else None

    con = None
    effective_db: str | None = None
    if not no_db:
        effective_db = db_path or (str(config.db_path()) if config.db_enabled() else None)
        if effective_db:
            con = store.open_db(effective_db)

    import os

    quiet = as_json or os.environ.get("MENDELL_QUIET") == "1"

    def _mtime(p: Path):
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    def _classify(paths: list[Path], be: str, label: str) -> list:
        """Run a backend over ``paths`` with per-file stderr progress."""
        if not paths:
            return []
        if not quiet:
            click.echo(f"  {label} ({len(paths)} files, backend={be})", err=True)
        n_todo = len(paths)
        done = [0]

        def _progress(item, _rec):
            if quiet:
                return
            done[0] += 1
            name = getattr(item, "filename", None) or getattr(item, "path", "?")
            click.echo(f"    [{done[0]}/{n_todo}] {name}", err=True)

        return classify_many([str(p) for p in paths], backend=be, on_result=_progress)

    try:
        results: dict[str, object] = {}  # abspath -> Tags
        to_classify: list[Path] = []
        to_escalate: list[Path] = []  # cached-by-primary-but-unplaced → escalate only
        cached_n = 0
        for p in files:
            ap = str(p.resolve())
            if con is not None and not rescan:
                hit = store.get_fresh(con, ap, _mtime(p), backend, accept_backends=accept)
                if hit is not None:
                    # An unplaced row still on the primary backend hasn't been
                    # escalated yet — send it straight to the model (its cheap
                    # heuristic pass is already done), but only escalate primary
                    # rows so files the model itself gave up on aren't re-run.
                    if escalate and _needs_escalation(hit) and hit.backend == backend:
                        results[ap] = hit
                        to_escalate.append(p)
                    else:
                        results[ap] = hit
                        cached_n += 1
                    continue
            to_classify.append(p)

        if not quiet:
            plan = f"{len(files)} audio files ({cached_n} cached, {len(to_classify)} to classify"
            plan += f", backend={backend}"
            if escalate:
                plan += f", escalate={escalate}"
            click.echo(f"scanning {root}: {plan})", err=True)

        # --- tier 1: primary backend ---
        for p, t in zip(to_classify, _classify(to_classify, backend, "tier 1")):
            ap = str(p.resolve())
            results[ap] = t
            if escalate and _needs_escalation(t):
                to_escalate.append(p)
            elif con is not None:
                store.upsert(con, ap, _mtime(p), t)
        if con is not None:
            con.commit()  # persist tier 1 before the slower, failable tier 2

        # --- tier 2: escalate only the files the primary couldn't place ---
        if escalate and to_escalate:
            esc = _classify(to_escalate, escalate, f"tier 2 (escalate {escalate})")
            for p, t in zip(to_escalate, esc):
                ap = str(p.resolve())
                results[ap] = t
                if con is not None:
                    store.upsert(con, ap, _mtime(p), t)

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
        summary = f"{len(to_classify)} classified, {cached_n} from cache"
        if escalate:
            summary += f", {len(to_escalate)} escalated to {escalate}"
        human += f"\n\n{summary} → {effective_db}"
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
@click.option("--genre", help="Match an entry carrying this genre tag.")
@click.option("--bpm-min", type=float)
@click.option("--bpm-max", type=float)
@click.option("--path", "path_like", help="Substring match on the stored file path.")
@click.option("--edited/--not-edited", "edited", default=None, help="Only manually-edited (or not) entries.")
@click.option("--collection", help="Only entries that belong to this collection.")
@click.option("--order", default="path", show_default=True)
@click.option("--limit", type=int)
@db_path_option
@json_option
def db_find(category, kind, key, scale, backend, instrument, genre, bpm_min, bpm_max, path_like, edited, collection, order, limit, db_path, as_json):
    """Query the store with filters (all ANDed)."""
    from . import query as _query

    try:
        tags = _query(
            db=db_path, category=category, kind=kind, key=key, scale=scale, backend=backend,
            instrument=instrument, genre=genre, bpm_min=bpm_min, bpm_max=bpm_max, path_like=path_like,
            edited=edited, collection=collection, order=order, limit=limit,
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


@cli.group("collection")
def collection_grp():
    """Create and manage named collections of samples."""


@collection_grp.command("list")
@db_path_option
@json_option
def collection_list(db_path, as_json):
    """List collections with member counts."""
    from . import collections as _collections

    cols = _collections(db=db_path)
    human = "\n".join(f"{c['name']}  ({c['count']})" for c in cols) or "(no collections)"
    _emit(cols, human, as_json)


@collection_grp.command("new")
@click.argument("name")
@db_path_option
@json_option
def collection_new(name, db_path, as_json):
    """Create a collection (idempotent)."""
    from . import collection_create as _create

    try:
        c = _create(name, db=db_path)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit(c, f"created collection {c['name']}", as_json)


@collection_grp.command("rename")
@click.argument("name")
@click.argument("new_name")
@db_path_option
@json_option
def collection_rename_cmd(name, new_name, db_path, as_json):
    """Rename collection NAME to NEW_NAME."""
    from . import collection_rename as _rename

    try:
        c = _rename(name, new_name, db=db_path)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit(c, f"renamed {name} → {c['name']}", as_json)


@collection_grp.command("rm")
@click.argument("name")
@db_path_option
@json_option
def collection_rm(name, db_path, as_json):
    """Delete a collection (its memberships go too; samples are untouched)."""
    from . import collection_delete as _del

    if not _del(name, db=db_path):
        _fail(TimbreError(f"no such collection: {name}", code=2), as_json)
        return
    _emit({"deleted": name}, f"deleted collection {name}", as_json)


@collection_grp.command("add")
@click.argument("name")
@click.argument("paths", nargs=-1, required=True)
@db_path_option
@json_option
def collection_add_cmd(name, paths, db_path, as_json):
    """Add one or more sample PATHS to collection NAME (creates it if needed)."""
    from . import collection_add as _add

    try:
        count = _add(name, list(paths), db=db_path)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit({"collection": name, "count": count}, f"{name} now has {count} members", as_json)


@collection_grp.command("remove")
@click.argument("name")
@click.argument("paths", nargs=-1, required=True)
@db_path_option
@json_option
def collection_remove_cmd(name, paths, db_path, as_json):
    """Remove one or more sample PATHS from collection NAME."""
    from . import collection_remove as _remove

    try:
        count = _remove(name, list(paths), db=db_path)
    except TimbreError as e:
        _fail(e, as_json)
        return
    _emit({"collection": name, "count": count}, f"{name} now has {count} members", as_json)


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

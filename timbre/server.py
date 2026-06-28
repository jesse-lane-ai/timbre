"""Thin HTTP front-end over :mod:`timbre.api` — the surface browser apps use.

Stdlib-only (no framework dep) so it ships with the base install. CORS is wide
open (``*``) because the intended client is a local single-file browser app
(e.g. a drag-and-drop sample looper) served from ``file://`` or another origin.

Endpoints::

    GET    /backends                     -> ["ace-step","clap","heuristic"]
    GET    /health                       -> {"status": "ok"}
    POST   /classify?backend=heuristic   -> {...Tags...}   (stateless; does not persist)
           body: raw audio bytes (e.g. fetch(url, {method:'POST', body: file}))
           optional header: X-Filename: kick_01.wav   (name hint for the name pass)

  Persistent store (read/write the configured DB) — for library managers:

    GET    /tags?category=kick&bpm_min=80&limit=50   -> [ {...Tags...}, ... ]
    GET    /tag?path=/abs/kick.wav                    -> {...Tags...}
    POST   /tag    body: {"path": "...", "category": "kick", "bpm": 90, ...}
           -> {...Tags...}   (create-or-update; marks the entry edited)
    DELETE /tag?path=/abs/kick.wav                    -> {"deleted": "..."}

  Collections (named groups of samples):

    GET    /collections                       -> [ {"name","count",...}, ... ]
    POST   /collections   body: {"name": "drums"}                 -> the collection
    POST   /collections/add     body: {"collection": "drums", "paths": [...]}
    POST   /collections/remove  body: {"collection": "drums", "paths": [...]}
           -> {"collection": "drums", "count": <remaining members>}
    DELETE /collections?name=drums            -> {"deleted": "drums"}
    GET    /tags?collection=drums             -> only that collection's members

All responses use the ``{"ok": bool, "data"|"error": ...}`` envelope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import (
    classify,
    collection_add,
    collection_create,
    collection_delete,
    collection_remove,
    collection_rename,
    collections,
    config,
    delete,
    get,
    list_backends,
    query,
    store,
    update,
)
from .errors import TimbreError
from .recognize.types import (
    GENRE_VOCAB,
    INSTRUMENT_VOCAB,
    KINDS,
    LOOP_CATEGORIES,
    ONESHOT_CATEGORIES,
    RECORDING_CATEGORIES,
)

_MAX_BYTES = 64 * 1024 * 1024  # 64 MB upload cap
_UI_DIR = Path(__file__).parent / "ui"


def _import_bytes(data: bytes, filename: str, backend: str):
    """Classify uploaded audio and persist both the bytes and the verdict.

    Browser uploads carry no real filesystem path, so we cache the bytes
    content-addressed under :func:`config.blob_dir` and make that the entry's
    canonical ``path`` — that way the UI's ``/audio`` preview can read it back.
    Identical content re-imports to the same row (hash-keyed)."""
    ext = Path(filename).suffix.lower() or ".wav"
    digest = hashlib.sha1(data).hexdigest()
    blobs = config.blob_dir()
    blobs.mkdir(parents=True, exist_ok=True)
    blob = blobs / f"{digest}{ext}"
    if not blob.exists():
        blob.write_bytes(data)

    tags = classify(data, backend=backend, filename=filename)
    tags = replace(tags, path=str(blob), filename=Path(filename).name)
    con = store.open_db(str(config.db_path()))
    try:
        store.upsert(con, str(blob), blob.stat().st_mtime, tags)
        con.commit()
        return store.get(con, str(blob))
    finally:
        con.close()


def _vocab() -> dict:
    """Taxonomy the library-manager UI uses to populate its edit dropdowns."""
    by_kind = {
        "one-shot": list(ONESHOT_CATEGORIES),
        "loop": list(LOOP_CATEGORIES),
        "recording": list(RECORDING_CATEGORIES),
        "unknown": list(ONESHOT_CATEGORIES),
    }
    all_categories = sorted(set().union(*(set(v) for v in by_kind.values())))
    return {
        "kinds": list(KINDS),
        "categories_by_kind": by_kind,
        "all_categories": all_categories,
        "instruments": list(INSTRUMENT_VOCAB),
        "genres": list(GENRE_VOCAB),
    }


# Documented HTTP surface — the single source of truth for the route reference.
# Each entry is (method, path, summary). The dispatch in do_GET/do_POST/do_DELETE
# is hand-written (stdlib handler), so a test (tests/test_http_routes.py) asserts
# this table and the actual dispatch stay in sync. `docs/reference.md` is
# generated from here; edit routes here, then run `python scripts/gen_docs.py`.
ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/", "library-manager web app (HTML); also served at /ui and /index.html"),
    ("GET", "/backends", "available backend names"),
    ("GET", "/vocab", "taxonomy: kinds, per-kind categories, instruments, genres"),
    ("GET", "/health", '{"status": "ok"} liveness probe'),
    ("GET", "/audio?path=…", "a stored sample's raw audio bytes (UI player; store files only)"),
    ("GET", "/tags?category=…&bpm_min=…&limit=…", "filtered list of stored Tags"),
    ("GET", "/tag?path=…", "one stored Tags entry"),
    ("GET", "/collections", "list collections with member counts"),
    ("POST", "/classify?backend=…", "classify the posted audio body → Tags (stateless, not persisted); name hint via X-Filename header"),
    ("POST", "/import?backend=…", "classify + cache the bytes + persist → stored Tags (path points at the cached blob)"),
    ("POST", "/tag", 'create-or-update a stored entry, marks it edited; body: {"path": …, "category": …, …}'),
    ("POST", "/collections", 'create a collection; body: {"name": "drums"}'),
    ("POST", "/collections/add", 'add members; body: {"collection": "drums", "paths": [...]}'),
    ("POST", "/collections/remove", 'remove members; body: {"collection": "drums", "paths": [...]}'),
    ("POST", "/collections/rename", 'rename; body: {"name": "drums", "new_name": "percussion"}'),
    ("DELETE", "/tag?path=…", "delete a stored entry"),
    ("DELETE", "/collections?name=…", "delete a collection (not the samples)"),
)

# Query params that map straight to store.query kwargs, with their coercions.
_FILTER_COERCE = {
    "category": str, "kind": str, "key": str, "scale": str, "backend": str,
    "instrument": str, "genre": str, "path_like": str, "order": str, "collection": str,
    "bpm_min": float, "bpm_max": float, "limit": int,
}


def _filters_from_qs(qs: dict) -> dict:
    out: dict = {}
    for k, coerce in _FILTER_COERCE.items():
        if k in qs:
            out[k] = coerce(qs[k][0])
    if "edited" in qs:
        out["edited"] = qs["edited"][0].lower() not in {"0", "false", "no", ""}
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "timbre"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_ui(self) -> None:
        try:
            body = (_UI_DIR / "index.html").read_bytes()
        except OSError:
            self._send(404, {"ok": False, "error": "library-manager UI not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    _AUDIO_CTYPE = {
        ".wav": "audio/wav", ".aif": "audio/aiff", ".aiff": "audio/aiff",
        ".flac": "audio/flac", ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
    }

    def _send_audio(self, p: str | None) -> None:
        """Stream a sample's bytes so the UI can render/play its waveform.

        Only files that have a row in the store are served — this is a local
        tool, but we still don't want it to be an arbitrary-file-read endpoint."""
        if not p:
            self._send(400, {"ok": False, "error": "missing ?path="})
            return
        if get(p) is None:
            self._send(404, {"ok": False, "error": f"no stored entry for: {p}"})
            return
        fp = Path(p).expanduser()
        if not fp.is_file():
            self._send(404, {"ok": False, "error": f"file not found on disk: {p}"})
            return
        try:
            body = fp.read_bytes()
        except OSError as e:
            self._send(500, {"ok": False, "error": f"could not read {p}: {e}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", self._AUDIO_CTYPE.get(fp.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "none")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path in ("/", "/ui", "/index.html"):
            self._send_ui()
        elif path == "/backends":
            self._send(200, {"ok": True, "data": list_backends()})
        elif path == "/vocab":
            self._send(200, {"ok": True, "data": _vocab()})
        elif path == "/health":
            self._send(200, {"ok": True, "data": {"status": "ok"}})
        elif path == "/collections":
            self._guard(collections)
        elif path == "/audio":
            self._send_audio(qs.get("path", [None])[0])
        elif path == "/tags":
            self._guard(lambda: query(**_filters_from_qs(qs)), transform=lambda r: [t.to_dict() for t in r])
        elif path == "/tag":
            p = qs.get("path", [None])[0]
            if not p:
                self._send(400, {"ok": False, "error": "missing ?path="})
                return

            def _do():
                t = get(p)
                if t is None:
                    raise TimbreError(f"no stored entry for: {p}", code=2)
                return t

            self._guard(_do, transform=lambda t: t.to_dict())
        else:
            self._send(404, {"ok": False, "error": f"no such endpoint: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/classify":
            self._handle_classify(qs)
        elif parsed.path == "/import":
            self._handle_import(qs)
        elif parsed.path == "/tag":
            self._handle_tag_write()
        elif parsed.path == "/collections":
            self._handle_collection_create()
        elif parsed.path == "/collections/add":
            self._handle_collection_members(collection_add)
        elif parsed.path == "/collections/remove":
            self._handle_collection_members(collection_remove)
        elif parsed.path == "/collections/rename":
            self._handle_collection_rename()
        else:
            self._send(404, {"ok": False, "error": f"no such endpoint: {parsed.path}"})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/collections":
            name = qs.get("name", [None])[0]
            if not name:
                self._send(400, {"ok": False, "error": "missing ?name="})
                return

            def _do_coll():
                if not collection_delete(name):
                    raise TimbreError(f"no such collection: {name}", code=2)
                return {"deleted": name}

            self._guard(_do_coll)
            return
        if parsed.path != "/tag":
            self._send(404, {"ok": False, "error": f"no such endpoint: {parsed.path}"})
            return
        p = qs.get("path", [None])[0]
        if not p:
            self._send(400, {"ok": False, "error": "missing ?path="})
            return

        def _do():
            if not delete(p):
                raise TimbreError(f"no stored entry for: {p}", code=2)
            return {"deleted": p}

        self._guard(_do)

    def _handle_classify(self, qs: dict) -> None:
        backend = qs.get("backend", ["heuristic"])[0]
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._send(400, {"ok": False, "error": "empty body — POST the audio file as the request body"})
            return
        if length > _MAX_BYTES:
            self._send(413, {"ok": False, "error": f"upload exceeds {_MAX_BYTES} bytes"})
            return
        data = self.rfile.read(length)
        filename = self.headers.get("X-Filename") or "upload.wav"
        self._guard(lambda: classify(data, backend=backend, filename=filename), transform=lambda t: t.to_dict())

    def _handle_import(self, qs: dict) -> None:
        backend = qs.get("backend", ["heuristic"])[0]
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._send(400, {"ok": False, "error": "empty body — POST the audio file as the request body"})
            return
        if length > _MAX_BYTES:
            self._send(413, {"ok": False, "error": f"upload exceeds {_MAX_BYTES} bytes"})
            return
        data = self.rfile.read(length)
        filename = self.headers.get("X-Filename") or "upload.wav"
        self._guard(lambda: _import_bytes(data, filename, backend), transform=lambda t: t.to_dict())

    def _handle_tag_write(self) -> None:
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return
        p = payload.get("path")
        if not p:
            self._send(400, {"ok": False, "error": "JSON body must include a 'path'"})
            return
        fields = {k: v for k, v in payload.items() if k != "path"}
        self._guard(lambda: update(p, fields), transform=lambda t: t.to_dict())

    def _handle_collection_create(self) -> None:
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return
        name = payload.get("name")
        if not name:
            self._send(400, {"ok": False, "error": "JSON body must include a 'name'"})
            return
        self._guard(lambda: collection_create(name))

    def _handle_collection_rename(self) -> None:
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return
        name, new_name = payload.get("name"), payload.get("new_name")
        if not name or not new_name:
            self._send(400, {"ok": False, "error": "JSON body must include 'name' and 'new_name'"})
            return
        self._guard(lambda: collection_rename(name, new_name))

    def _handle_collection_members(self, fn) -> None:
        """Shared handler for /collections/add and /collections/remove.

        Body: {"collection": "name", "paths": ["/abs/a.wav", ...]}. Returns the
        collection's resulting member count."""
        try:
            payload = json.loads(self._read_body() or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return
        name = payload.get("collection")
        paths = payload.get("paths")
        if not name:
            self._send(400, {"ok": False, "error": "JSON body must include a 'collection'"})
            return
        if not isinstance(paths, list) or not paths:
            self._send(400, {"ok": False, "error": "JSON body must include a non-empty 'paths' list"})
            return
        self._guard(lambda: {"collection": name, "count": fn(name, paths)})

    def _guard(self, fn, transform=lambda x: x) -> None:
        """Run a store op, mapping TimbreError codes to HTTP status and any
        other exception to 500, all in the standard envelope."""
        try:
            result = fn()
        except TimbreError as e:
            status = {1: 400, 2: 404}.get(e.code, 500)
            self._send(status, {"ok": False, "error": e.message, "code": e.code})
            return
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(e)})
            return
        self._send(200, {"ok": True, "data": transform(result)})

    def log_message(self, fmt, *args):  # quieter default logging
        pass


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"timbre serve → http://{host}:{port}  (library manager UI at /  ·  API: /classify /tags /tag /collections /vocab)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

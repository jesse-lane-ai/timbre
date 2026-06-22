"""Thin HTTP front-end over :mod:`timbre.api` — the surface browser apps use.

Stdlib-only (no framework dep) so it ships with the base install. CORS is wide
open (``*``) because the intended client is a local single-file browser app
(e.g. a drag-and-drop sample looper) served from ``file://`` or another origin.

Endpoints::

    GET  /backends                     -> {"ok": true, "data": ["ace-step","clap","heuristic"]}
    GET  /health                       -> {"ok": true, "data": {"status": "ok"}}
    POST /classify?backend=heuristic   -> {"ok": true, "data": {...Tags...}}
         body: raw audio bytes (e.g. fetch(url, {method:'POST', body: file}))
         optional header: X-Filename: kick_01.wav   (name hint for the name pass)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import classify, list_backends
from .errors import TimbreError

_MAX_BYTES = 64 * 1024 * 1024  # 64 MB upload cap


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

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/backends":
            self._send(200, {"ok": True, "data": list_backends()})
        elif path == "/health":
            self._send(200, {"ok": True, "data": {"status": "ok"}})
        else:
            self._send(404, {"ok": False, "error": f"no such endpoint: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/classify":
            self._send(404, {"ok": False, "error": f"no such endpoint: {parsed.path}"})
            return
        qs = parse_qs(parsed.query)
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
        try:
            tags = classify(data, backend=backend, filename=filename)
        except TimbreError as e:
            self._send(400 if e.code == 1 else 500, {"ok": False, "error": e.message, "code": e.code})
            return
        except Exception as e:  # noqa: BLE001 — surface unexpected failures as 500
            self._send(500, {"ok": False, "error": str(e)})
            return
        self._send(200, {"ok": True, "data": tags.to_dict()})

    def log_message(self, fmt, *args):  # quieter default logging
        pass


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"timbre serve → http://{host}:{port}  (POST /classify, GET /backends)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

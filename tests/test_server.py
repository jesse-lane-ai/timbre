"""HTTP surface smoke tests: the UI route, /vocab, and a tag round-trip.

Boots the real ThreadingHTTPServer on an ephemeral port against a temp DB so the
library-manager front-end has something concrete to talk to.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMBRE_DB", str(tmp_path / "tags.db"))
    monkeypatch.setenv("TIMBRE_DB_ENABLED", "1")
    # import after env is set so config picks up the temp DB
    from timbre.server import _Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, json.loads(r.read())


def _req(base, path, method, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def test_ui_served_at_root(server):
    with urllib.request.urlopen(server + "/") as r:
        assert r.status == 200
        assert r.headers["Content-Type"].startswith("text/html")
        html = r.read().decode()
    assert "library manager" in html
    assert "/tags" in html  # the app talks to the store endpoint


def test_vocab_shape(server):
    status, body = _get(server, "/vocab")
    assert status == 200 and body["ok"]
    data = body["data"]
    assert "one-shot" in data["kinds"] and "recording" in data["kinds"]
    assert data["categories_by_kind"]["loop"]
    assert "kick" in data["all_categories"]
    assert data["instruments"]


def test_health(server):
    status, body = _get(server, "/health")
    assert status == 200 and body["data"]["status"] == "ok"


def test_tag_write_read_delete_round_trip(server):
    p = "/abs/example/kick_01.wav"
    # create-or-update
    _, body = _req(server, "/tag", "POST", {"path": p, "category": "kick", "kind": "one-shot",
                                            "instruments": ["kick"], "bpm": 90})
    assert body["ok"] and body["data"]["category"] == "kick" and body["data"]["edited"]
    # shows up in a filtered listing
    _, body = _get(server, "/tags?category=kick")
    assert any(t["path"].endswith("kick_01.wav") for t in body["data"])
    # single fetch
    _, body = _req(server, "/tag?path=" + urllib.parse.quote(p), "GET")
    assert body["data"]["bpm"] == 90
    # delete
    _, body = _req(server, "/tag?path=" + urllib.parse.quote(p), "DELETE")
    assert body["ok"] and body["data"]["deleted"] == p
    # gone -> 404
    with pytest.raises(urllib.error.HTTPError) as ei:
        _req(server, "/tag?path=" + urllib.parse.quote(p), "GET")
    assert ei.value.code == 404

"""HTTP surface smoke tests: the UI route, /vocab, and a tag round-trip.

Boots the real ThreadingHTTPServer on an ephemeral port against a temp DB so the
library-manager front-end has something concrete to talk to.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
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


def test_audio_served_only_for_stored_files(server, tmp_path):
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "snare.wav"
    sf.write(str(wav), (np.random.randn(2205) * 0.2).astype("float32"), 22050)
    raw = wav.read_bytes()
    q = "?path=" + urllib.parse.quote(str(wav))

    # not in the store yet -> 404, no arbitrary file read
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(server + "/audio" + q)
    assert ei.value.code == 404

    # register it, then it streams byte-for-byte
    _req(server, "/tag", "POST", {"path": str(wav), "category": "snare", "kind": "one-shot"})
    with urllib.request.urlopen(server + "/audio" + q) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "audio/wav"
        assert r.read() == raw


def test_import_caches_bytes_so_preview_works(server, tmp_path):
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, (np.random.randn(2205) * 0.3).astype("float32"), 22050, format="WAV")
    raw = buf.getvalue()

    req = urllib.request.Request(
        server + "/import?backend=heuristic", data=raw, method="POST",
        headers={"X-Filename": "clap_77.wav"},
    )
    with urllib.request.urlopen(req) as r:
        saved = json.loads(r.read())["data"]
    # canonical path is a real on-disk blob, not the bare filename
    assert saved["filename"] == "clap_77.wav"
    assert saved["path"] != "clap_77.wav" and Path(saved["path"]).is_file()

    # the preview endpoint can now read it back byte-for-byte
    with urllib.request.urlopen(server + "/audio?path=" + urllib.parse.quote(saved["path"])) as r:
        assert r.status == 200 and r.read() == raw

    # identical content re-imports to the same row (hash-keyed)
    with urllib.request.urlopen(urllib.request.Request(
        server + "/import", data=raw, method="POST", headers={"X-Filename": "clap_77.wav"})) as r:
        again = json.loads(r.read())["data"]
    assert again["path"] == saved["path"]
    _, listing = _get(server, "/tags")
    assert sum(1 for t in listing["data"] if t["path"] == saved["path"]) == 1


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


def test_collections_round_trip(server):
    # seed two tag rows to put into a collection
    a, b = "/abs/x/a.wav", "/abs/x/b.wav"
    for p in (a, b):
        _req(server, "/tag", "POST", {"path": p, "category": "kick", "kind": "one-shot"})

    # create a collection
    _, body = _req(server, "/collections", "POST", {"name": "drums"})
    assert body["ok"] and body["data"]["name"] == "drums"

    # add both samples
    _, body = _req(server, "/collections/add", "POST", {"collection": "drums", "paths": [a, b]})
    assert body["ok"] and body["data"]["count"] == 2

    # it shows up in the listing with the right count
    _, body = _get(server, "/collections")
    assert any(c["name"] == "drums" and c["count"] == 2 for c in body["data"])

    # filtering tags by collection returns only its members
    _, body = _get(server, "/tags?collection=drums")
    paths = {t["path"] for t in body["data"]}
    assert paths == {a, b}

    # remove one
    _, body = _req(server, "/collections/remove", "POST", {"collection": "drums", "paths": [a]})
    assert body["data"]["count"] == 1
    _, body = _get(server, "/tags?collection=drums")
    assert {t["path"] for t in body["data"]} == {b}

    # delete the collection (samples remain in the store)
    _, body = _req(server, "/collections?name=drums", "DELETE")
    assert body["ok"] and body["data"]["deleted"] == "drums"
    _, body = _get(server, "/collections")
    assert not any(c["name"] == "drums" for c in body["data"])
    _, body = _req(server, "/tag?path=" + urllib.parse.quote(b), "GET")
    assert body["ok"]  # the sample itself survived

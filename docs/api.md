# timbre — API & HTTP integration guide

For people **building against timbre** (browser apps, library managers, services
like Mendell). For install + everyday CLI use see [`README.md`](../README.md); for
the exhaustive auto-generated vocab/CLI/HTTP tables see
[`reference.md`](./reference.md); for architecture see [`CLAUDE.md`](../CLAUDE.md).

Everything funnels through one core (`timbre.api`); the CLI, HTTP server, and
Python callers are thin wrappers over it. A classification result is a `Tags`
object — see the [`Tags` schema](./reference.md#tags-schema) for the exact fields.

## Python API

```python
import timbre

tags = timbre.classify("kick_01.wav")                       # -> Tags
tags = timbre.classify(raw_bytes, filename="x.wav",         # bytes work too
                       backend="ace-step")
many = timbre.classify_many(paths, backend="heuristic")     # batch (amortizes model load)
```

Reading and writing the persistent store:

```python
timbre.query(category="kick", bpm_min=80)        # -> [Tags, ...]
timbre.get("/abs/kick.wav")                       # -> Tags | None
timbre.update("/abs/kick.wav", category="snare", bpm=92)   # marks the entry edited
timbre.delete("/abs/kick.wav")
```

Collections (named groups of samples; a sample can be in many):

```python
timbre.collection_create("drums")
timbre.collection_add("drums", ["/abs/kick.wav"])
timbre.collections()                        # [{"name","count",...}, ...]
timbre.query(collection="drums")
```

## HTTP server

For browser clients (e.g. a drag-and-drop sample app) that can't import Python or
shell out:

```sh
timbre serve --port 8765
```

```js
// POST the File straight as the request body
const res = await fetch("http://127.0.0.1:8765/classify?backend=heuristic", {
  method: "POST",
  headers: { "X-Filename": file.name },
  body: file,
});
const { ok, data } = await res.json();
```

CORS is wide open (`*`) — the server is meant to be run locally next to the app.
`timbre serve` also hosts the built-in library-manager UI at `/` (see
[README → Library manager UI](../README.md#library-manager-ui)).

### Endpoints

The full route table is generated from code in
[`reference.md` → HTTP API](./reference.md#http-api). The most-used ones:

| Method | Endpoint | Returns |
|---|---|---|
| `POST` | `/classify?backend=…` | a `Tags` object (audio file as raw body; stateless, not persisted) |
| `POST` | `/import?backend=…` | classify + cache the bytes + persist; returns the stored `Tags` |
| `GET` | `/tags?category=kick&bpm_min=80&limit=50` | filtered list of stored `Tags` |
| `GET` · `POST` · `DELETE` | `/tag` | read / create-or-update / delete one stored entry |
| `GET` | `/vocab` | taxonomy: kinds, per-kind categories, instruments |
| `GET` | `/backends` | available backend names |
| `GET` · `POST` · `DELETE` | `/collections` (+ `/collections/add`, `/remove`, `/rename`) | manage named collections |

Every JSON response is enveloped: `{"ok": true, "data": …}` or
`{"ok": false, "error": …}`.

You can also query the sqlite store directly:

```sh
sqlite3 ~/.local/share/timbre/tags.db "SELECT category, count(*) FROM tags GROUP BY category"
```

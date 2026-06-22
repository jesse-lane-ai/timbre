# timbre

Content + filename based audio sample classifier. Given an audio file, timbre
tells you what it is — `kind` (loop / one-shot), a coarse `category`
(kick/snare/bass/melodic/…), `instruments`, and name-derived `key`/`scale`/`bpm`.

It fuses two passes:

1. **name pass** — parses the filename and parent folders, no audio loaded.
2. **content pass** — a pluggable recognizer backend listens to the audio:
   - `heuristic` — local, deterministic, zero extra deps (spectral features).
   - `clap` — local zero-shot audio-text embedding (`pip install 'timbre[clap]'`).
   - `ace-step` — local ACE-Step captioner (`pip install 'timbre[ace-step]'`).

timbre exposes the same core through three surfaces.

## Python API

```python
import timbre

tags = timbre.classify("kick_01.wav")                       # -> Tags
tags = timbre.classify(raw_bytes, filename="x.wav",         # bytes work too
                       backend="ace-step")
many = timbre.classify_many(paths, backend="heuristic")     # batch (amortizes model load)
```

## CLI

```sh
timbre probe kick.wav                  # human-readable
timbre probe kick.wav --json           # {"ok": true, "data": {...}}
timbre scan ./packs --backend heuristic --json
timbre backends
```

Every command takes `--json` and returns `{"ok": true, "data": ...}` or
`{"ok": false, "error": ...}`.

### Persistent scans (`--db`)

The classifiers are stateless, but `scan` can persist results to a sqlite DB and
skip files that haven't changed since the last scan:

```sh
timbre scan ./packs --db tags.db          # first run: classifies everything
timbre scan ./packs --db tags.db          # later: unchanged files served from cache
timbre scan ./packs --db tags.db --rescan # force a full re-classification
```

Cache validity is keyed on file **mtime** and the **backend** used — touch a file
or switch backends and it's re-classified. The `tags` table mirrors the `Tags`
fields, so you can query it directly:

```sh
sqlite3 tags.db "SELECT category, count(*) FROM tags GROUP BY category"
```

## HTTP server

For browser clients (e.g. a drag-and-drop sample app) that can't import Python
or shell out:

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

| Method | Endpoint | Returns |
|---|---|---|
| `GET` | `/backends` | available backend names |
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/classify?backend=…` | a `Tags` object (audio file as raw body) |

## Install

```sh
pip install -e .                 # base (heuristic backend)
pip install -e '.[clap]'         # + CLAP
pip install -e '.[ace-step]'     # + ACE-Step captioner
```

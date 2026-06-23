# timbre

[![CI](https://github.com/jesse-lane-ai/timbre/actions/workflows/ci.yml/badge.svg)](https://github.com/jesse-lane-ai/timbre/actions/workflows/ci.yml)

Content + filename based audio sample classifier. Given an audio file, timbre
tells you what it is — `kind` (one-shot / loop / recording), a coarse `category`
(kick/snare/bass/lead/riser/…), `instruments`, and name-derived `key`/`scale`/`bpm`.

A **recording** is long-form audio that's neither a single hit nor a seamless
loop — a field recording, jam, take, voice memo, or full mix. It's detected from
name cues (`jam`, `take`, `field rec`, `voice memo`, `full mix`, …) or, for
unnamed files, from duration. Category vocabularies are per-kind; see
`timbre/recognize/types.py`.

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

### Persistent store (on by default)

The classifiers are stateless, but `scan` **persists results by default** to a
sqlite DB at the XDG data location, and skips files that haven't changed since
the last scan:

```sh
timbre scan ./packs                 # classifies + persists to the configured DB
timbre scan ./packs                 # later: unchanged files served from cache
timbre scan ./packs --rescan        # force a full re-classification
timbre scan ./packs --no-db         # don't persist this scan
timbre scan ./packs --db other.db   # override the DB path for this scan
```

Cache validity is keyed on file **mtime** and the **backend** used — touch a file
or switch backends and it's re-classified.

**Config** (env var > config file > default):

```sh
timbre config show                       # effective settings + config file path
timbre config set db.enabled false       # turn persistence off by default
timbre config set db.path ~/my-tags.db   # change the default DB location
```

Env overrides: `TIMBRE_DB`, `TIMBRE_DB_ENABLED`, `TIMBRE_CONFIG`.

### Reading and writing tags

The store is queryable and editable from all three surfaces. Manual writes mark
an entry **edited** — those survive normal re-scans (only `--rescan` overwrites
them), so an external library manager can correct tags without losing them.

**CLI:**

```sh
timbre db find --category kick --bpm-min 80 --bpm-max 100 --limit 20
timbre db find --instrument snare --json
timbre db get  /abs/path/kick.wav
timbre db set  /abs/path/kick.wav --category snare --instruments "snare,clap"
timbre db rm   /abs/path/kick.wav
```

**Python:**

```python
import timbre
timbre.query(category="kick", bpm_min=80)        # -> [Tags, ...]
timbre.get("/abs/kick.wav")                       # -> Tags | None
timbre.update("/abs/kick.wav", category="snare", bpm=92)
timbre.delete("/abs/kick.wav")
```

**HTTP** (`timbre serve`):

| Method | Endpoint | Body / query |
|---|---|---|
| `GET` | `/tags?category=kick&bpm_min=80&limit=50` | filters |
| `GET` | `/tag?path=/abs/kick.wav` | one entry |
| `POST` | `/tag` | `{"path": "...", "category": "snare", "instruments": ["snare"]}` |
| `DELETE` | `/tag?path=/abs/kick.wav` | — |

You can still query the DB directly too:

```sh
sqlite3 ~/.local/share/timbre/tags.db "SELECT category, count(*) FROM tags GROUP BY category"
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
pip install -e .                       # base (heuristic backend)
pip install -e '.[clap]'               # + CLAP
pip install -e '.[ace-step]'           # + ACE-Step captioner
pip install -e '.[ace-step,quant]'     # + 8/4-bit quantization (CUDA only)
```

### ACE-Step quantization

The ACE-Step captioner is large (~22 GB in `full` mode). On an NVIDIA GPU you can
quantize it in-flight via the `ACESTEP_CAPTIONER_LOAD` env var:

| Mode | VRAM | Notes |
|---|---|---|
| `full` (default) | ~22 GB | fp16/bf16 |
| `8bit` | ~11 GB | bitsandbytes `load_in_8bit` |
| `4bit` | ~6–7 GB | bitsandbytes `nf4`, fp16 compute |

```sh
ACESTEP_CAPTIONER_LOAD=4bit timbre scan ./packs --backend ace-step
```

Quantization applies only to the language-model layers and is **CUDA-only** — it
needs the `quant` extra (`bitsandbytes` + `accelerate`).

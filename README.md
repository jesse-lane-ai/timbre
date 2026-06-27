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

> **Which doc do I want?**
> - **Installing or using the CLI?** You're in the right place — read on.
> - **Building against timbre** (HTTP / Python API, browser apps)? → [`docs/api.md`](docs/api.md)
> - **Exact vocab / CLI flags / HTTP routes?** → [`docs/reference.md`](docs/reference.md) (auto-generated from code)
> - **Hacking on internals?** → [`CLAUDE.md`](CLAUDE.md)

## Python API

```python
import timbre

tags = timbre.classify("kick_01.wav")                       # -> Tags
tags = timbre.classify(raw_bytes, filename="x.wav",         # bytes work too
                       backend="ace-step")
many = timbre.classify_many(paths, backend="heuristic")     # batch (amortizes model load)
```

Querying/editing the store and collections from Python, plus the full HTTP
contract for browser clients, live in **[`docs/api.md`](docs/api.md)**.

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
or switch backends and it's re-classified. (So name/caption-parsing changes only
take effect on already-scanned files once you `--rescan` them.)

### Progressive scan (`--escalate`)

Spend the expensive model only where the cheap pass falls short. With
`--escalate`, every file is first classified by the primary `--backend` (cheap,
e.g. `heuristic`), then **only the files it couldn't place** — no category *and*
no instruments — are re-run through the heavier escalation backend:

```sh
timbre scan ./packs --backend heuristic --escalate ace-step
```

Both tiers' verdicts are cached, so a re-scan re-runs neither. A file the
escalation backend itself gave up on is not retried on the next pass; only
unplaced rows still on the primary backend get escalated.

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

The same operations are exposed over HTTP (`/tags`, `/tag`) for browser clients —
see **[`docs/api.md`](docs/api.md)**.

**Collections** group samples by name (a sample can be in many). They're
available from all three surfaces:

```sh
timbre collection new drums
timbre collection add drums /abs/kick.wav /abs/snare.wav
timbre collection list
timbre db find --collection drums          # filter the store to a collection
timbre collection rename drums percussion
timbre collection remove drums /abs/kick.wav
timbre collection rm drums                 # deletes the collection, not the samples
```

```python
timbre.collection_create("drums")
timbre.collection_add("drums", ["/abs/kick.wav"])
timbre.collections()                        # [{"name","count",...}, ...]
timbre.query(collection="drums")
```

Collections are also exposed over HTTP (`/collections` + `/add`/`/remove`/
`/rename`) — see **[`docs/api.md`](docs/api.md)**. You can query the DB directly
too:

```sh
sqlite3 ~/.local/share/timbre/tags.db "SELECT category, count(*) FROM tags GROUP BY category"
```

## Library manager UI

`timbre serve` also hosts a built-in, zero-build library manager at the server
root — open it in a browser to browse, filter, edit, and delete tags in the
persistent store:

```sh
timbre serve --port 8765        # then open http://127.0.0.1:8765/
```

It's a single static HTML file (no framework, no build step) that talks to the
same HTTP API documented below. You can:

- **filter** the store by kind, category, instrument, key/scale, BPM range, path
  substring, backend, and edited-state;
- **edit** any entry's tags inline (kind/category/instruments/key/scale/bpm and
  the free-text **caption**) in the side panel — dropdowns are populated from the
  live taxonomy via `/vocab`, and saves mark the entry **edited** so it survives
  re-scans. Captions also show as their own table column;
- **organize** samples into **collections** — a left sidebar lists every
  collection with its count; click one to browse it, hover to rename or delete
  it, and use the toolbar to add/remove the selected rows as a group;
- **select** rows (per-row checkbox or the header select-all) and **copy their
  metadata** to the clipboard as JSON, CSV, or just paths;
- **delete** entries;
- **import** audio via the OS file picker — *Import files…* (multi-select) or
  *Import folder…* (recursive) — to classify and add them to the library in one
  step, with a progress bar;
- **preview** any entry: every row has an inline play button + waveform
  thumbnail, and clicking a row slides open a larger player (play/pause, scrub by
  clicking the waveform). Thumbnails decode lazily as they scroll into view.

Browsers don't expose a picked file's real filesystem path, so imported audio is
cached server-side (content-addressed, under `<data-dir>/blobs/`) and that cached
copy becomes the entry's path — which is what makes the waveform preview work for
imports. Entries created by `timbre scan` keep their real on-disk paths and
preview directly. (Identical content imported twice dedupes to one entry.)

Manually-edited rows are flagged with a ✎ in the list.

## HTTP server

`timbre serve` exposes the classifier and store over HTTP for browser clients
(e.g. a drag-and-drop sample app) that can't import Python or shell out:

```sh
timbre serve --port 8765
```

The JS quickstart, CORS notes, and the full endpoint table are in
**[`docs/api.md`](docs/api.md)** (and the generated route table in
[`docs/reference.md`](docs/reference.md)).

## Install

> **New to Python virtual environments? Follow these steps in order.** They set up
> an isolated environment so `timbre` doesn't interfere with other Python tools on
> your machine. Run every command from the project's top-level `timbre/` folder (the
> one containing `pyproject.toml`).

**1. Move into the project folder:**

```sh
cd /path/to/timbre        # the folder with pyproject.toml in it
```

**2. Create a virtual environment** (a private Python just for this project).
This makes a `.venv/` folder:

```sh
python3 -m venv .venv
```

**3. Activate it.** Do this every time you open a new terminal to work on timbre:

```sh
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows PowerShell
```

Your prompt should now start with `(.venv)`.

**4. Install timbre into the environment.** Pick the one line that matches what you
need — the base install is enough to get started:

```sh
pip install -e .                       # base (heuristic backend) — start here
pip install -e '.[clap]'               # + CLAP
pip install -e '.[ace-step]'           # + ACE-Step captioner
pip install -e '.[ace-step,quant]'     # + 8/4-bit quantization (CUDA only)
```

**5. Check it worked:**

```sh
timbre --help
timbre serve                           # start the HTTP server + UI
```

> **Common mistakes**
>
> - `timbre: command not found` — your virtual environment isn't active. Re-run the
>   `source .venv/bin/activate` step (step 3). The `(.venv)` prefix must be visible.
> - **Moved or renamed the project folder?** A virtual environment can't be moved.
>   Delete it and start over from step 2: `rm -rf .venv`, then recreate.
> - **Use straight quotes, not backticks:** write `'.[ace-step]'` with single
>   quotes. Backticks (`` ` ``) tell the shell to *run* the text as a command and
>   will fail.
> - The argument is `.` (a dot, meaning "this folder"), **not** `timbre`. Use
>   `pip install -e '.[ace-step]'`, not `pip install timbre[ace-step]`.

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

#### Picking a mode for your GPU

Quantization's job here is **fitting the model on a smaller card, not speeding up
a big one.** The weights are stored in 4/8-bit but dequantized back to fp16 for
every matmul, so on a GPU that already fits `full` the quantized modes are
typically *slightly slower* per token. They only win on speed when `full` would
otherwise offload to CPU or OOM.

| Free VRAM | Recommended | Why |
|---|---|---|
| ≥ ~24 GB | `full` | Fastest, best caption quality (no dequant overhead). |
| ~12–16 GB | `8bit` | `full` would be tight/offload; 8bit fits with minimal quality loss. |
| ~8–12 GB | `4bit` | Often the only mode that fits; faster than `full` here only because `full` can't run. |
| < 8 GB | use `clap` or `heuristic` | The captioner won't fit even at 4-bit. |

Caption quality degrades slightly with `4bit` (nf4), which can matter since
timbre keyword-maps the caption to a category — prefer the highest mode that fits.

For **throughput**, the bigger lever than quantization is batch size: the
captioner has a large fixed per-call cost, so raising `ACESTEP_CAPTIONER_BATCH`
(default 8) amortizes it across more files per `generate()` call.

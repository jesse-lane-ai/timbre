<!--
  GENERATED FILE — do not edit by hand.
  Source: scripts/gen_docs.py (imports the vocab, CLI, and HTTP route tables).
  Regenerate with:  python scripts/gen_docs.py
  CI fails if this file is out of date (python scripts/gen_docs.py --check).
-->

# timbre reference

Auto-generated from code. For tutorials and recipes see [`README.md`](../README.md);
for architecture and internals see [`CLAUDE.md`](../CLAUDE.md).

## Vocabulary

**Kinds:** `one-shot`, `loop`, `recording`, `unknown`

**Categories** (per kind — the valid `category` values):

| Kind | Categories |
|---|---|
| `one-shot` | `kick`, `snare`, `clap`, `snap`, `hat`, `tom`, `crash`, `ride`, `rim`, `perc`, `bass`, `808`, `sub`, `reese`, `melodic`, `vocal`, `fx`, `riser`, `sweep`, `impact`, `drone`, `texture`, `ambience`, `foley`, `noise` |
| `loop` | `drum`, `perc`, `bass`, `sub`, `808`, `melodic`, `vocal`, `fx`, `riser`, `texture`, `ambience`, `foley`, `full` |
| `recording` | `full`, `vocal`, `instrument`, `drum`, `melodic`, `ambience`, `field`, `foley`, `fx` |

**Instruments** (shared across all kinds): `kick`, `snare`, `clap`, `snap`, `rimshot`, `hihat`, `open hat`, `crash`, `ride`, `cymbal`, `tom`, `conga`, `bongo`, `shaker`, `tambourine`, `cowbell`, `woodblock`, `clave`, `triangle`, `djembe`, `timbale`, `agogo`, `cabasa`, `drums`, `percussion`, `bass`, `808`, `sub`, `reese`, `lead`, `pad`, `pluck`, `arp`, `stab`, `synth`, `keys`, `piano`, `organ`, `bell`, `guitar`, `strings`, `violin`, `cello`, `brass`, `trumpet`, `sax`, `flute`, `choir`, `koto`, `guzheng`, `zither`, `sitar`, `harp`, `banjo`, `ukulele`, `marimba`, `kalimba`, `vibraphone`, `xylophone`, `glockenspiel`, `harpsichord`, `accordion`, `vocal`, `singing`, `spoken`, `fx`, `sound design`, `noise`

**Genres** (scored, ranked; populated by the `clap`/`ace-step` backends only): `house`, `deep house`, `tech house`, `techno`, `trance`, `dubstep`, `drum and bass`, `breakbeat`, `garage`, `trap`, `lo-fi`, `synthwave`, `future bass`, `hardstyle`, `jungle`, `grime`, `electro`, `ambient`, `idm`, `downtempo`, `hip hop`, `boom bap`, `rnb`, `soul`, `funk`, `disco`, `reggae`, `dancehall`, `afrobeat`, `latin`, `pop`, `rock`, `metal`, `punk`, `jazz`, `blues`, `country`, `folk`, `gospel`, `cinematic`, `orchestral`, `classical`

## `Tags` schema

| Field | Type | Default |
|---|---|---|
| `filename` | `str` | `—` |
| `kind` | `str` | `—` |
| `category` | `str | None` | `—` |
| `instruments` | `list[str]` | `[]` |
| `genres` | `list[dict]` | `[]` |
| `key` | `str | None` | `None` |
| `scale` | `str | None` | `None` |
| `bpm` | `float | None` | `None` |
| `duration` | `float | None` | `None` |
| `confidence` | `float` | `0.0` |
| `caption` | `str | None` | `None` |
| `backend` | `str` | `'heuristic'` |
| `path` | `str | None` | `None` |
| `edited` | `bool` | `False` |

## CLI

Every command takes `--json` and returns `{"ok": true, "data": …}` or `{"ok": false, "error": …}`.

### `timbre backends`

List available recognizer backends.

- `--json` — Emit a JSON envelope.

### `timbre collection`

Create and manage named collections of samples.

#### `timbre collection add`

Add one or more sample PATHS to collection NAME (creates it if needed).

- `NAME` — positional argument
- `PATHS` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre collection list`

List collections with member counts.

- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre collection new`

Create a collection (idempotent).

- `NAME` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre collection remove`

Remove one or more sample PATHS from collection NAME.

- `NAME` — positional argument
- `PATHS` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre collection rename`

Rename collection NAME to NEW_NAME.

- `NAME` — positional argument
- `NEW_NAME` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre collection rm`

Delete a collection (its memberships go too; samples are untouched).

- `NAME` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

### `timbre config`

Show or change timbre's configuration.

#### `timbre config path`

Print the config file path.

#### `timbre config set`

Set a config KEY (e.g. db.enabled, db.path) to VALUE.

- `KEY` — positional argument
- `VALUE` — positional argument
- `--json` — Emit a JSON envelope.

#### `timbre config show`

Show the effective configuration (env + file + defaults).

- `--json` — Emit a JSON envelope.

### `timbre db`

Read and write the persistent tag store.

#### `timbre db find`

Query the store with filters (all ANDed).

- `--category`
- `--kind`
- `--key`
- `--scale`
- `--backend`
- `--instrument` — Match an entry carrying this instrument tag.
- `--genre` — Match an entry carrying this genre tag.
- `--bpm-min`
- `--bpm-max`
- `--path` — Substring match on the stored file path.
- `--edited` — Only manually-edited (or not) entries.
- `--collection` — Only entries that belong to this collection.
- `--order`
- `--limit`
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre db get`

Fetch one stored entry by file PATH.

- `PATH` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre db rm`

Delete a stored entry by file PATH.

- `PATH` — positional argument
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

#### `timbre db set`

Create or update an entry's tags (marks it edited; survives re-scans).

- `PATH` — positional argument
- `--category`
- `--kind`
- `--key`
- `--scale`
- `--bpm`
- `--instruments` — Comma-separated instrument tags (replaces existing).
- `--caption`
- `--backend`
- `--db` — DB path (default: configured DB).
- `--json` — Emit a JSON envelope.

### `timbre probe`

Classify a single audio file at PATH.

- `PATH` — positional argument
- `--backend` — Recognizer backend (see `timbre backends`).
- `--json` — Emit a JSON envelope.

### `timbre scan`

Recursively classify every audio file under FOLDER.

- `FOLDER` — positional argument
- `--backend` — Recognizer backend, or 'auto' (default): heuristic plus whichever of clap/ace-step are installed, unioning their verdicts. See `timbre backends`.
- `--escalate` — Progressive scan: after the primary backend, re-classify only the files it left with no category/instruments using this heavier backend (e.g. --escalate ace-step).
- `--db` — Override the DB path for this scan (default: configured DB).
- `--no-db` — Don't persist this scan (DB is on by default).
- `--rescan` — Re-classify every file, ignoring cached rows.
- `--json` — Emit a JSON envelope.

### `timbre serve`

Run the HTTP classification server (POST /classify).

- `--host`
- `--port`

## HTTP API

Served by `timbre serve`. CORS is wide open (`*`) — meant to run locally next to the client app.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | library-manager web app (HTML); also served at /ui and /index.html |
| `GET` | `/backends` | available backend names |
| `GET` | `/vocab` | taxonomy: kinds, per-kind categories, instruments, genres |
| `GET` | `/health` | {"status": "ok"} liveness probe |
| `GET` | `/audio?path=…` | a stored sample's raw audio bytes (UI player; store files only) |
| `GET` | `/tags?category=…&bpm_min=…&limit=…` | filtered list of stored Tags |
| `GET` | `/tag?path=…` | one stored Tags entry |
| `GET` | `/collections` | list collections with member counts |
| `POST` | `/classify?backend=…` | classify the posted audio body → Tags (stateless, not persisted); name hint via X-Filename header |
| `POST` | `/import?backend=…` | classify + cache the bytes + persist → stored Tags (path points at the cached blob) |
| `POST` | `/tag` | create-or-update a stored entry, marks it edited; body: {"path": …, "category": …, …} |
| `POST` | `/collections` | create a collection; body: {"name": "drums"} |
| `POST` | `/collections/add` | add members; body: {"collection": "drums", "paths": [...]} |
| `POST` | `/collections/remove` | remove members; body: {"collection": "drums", "paths": [...]} |
| `POST` | `/collections/rename` | rename; body: {"name": "drums", "new_name": "percussion"} |
| `DELETE` | `/tag?path=…` | delete a stored entry |
| `DELETE` | `/collections?name=…` | delete a collection (not the samples) |

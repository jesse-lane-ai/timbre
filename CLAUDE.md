# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Doc map.** This file = architecture & internals (you are here). User-facing
> install/CLI = [`README.md`](README.md). Integration (HTTP/Python API) =
> [`docs/api.md`](docs/api.md). Exhaustive vocab/CLI/HTTP tables =
> [`docs/reference.md`](docs/reference.md) — **generated**, never hand-edit it:
> run `python scripts/gen_docs.py` after changing the vocab (`recognize/types.py`),
> CLI (`cli.py`), or HTTP routes (`server.py` `ROUTES`). CI fails if it's stale.

## What this is

`timbre` is a content + filename based audio sample classifier exposed through three
surfaces (Python API, CLI, HTTP server). Given an audio file it returns a `Tags`
verdict: `kind` (one-shot / loop / recording), coarse `category` (kick/snare/bass/
lead/riser/…), `instruments`, and name-derived `key`/`scale`/`bpm`.

## Commands

The local virtualenv is `.venv` (Python 3.14). Use it directly — there is no
activation step needed for tooling:

```sh
.venv/bin/python -m pytest -q                       # full suite
.venv/bin/python -m pytest tests/test_recognize.py -q
.venv/bin/python -m pytest tests/test_recognize.py::test_list_backends -q   # single test
```

**`tests/test_name_classify.py` is NOT a pytest module** — it's a script-style
runner with a `main()`. Run it directly; pytest collects no tests from it:

```sh
.venv/bin/python tests/test_name_classify.py
```

It has one long-standing expected failure (`fx_riser_oneshot_Fm` resolves to
`riser`, the test wants `fx`) — pre-existing, unrelated to most changes.

Install with optional backends:

```sh
pip install -e .                  # heuristic backend only (no torch)
pip install -e '.[clap]'          # + clap backend
pip install -e '.[ace-step]'      # + ace-step captioner
pip install -e '.[quant]'         # + 4bit/8bit quantization for ace-step (CUDA only)
pip install -e '.[dev]'           # + pytest + pre-commit (contributor tooling)
```

After `pip install -e '.[dev]'`, run `pre-commit install` once. The hook
regenerates `docs/reference.md` when its code sources change (commit from the
activated venv). See [`docs/reference.md`](docs/reference.md) — generated, never
hand-edited.

CI (`.github/workflows`) runs `pytest -q` on Python 3.11/3.12/3.13 after
`apt-get install libsndfile1`. `tests/test_docs.py` fails the build if
`docs/reference.md` is stale or `server.ROUTES` drifts from the dispatch.

## Architecture

**One core, three surfaces.** Everything funnels through `api.classify` /
`api.classify_many` (`timbre/api.py`). The CLI (`cli.py`), HTTP server
(`server.py`), and external callers (e.g. Mendell) are thin wrappers over it —
never duplicate classification logic into a surface.

**Two-pass classification**, cheap-to-expensive:

1. **name pass** — `names.classify_from_names` parses the filename *and every
   parent folder* (no audio loaded) for kind/category/key/scale/bpm/instruments.
   This is the cheap first signal; folder names like `samples/loops/` force
   `kind=loop` on everything beneath them.
2. **content pass** — a pluggable recognizer backend listens to the audio.
   `api._resolve_kind` then reconciles the name-kind with duration/bar-alignment.

**Recognizer backends** live in `timbre/recognize/`, selected by name through
`registry.get_recognizer` (data-driven `_BACKENDS` dict). All implement the
`Recognizer` protocol in `recognize/types.py` (`recognize(items, on_result)` —
batch-first so model backends amortize load). Optional-dependency backends
lazy-import inside `__init__` and raise `BadInputError` when their dep is
missing — let that propagate, it's already actionable.

- `heuristic` — default; local, deterministic, zero extra deps. Reuses spectral/
  temporal features from `audio_analysis._AnalysisCache`. **Reliable on percussion,
  unreliable on pitched/melodic material** — this asymmetry drives the escalation
  design below.
- `clap` — local zero-shot audio-text embedding.
- `ace-step` — local ACE-Step captioner (Qwen2.5-Omni); `captioner.py` loads the
  model, `ace_step.py` maps free-text captions onto the category/instrument
  vocabularies in `recognize/types.py`.

**`category` is the coarse role; `instruments` is the specific detail.** Pitched
material collapses to a single `melodic` category (piano/guitar/lead/… live in
`instruments`); `vocal` is the category with `singing`/`spoken` as instrument
subtypes. Consequently `api.classify` drops any instrument tag equal to the final
category (a `kick` one-shot is `category=kick, instruments=["drums"]`, not
`["drums","kick"]`) — the category already states it, so the tag is redundant.

**`genres` is a scored, ranked, multi-label axis** (`GENRE_VOCAB` in
`recognize/types.py`) — a list of `{"genre", "score"}` (0..1), not a single pick.
Only the *scoring* backends populate it: `clap` gives calibrated scores
(softmax over per-genre audio·text cosine sims; `clap._rank_genres`), `ace-step`
harvests genres from its caption with occurrence-weighted pseudo-scores
(`ace_step._score_genres`) — same whole-word matcher as instruments. `heuristic`
and the name pass produce none. Genre needs a loop/recording to have signal, so
clap only scores those kinds (one-shots → empty), and `_rank_genres` has an
**absolute raw-cosine floor** (`GENRE_MIN_SIM`) as the real no-signal gate —
softmax alone is relative within a file and would manufacture a "winner" from
noise (a noise burst's top sim ~0.06 vs a confident match's ~0.25). Persisted as
a JSON column (`store._genres_to_json`); filterable via `genre=`.

**Vocabularies are the single source of truth** for valid categories/instruments:
`ONESHOT_CATEGORIES` / `LOOP_CATEGORIES` / `RECORDING_CATEGORIES` and
`INSTRUMENT_VOCAB` in `recognize/types.py`. `categories_for_kind(kind)` picks the
set. Caption→category/instrument matching (`ace_step._word_re`, `_match_vocab`,
`_score_categories`, `_VOCAB_VARIANTS`, `_INSTRUMENT_TO_CATEGORY`) is whole-word
anchored with an explicit variant map — keep that anchoring (it's what stops
`sub`→"subtle", `rim`→"primary").

**Persistence is a cache, not a source of truth** (`store.py`). `scan` writes
sqlite rows keyed on `(path, mtime, backend)`; a re-scan serves unchanged files
from cache. Because the key includes the backend, **name/caption-parsing changes
only take effect on already-scanned files after `--rescan`**. Manually-edited
rows are marked `edited=true` and survive normal re-scans (only `--rescan`
overwrites them).

**Progressive scan** (`cli.py` `scan --escalate`): classify everything with the
cheap primary backend, then re-run *only* the files it couldn't place through the
heavier backend. `cli._needs_escalation` is the gate — it escalates empty/catch-all
verdicts and audio-*guessed* pitched categories, but keeps drums and
name-confirmed verdicts local. A file the escalation backend itself gave up on is
not retried.

**Loop-context disambiguation** (`ace_step._apply_loop_context`): a narrowly-scoped
fix for short unpitched drum hits the captioner mislabels as tonal — it rebuilds
the hit as a four-on-the-floor loop and re-captions for rhythmic context.
Deliberately gated (short + unpitched + tonal-verdict only); do not broaden it to
all one-shots. Toggle with `ACESTEP_LOOP_CONTEXT=0`.

## Conventions

- **JSON envelope:** every CLI command takes `--json` and returns
  `{"ok": true, "data": ...}` or `{"ok": false, "error": ...}`.
- **ace-step defaults to 4bit load** (`ACESTEP_CAPTIONER_LOAD`). The full ~22 GB
  load maxes a 24 GB card and, notably under WSL2, silently spills to system RAM
  (~15× slower); 4bit (~7 GB) stays resident. Falls back to full precision with a
  warning if bitsandbytes/accelerate are absent.
- **`transformers` is pinned `<5`** — the 5.x image-processor registry dropped the
  `Qwen2VLImageProcessor` mapping the captioner checkpoint needs. Don't bump it.
- **Config precedence:** env var > config file > default (`config.py`). Env
  overrides: `TIMBRE_DB`, `TIMBRE_DB_ENABLED`, `TIMBRE_CONFIG`.

See `README.md` for the full CLI/API/HTTP reference.

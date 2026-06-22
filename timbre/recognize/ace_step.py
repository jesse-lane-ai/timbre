"""``ace-step`` recognizer — content recognition via ACE-Step's purpose-built
captioner model (``ACE-Step/acestep-captioner``, a Qwen2.5-Omni multimodal
model), mapped onto Mendell's taxonomy.

The captioner emits a free-text description per file; we keyword-map that
caption onto the same ``category`` / ``instruments`` vocabularies the other
recognizers emit, so the library fusion logic (``library._fuse_category``)
treats it identically.

This path needs only ``transformers`` + ``torch`` (no ACE-Step generation
checkpoint), so it's far lighter than reusing the generative engine. Like
``clap`` it's opt-in and can't be exercised in this environment (no model
download). Selecting it without ``transformers`` raises an actionable
``BadInputError`` via the captioner constructor's first use.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .types import (
    INSTRUMENT_VOCAB,
    FileProbe,
    Recognition,
    ResultSink,
    categories_for_kind,
)

NAME = "ace-step"

# A caption hit on the taxonomy is a strong but text-derived signal.
CAPTION_CONFIDENCE = 0.7
INSTRUMENT_CAP = 4


def _batch_size() -> int:
    """Files per captioner ``generate()`` call (env ``ACESTEP_CAPTIONER_BATCH``,
    default 8). Clamped to >= 1; a non-integer value falls back to the default.

    The captioner has a large fixed per-call cost (multimodal prefill) that is
    roughly batch-independent, so batching is the dominant throughput lever:
    per-file time falls ~linearly with batch size until the GPU saturates. With
    the audio-encoder padding capped to real clip length (see
    ``AceCaptioner._max_audio_seconds``) the extra VRAM per batched file is just
    a small KV cache, so a moderate default is safe; lower it if you OOM on long
    loops or a small card."""
    try:
        return max(1, int(os.environ.get("ACESTEP_CAPTIONER_BATCH", "8")))
    except ValueError:
        return 8


def _plan_batches(items: list[FileProbe], batch_size: int) -> list[list[int]]:
    """Group item indices into batches of up to ``batch_size``, sorted by
    duration (unknown = 0) so similar-length clips ride together.

    The captioner pads each batch's mel to its longest clip, so grouping by
    length keeps a long loop from inflating the padding of a batch of short
    one-shots — but only when a library is large enough to span multiple
    batches. For a folder that fits in a single batch this is a no-op, which is
    what we want: the captioner's cost is dominated by a fixed per-call overhead,
    so the *fewest* batches is fastest (the small mel-activation saving from
    tighter padding doesn't come close to paying for an extra call)."""
    order = sorted(range(len(items)), key=lambda idx: items[idx].duration or 0.0)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def _gpu_mem_mib() -> int | None:
    """Best-effort current GPU memory use (MiB) via nvidia-smi; None if no GPU
    / nvidia-smi. Cheap enough to sample per file (4 samples = 4 calls)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


class _Analytics:
    """Append-only JSONL sink for per-file recognition analytics. A no-op when
    no path is configured, so the hot path stays free unless asked for."""

    def __init__(self, path: str | None) -> None:
        self._fh = None
        if path:
            try:
                self._fh = open(path, "a", encoding="utf-8")
            except OSError:
                self._fh = None

    def emit(self, record: dict) -> None:
        if self._fh is None:
            return
        record = {"ts": round(time.time(), 3), **record}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _categories(kind: str) -> tuple[str, ...]:
    return categories_for_kind(kind)


def _word_re(term: str) -> "re.Pattern[str]":
    """A whole-word matcher for a vocab term — so 'snap' doesn't fire on
    'snappy', 'sub' on 'subtle', or 'rim' on 'primary'. Cached per term."""
    cached = _WORD_RE_CACHE.get(term)
    if cached is None:
        cached = re.compile(rf"\b{re.escape(term)}\b")
        _WORD_RE_CACHE[term] = cached
    return cached


_WORD_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _match_vocab(caption: str, vocab: tuple[str, ...]) -> list[str]:
    low = caption.lower()
    return [term for term in vocab if _word_re(term).search(low)]


def _score_categories(caption: str, vocab: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Rank the categories that appear in ``caption`` by how strongly the
    caption supports them, returning ``(term, occurrences, first_pos)`` tuples
    sorted strongest-first.

    The old behaviour took the *first vocabulary entry* found (``matched[0]``),
    which let a fixed priority order (``drum``/``perc`` precede ``bass`` for
    loops) outrank a category the caption actually emphasizes — two near-identical
    "FlatBass" files would split into ``bass``/``perc``/``drum`` on an incidental
    word like "percussive". We instead score by:

      1. occurrence count   — a caption that says "bass" twice means it,
      2. earliest mention    — the lead noun usually names the sound,
      3. vocabulary order    — stable tie-break only (its index in ``vocab``),

    so the category the model leaned on wins regardless of where it sits in the
    taxonomy list."""
    low = caption.lower()
    scored: list[tuple[str, int, int]] = []
    for rank, term in enumerate(vocab):
        matches = list(_word_re(term).finditer(low))
        if not matches:
            continue
        first = matches[0].start()
        occurrences = len(matches)
        scored.append((term, occurrences, first, rank))  # type: ignore[arg-type]
    # Strongest first: most occurrences, then earliest mention, then vocab order.
    scored.sort(key=lambda s: (-s[1], s[2], s[3]))
    return [(term, occ, first) for term, occ, first, _rank in scored]


class AceStepRecognizer:
    """Caption-based recognition via the ACE-Step captioner."""

    name = NAME

    def __init__(self) -> None:
        # Construct the captioner eagerly so a missing dependency fails fast at
        # backend-selection time (matches ClapRecognizer's contract); the model
        # itself is loaded lazily on first caption.
        from .captioner import get_captioner

        self._captioner = get_captioner()
        # Surface a missing `transformers`/`torch` now rather than mid-batch
        # (cheap import check — does not download the model).
        self._captioner.check_available()

    def recognize(self, items: list[FileProbe],
                  on_result: "ResultSink | None" = None) -> list[Recognition | None]:
        # Captioning is the slow part (seconds per file). Emit per-file progress
        # to stderr — it never touches the JSON envelope on stdout and shows up
        # in both the CLI and the `library serve` terminal. Silence with
        # MENDELL_QUIET=1. ``on_result`` (when given) is called as each file's
        # verdict lands so the caller can checkpoint a long scan for resume.
        #
        # Set MENDELL_ACE_ANALYTICS=<path> to also append a JSONL analytics
        # record per file (timing, caption, derived tags, VRAM) plus load/summary
        # events — readable after the fact without touching the server's stdout.
        total = len(items)
        analytics = _Analytics(os.environ.get("MENDELL_ACE_ANALYTICS"))

        # Load the model up front (if not already) so its cost is measured
        # separately from per-file inference rather than hiding in file #1.
        already = self._captioner._model is not None
        t_load = time.time()
        try:
            self._captioner._load()
        except Exception as err:
            analytics.emit({"event": "load_error", "error": f"{type(err).__name__}: {err}"})
            analytics.close()
            raise
        analytics.emit({
            "event": "load", "total_files": total, "already_loaded": already,
            "model_load_seconds": round(time.time() - t_load, 2) if not already else 0.0,
            "load_mode": self._captioner._load_mode(), "vram_mib": _gpu_mem_mib(),
        })

        results: list[Recognition | None] = [None] * total
        counts = {"captioned": 0, "deferred": 0}
        batch_size = _batch_size()
        run_start = time.time()

        # Plan batches length-first: the captioner pads each batch's mel to its
        # longest clip, so similar-length clips ride together (a long loop only
        # slows its own batch). Output order is preserved by scattering results
        # back to each item's input index.
        batches = _plan_batches(items, batch_size)
        analytics.emit({"event": "config", "batch_size": batch_size,
                        "num_batches": len(batches), "length_bucketed": True})

        done = 0
        for batch_no, idx_chunk in enumerate(batches, start=1):
            chunk = [items[idx] for idx in idx_chunk]
            # The longest clip in the batch sets the mel padding, so surface it
            # — makes the cost of a loop-heavy vs one-shot batch visible.
            durs = [c.duration for c in chunk if c.duration is not None]
            t_batch = time.time()
            captions = self._caption_chunk(chunk)
            analytics.emit({"event": "batch", "batch": batch_no, "size": len(chunk),
                            "max_duration": round(max(durs), 2) if durs else None,
                            "seconds": round(time.time() - t_batch, 2),
                            "vram_mib": _gpu_mem_mib()})
            for idx, item, caption in zip(idx_chunk, chunk, captions):
                done += 1
                results[idx] = self._record_file(
                    done, total, item, caption, analytics, counts
                )
                if on_result is not None:
                    # Stream the verdict out per file so the caller can
                    # checkpoint a long scan. Best-effort: a sink failure must
                    # not abort captioning (we still return the full list).
                    try:
                        on_result(item, results[idx])
                    except Exception:
                        pass

        elapsed = time.time() - run_start
        analytics.emit({"event": "summary", "total_files": total,
                        "captioned": counts["captioned"], "deferred": counts["deferred"],
                        "batch_size": batch_size, "inference_seconds": round(elapsed, 2),
                        "avg_seconds_per_file": round(elapsed / max(total, 1), 2),
                        "vram_mib": _gpu_mem_mib()})
        analytics.close()
        return results

    def _caption_chunk(self, chunk: list[FileProbe]) -> list[str | None]:
        """Caption a batch, returning one entry per item (``None`` marks a
        failure to defer). A whole-batch error is retried file-by-file so one
        bad file can't sink the rest of the batch."""
        if len(chunk) == 1:
            try:
                return [self._captioner.caption(str(chunk[0].path))]
            except Exception:
                return [None]
        try:
            return list(self._captioner.caption_batch([str(c.path) for c in chunk]))
        except Exception:
            out: list[str | None] = []
            for c in chunk:
                try:
                    out.append(self._captioner.caption(str(c.path)))
                except Exception:
                    out.append(None)
            return out

    def _record_file(self, i: int, total: int, item: FileProbe,
                     caption: str | None, analytics: "_Analytics",
                     counts: dict) -> Recognition | None:
        """Map one file's caption onto the taxonomy, emit progress/analytics,
        and return its ``Recognition`` (or ``None`` to defer to the filename
        guess)."""
        if caption is None:
            counts["deferred"] += 1
            self._progress(i, total, item.filename, "deferred (caption error)")
            analytics.emit({"event": "file", "i": i, "total": total,
                            "file": item.filename, "deferred": "caption error"})
            return None
        if not caption:
            counts["deferred"] += 1
            self._progress(i, total, item.filename, "deferred (empty caption)")
            analytics.emit({"event": "file", "i": i, "total": total,
                            "file": item.filename, "deferred": "empty caption"})
            return None

        cats = _categories(item.kind)
        scored_cats = _score_categories(caption, cats)
        category = scored_cats[0][0] if scored_cats else cats[0]
        instruments = _match_vocab(caption, INSTRUMENT_VOCAB)[:INSTRUMENT_CAP]
        confidence = CAPTION_CONFIDENCE if scored_cats else 0.3
        counts["captioned"] += 1

        self._progress(i, total, item.filename, f"-> {category}")
        analytics.emit({"event": "file", "i": i, "total": total, "file": item.filename,
                        "kind": item.kind, "category": category,
                        "category_matched": bool(scored_cats),
                        # All caption-supported categories, strongest-first, so a
                        # mis-bucketing can be diagnosed from the analytics log.
                        "category_candidates": [c[0] for c in scored_cats],
                        "instruments": instruments,
                        "confidence": confidence, "vram_mib": _gpu_mem_mib(),
                        "caption": caption})
        return Recognition(
            category=category,
            instruments=instruments,
            source=NAME,
            confidence=confidence,
            caption=caption,
        )

    @staticmethod
    def _progress(done: int, total: int, filename: str, note: str) -> None:
        if os.environ.get("MENDELL_QUIET"):
            return
        sys.stderr.write(f"[ace-step] {done}/{total} {filename} {note}\n")
        sys.stderr.flush()

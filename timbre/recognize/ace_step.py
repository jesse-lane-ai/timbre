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
from pathlib import Path

from .types import (
    GENRE_VOCAB,
    INSTRUMENT_VOCAB,
    FileProbe,
    Recognition,
    ResultSink,
    categories_for_kind,
    genre_score,
)

NAME = "ace-step"

# A caption hit on the taxonomy is a strong but text-derived signal.
CAPTION_CONFIDENCE = 0.7
# A category inferred indirectly from a matched instrument (no category word in
# the caption) — weaker than a direct hit, stronger than the neutral catch-all.
INSTRUMENT_CATEGORY_CONFIDENCE = 0.5
NEUTRAL_CATEGORY_CONFIDENCE = 0.3
INSTRUMENT_CAP = 4

# Neutral catch-all when nothing in the caption names a category (present in
# every kind's vocab) — far better than defaulting to the first list entry,
# which is the arbitrary, confidently-wrong "kick"/"drum".
NEUTRAL_CATEGORY = "fx"

# Recover a category from a matched instrument when the caption names an
# instrument ("cowbell", "piano") but no category word ("perc", "keys"). Each
# instrument maps to candidate categories in preference order; the first that's
# valid for the file's kind wins. Keeps a struck cowbell out of "kick".
_INSTRUMENT_TO_CATEGORY: dict[str, tuple[str, ...]] = {
    "kick": ("kick", "drum"), "snare": ("snare", "drum"),
    "clap": ("clap", "perc", "drum"), "snap": ("snap", "perc", "drum"),
    "rimshot": ("rim", "perc", "drum"), "hihat": ("hat", "drum"),
    "open hat": ("hat", "drum"), "crash": ("crash", "perc", "drum"),
    "ride": ("ride", "perc", "drum"), "cymbal": ("hat", "perc", "drum"),
    "tom": ("tom", "perc", "drum"),
    "conga": ("perc", "drum"), "bongo": ("perc", "drum"), "shaker": ("perc", "drum"),
    "tambourine": ("perc", "drum"), "cowbell": ("perc", "drum"),
    "woodblock": ("perc", "drum"), "clave": ("perc", "drum"),
    "triangle": ("perc", "drum"), "djembe": ("perc", "drum"),
    "timbale": ("perc", "drum"), "agogo": ("perc", "drum"), "cabasa": ("perc", "drum"),
    "drums": ("drum", "perc"), "percussion": ("perc", "drum"),
    "bass": ("bass",), "808": ("808", "bass"), "sub": ("sub", "bass"),
    "reese": ("reese", "bass"),
    # Pitched/tonal instruments all resolve to the single coarse `melodic`
    # category — the specific instrument is already carried in `instruments`.
    "lead": ("melodic",), "pad": ("melodic",),
    "pluck": ("melodic",), "arp": ("melodic",),
    "synth": ("melodic",),
    "keys": ("melodic",), "piano": ("melodic",),
    "organ": ("melodic",), "bell": ("perc", "melodic"),
    "guitar": ("melodic",), "strings": ("melodic",),
    "violin": ("melodic",), "cello": ("melodic",),
    "brass": ("melodic",), "trumpet": ("melodic",),
    "sax": ("melodic",), "flute": ("melodic",),
    # plucked world / folk strings — pitched, plucked.
    "koto": ("melodic",), "guzheng": ("melodic",),
    "zither": ("melodic",), "sitar": ("melodic",),
    "harp": ("melodic",), "banjo": ("melodic",),
    "ukulele": ("melodic",),
    # mallets / pitched percussion — struck, so prefer the drum `perc` bucket,
    # falling back to `melodic` for kinds without `perc`.
    "marimba": ("perc", "melodic"), "kalimba": ("perc", "melodic"),
    "vibraphone": ("perc", "melodic"), "xylophone": ("perc", "melodic"),
    "glockenspiel": ("perc", "melodic"),
    # other keyboards
    "harpsichord": ("melodic",), "accordion": ("melodic",),
    "choir": ("vocal",), "vocal": ("vocal",),
    "singing": ("vocal",), "spoken": ("vocal",),
    # sound design / fx — "noise" maps to the dedicated one-shot category first,
    # falling back to "fx" for kinds (loop/recording) that have no "noise" cat.
    "fx": ("fx",), "sound design": ("fx",), "noise": ("noise", "fx"),
}


def _category_from_instruments(instruments: list[str], cats: tuple[str, ...]) -> str | None:
    """First category (in instrument preference order) that's valid for the
    file's kind, or None if no matched instrument maps into ``cats``."""
    for ins in instruments:
        for cand in _INSTRUMENT_TO_CATEGORY.get(ins, ()):
            if cand in cats:
                return cand
    return None


# --- loop-context disambiguation -------------------------------------------
# A short, *unpitched* one-shot that the model captions as a tonal/melodic
# sound is a known failure: a dry drum hit (esp. with a bright click) reads to
# the captioner as "a single organ/piano chord". Re-captioning the same hit laid
# out as a four-on-the-floor loop gives the model the rhythmic context it was
# trained on, and it reliably flips to "kick/snare drum loop". We only do this
# for the narrow conflict — never for every one-shot — so melodic hits (guitar,
# piano) are untouched and the model cost stays near zero.
LOOP_CONTEXT_MAX_SECONDS = 1.5   # only short single hits
LOOP_CONTEXT_MAX_VOICED = 0.6    # only *unpitched* audio (drums), not tonal notes
LOOP_CONTEXT_BPM = 120.0
LOOP_CONTEXT_BARS = 2

# Tonal/melodic one-shot categories whose verdict on unpitched audio is suspect
# — the trigger for a loop re-caption.
_LOOP_CONTEXT_TONAL = {
    "bass", "sub", "808", "reese", "melodic",
}
# Categories that, when the *loop* caption lands on them, confirm a drum hit and
# override the tonal verdict.
_LOOP_CONTEXT_DRUM = {
    "kick", "snare", "clap", "snap", "hat", "tom", "crash", "ride", "rim",
    "perc", "808",
}


def _loop_context_enabled() -> bool:
    """On by default for the ace-step backend; disable with
    ACESTEP_LOOP_CONTEXT=0."""
    return os.environ.get("ACESTEP_LOOP_CONTEXT", "1").lower() not in ("0", "false", "no")


def _build_four_on_floor(path: str) -> str:
    """Render the one-shot at ``path`` as a peak-normalized four-on-the-floor
    loop (LOOP_CONTEXT_BARS bars at LOOP_CONTEXT_BPM) to a temp WAV; returns its
    path. Caller deletes it."""
    import tempfile

    import librosa
    import numpy as np
    import soundfile as sf

    sr = 44100
    y, _ = librosa.load(path, sr=sr, mono=True)
    step = int((60.0 / LOOP_CONTEXT_BPM) * sr)
    beats = LOOP_CONTEXT_BARS * 4
    buf = np.zeros(step * beats + len(y), dtype=np.float32)
    for i in range(beats):
        s = i * step
        buf[s:s + len(y)] += y
    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf /= peak
    fh = tempfile.NamedTemporaryFile(suffix="_4otf.wav", delete=False)
    fh.close()
    sf.write(fh.name, buf, sr)
    return fh.name


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


# Extra surface forms a vocab term should also match — captions inflect freely
# ("a string stab" for `strings`, "lo-fi synthesizer" for `synth`). Kept as an
# explicit, conservative list rather than algorithmic stemming (which mangles
# "bass"->"bas", "brass"->"bras"); every form stays whole-word anchored, so
# `sub` still won't fire on "subtle".
_VOCAB_VARIANTS: dict[str, tuple[str, ...]] = {
    # `melodic` is the coarse pitched-role category; captions usually say
    # "melody"/"melodic line" rather than the bare term.
    "melodic": ("melody", "melodies"),
    "strings": ("string",),
    "synth": ("synthesizer", "synthesizers", "synthesised", "synthesized", "synth pad", "synth lead"),
    "keys": ("keyboard", "keyboards"),
    "piano": ("pianos",),
    "organ": ("organs",),
    "guitar": ("guitars",),
    "pluck": ("plucked", "plucks"),
    "bell": ("bells", "chime", "chimes"),
    "pad": ("pads",),
    "lead": ("leads",),
    # Generic vocal — `singing`/`spoken` carry their own (subtype) terms below,
    # so they're kept off the generic `vocal` form to avoid double-matching.
    "vocal": ("vocals", "voice", "voices", "vox"),
    "singing": ("sung", "sings", "sung vocal", "vocal melody", "acapella"),
    "spoken": ("speech", "speaking", "spoken word", "spoken-word", "talking",
               "narration", "monologue", "dialogue"),
    "choir": ("choral", "chorale"),
    "brass": ("horns", "horn section"),
    "drums": ("drum",),
    "sound design": ("sound-design", "sfx", "sound effect", "sound effects", "sound fx"),
    "noise": ("static", "hiss", "white noise", "surface noise"),
    # genre surface forms (GENRE_VOCAB terms; matched only when scoring genres,
    # since these words aren't in the category/instrument vocabs).
    "hip hop": ("hip-hop", "hiphop"),
    "lo-fi": ("lofi", "low-fi", "low fi"),
    "drum and bass": ("dnb", "drum n bass", "drum'n'bass", "d&b"),
    "rnb": ("r&b", "r and b", "rhythm and blues"),
    "boom bap": ("boombap",),
    "tech house": ("tech-house",),
    "deep house": ("deep-house",),
    "future bass": ("future-bass",),
}


def _word_re(term: str) -> "re.Pattern[str]":
    """A whole-word matcher for a vocab term (plus its `_VOCAB_VARIANTS` surface
    forms) — so 'snap' doesn't fire on 'snappy', 'sub' on 'subtle', or 'rim' on
    'primary', but 'strings' still catches "string". Cached per term."""
    cached = _WORD_RE_CACHE.get(term)
    if cached is None:
        forms = (term, *_VOCAB_VARIANTS.get(term, ()))
        alt = "|".join(re.escape(f) for f in forms)
        cached = re.compile(rf"\b(?:{alt})\b")
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


# How many genres a caption may yield, and the source confidence ceiling. These
# are caption-derived pseudo-scores (occurrence-weighted), not the calibrated
# softmax scores the clap backend produces — kept modest to signal that.
GENRE_CAP = 3
GENRE_CAPTION_CONFIDENCE = 0.6


def _score_genres(caption: str) -> list[dict]:
    """Harvest ranked genres from a free-text caption: whole-word match against
    GENRE_VOCAB, weight by occurrence count, normalise to 0..1, keep the top
    ``GENRE_CAP``. Empty when the caption names no genre (the common one-shot
    case). Pseudo-confidence — a caption that says "techno" twice scores it
    higher than an incidental single mention, but it is not a probability."""
    low = caption.lower()
    counts: list[tuple[str, int]] = []
    for term in GENRE_VOCAB:
        n = len(list(_word_re(term).finditer(low)))
        if n:
            counts.append((term, n))
    if not counts:
        return []
    counts.sort(key=lambda c: (-c[1], GENRE_VOCAB.index(c[0])))
    counts = counts[:GENRE_CAP]
    total = sum(n for _, n in counts)
    return [genre_score(term, GENRE_CAPTION_CONFIDENCE * n / total) for term, n in counts]


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
            "load_mode": getattr(self._captioner, "_effective_mode", None) or self._captioner._load_mode(),
            "vram_mib": _gpu_mem_mib(),
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

        # Loop-context pass: re-caption the narrow set of short, unpitched
        # one-shots the model mislabeled as tonal, using a four-on-the-floor
        # loop to supply rhythmic context. Overrides in place.
        if _loop_context_enabled():
            self._apply_loop_context(items, results, analytics, on_result)

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
        instruments = _match_vocab(caption, INSTRUMENT_VOCAB)[:INSTRUMENT_CAP]
        genres = _score_genres(caption)
        if scored_cats:
            category, confidence = scored_cats[0][0], CAPTION_CONFIDENCE
        else:
            # No category word in the caption: recover it from a matched
            # instrument if we can, else a neutral catch-all — never the
            # arbitrary first list entry ("kick").
            derived = _category_from_instruments(instruments, cats)
            if derived is not None:
                category, confidence = derived, INSTRUMENT_CATEGORY_CONFIDENCE
            else:
                category, confidence = NEUTRAL_CATEGORY, NEUTRAL_CATEGORY_CONFIDENCE
        counts["captioned"] += 1

        self._progress(i, total, item.filename, f"-> {category}")
        analytics.emit({"event": "file", "i": i, "total": total, "file": item.filename,
                        "kind": item.kind, "category": category,
                        "category_matched": bool(scored_cats),
                        # All caption-supported categories, strongest-first, so a
                        # mis-bucketing can be diagnosed from the analytics log.
                        "category_candidates": [c[0] for c in scored_cats],
                        "instruments": instruments,
                        "genres": genres,
                        "confidence": confidence, "vram_mib": _gpu_mem_mib(),
                        "caption": caption})
        return Recognition(
            category=category,
            instruments=instruments,
            source=NAME,
            confidence=confidence,
            caption=caption,
            genres=genres,
        )

    def _apply_loop_context(self, items: list[FileProbe],
                            results: list["Recognition | None"],
                            analytics: "_Analytics", on_result) -> None:
        """Second-opinion pass for the narrow drum-as-tonal failure: short,
        *unpitched* one-shots the model captioned as a tonal/melodic category get
        re-captioned as a four-on-the-floor loop; if the loop caption lands on a
        drum category, override the verdict. Only the conflicted files are
        touched (typically a handful), so the cost is a few short extra captions."""
        from .. import audio_analysis

        candidates: list[int] = []
        for idx, (item, rec) in enumerate(zip(items, results)):
            if rec is None or item.kind != "one-shot":
                continue
            if (item.duration or 1e9) > LOOP_CONTEXT_MAX_SECONDS:
                continue
            if rec.category not in _LOOP_CONTEXT_TONAL:
                continue
            # Cheap filters passed; now the audio check — only *unpitched* hits
            # (a real tonal note keeps its caption).
            try:
                voiced = audio_analysis._AnalysisCache(str(item.path)).voiced_ratio()
            except Exception:
                continue
            if voiced < LOOP_CONTEXT_MAX_VOICED:
                candidates.append(idx)

        if not candidates:
            return

        loops: dict[int, str] = {}
        try:
            for idx in candidates:
                try:
                    loops[idx] = _build_four_on_floor(str(items[idx].path))
                except Exception:
                    pass
            if not loops:
                return
            order = list(loops.keys())
            captions = self._caption_chunk([
                FileProbe(path=Path(loops[i]), filename=items[i].filename,
                          duration=float(LOOP_CONTEXT_BARS * 4 * 60.0 / LOOP_CONTEXT_BPM),
                          kind="loop")
                for i in order
            ])
        finally:
            for p in loops.values():
                try:
                    os.unlink(p)
                except OSError:
                    pass

        cats = _categories("one-shot")
        for idx, caption in zip(order, captions):
            if not caption:
                continue
            scored = _score_categories(caption, cats)
            new_cat = scored[0][0] if scored else None
            if new_cat not in _LOOP_CONTEXT_DRUM:
                continue
            item, old = items[idx], results[idx]
            instruments = _match_vocab(caption, INSTRUMENT_VOCAB)[:INSTRUMENT_CAP]
            results[idx] = Recognition(
                category=new_cat, instruments=instruments, source=NAME,
                confidence=CAPTION_CONFIDENCE, caption=caption,
            )
            self._progress(idx + 1, len(items), item.filename,
                           f"-> {new_cat} (loop-context, was {old.category})")
            analytics.emit({"event": "loop_context", "file": item.filename,
                            "was": old.category, "now": new_cat,
                            "instruments": instruments, "caption": caption})
            if on_result is not None:
                try:
                    on_result(item, results[idx])
                except Exception:
                    pass

    @staticmethod
    def _progress(done: int, total: int, filename: str, note: str) -> None:
        if os.environ.get("MENDELL_QUIET"):
            return
        sys.stderr.write(f"[ace-step] {done}/{total} {filename} {note}\n")
        sys.stderr.flush()

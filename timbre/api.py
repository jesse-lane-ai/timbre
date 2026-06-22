"""timbre's one core: ``classify`` / ``classify_many``.

Everything else (CLI, HTTP server, and external callers like Mendell) is a thin
wrapper over this. Given an audio source it returns a :class:`Tags` verdict that
fuses two cheap-to-expensive passes:

  1. **name pass** — :func:`timbre.names.classify_from_names` parses the
     filename + parent folders (no audio loaded): kind/category/key/scale/bpm.
  2. **content pass** — a pluggable recognizer backend (``heuristic`` / ``clap``
     / ``ace-step``) listens to the audio and returns a category + instruments.

A *source* is either a filesystem path (str/Path) or raw ``bytes`` (an upload).
Bytes are spilled to a temp file so the path-based probes/backends work
unchanged; pass ``filename=`` alongside bytes so the name pass still has
something to parse.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from . import audio_analysis
from .names import classify_from_names
from .recognize import FileProbe, get_recognizer
from .recognize.heuristic import HeuristicRecognizer

Source = "str | Path | bytes"


@dataclass(frozen=True)
class Tags:
    """Fused name + content verdict for one audio file."""

    filename: str
    kind: str  # "loop" | "one-shot" | "unknown"
    category: str | None
    instruments: list[str] = field(default_factory=list)
    key: str | None = None
    scale: str | None = None
    bpm: float | None = None
    duration: float | None = None
    confidence: float = 0.0
    caption: str | None = None
    backend: str = "heuristic"
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(source, backend: str = "heuristic", *, filename: str | None = None) -> Tags:
    """Classify a single audio *source* (path or raw bytes)."""
    return classify_many([source], backend=backend, filenames=[filename] if filename else None)[0]


def classify_many(
    sources: Iterable,
    backend: str = "heuristic",
    *,
    filenames: list[str] | None = None,
    on_result=None,
) -> list[Tags]:
    """Batch classify — backends amortize model load over the whole list.

    ``filenames`` is an optional parallel list of name hints, required to get a
    useful name pass for ``bytes`` sources (ignored for path sources, which
    carry their own name).
    """
    sources = list(sources)
    hints = filenames or [None] * len(sources)
    if len(hints) != len(sources):
        from .errors import BadInputError

        raise BadInputError("filenames must be the same length as sources")

    # Spill any bytes sources to temp files so path-based probes/backends work.
    # `name_src` is what the name pass parses — the FULL path/name (incl. parent
    # folders, which carry bpm/key/kind context), distinct from the temp `real`
    # path that bytes sources land at. `display` is the bare filename.
    tmpdir: tempfile.TemporaryDirectory | None = None
    resolved: list[tuple[Path, str, str]] = []  # (real path, display name, name source)
    try:
        for i, src in enumerate(sources):
            if isinstance(src, (bytes, bytearray)):
                if tmpdir is None:
                    tmpdir = tempfile.TemporaryDirectory(prefix="timbre-")
                name = hints[i] or f"upload-{i}.wav"
                p = Path(tmpdir.name) / Path(name).name
                p.write_bytes(bytes(src))
                resolved.append((p, Path(name).name, name))
            else:
                # Keep the path as given so the name pass sees parent folders;
                # resolve only the real path used for reading audio.
                p = Path(src).expanduser()
                resolved.append((p.resolve(), p.name, str(src)))

        # --- name pass + probes ---
        name_infos: list[dict] = []
        probes: list[FileProbe] = []
        for real, display, name_src in resolved:
            ni = classify_from_names(name_src)
            name_infos.append(ni)
            duration = audio_analysis.probe_duration_seconds(str(real))
            kind = ni["kind"] or "unknown"
            probes.append(FileProbe(path=real, filename=Path(display).name, duration=duration, kind=kind))

        # --- content pass ---
        if backend == "heuristic":
            recognizer = HeuristicRecognizer()
        else:
            recognizer = get_recognizer(backend)
        import inspect

        if on_result is not None and "on_result" in inspect.signature(recognizer.recognize).parameters:
            recs = recognizer.recognize(probes, on_result=on_result)
        else:
            recs = recognizer.recognize(probes)

        # --- fuse ---
        out: list[Tags] = []
        for (real, display, _name_src), ni, probe, rec in zip(resolved, name_infos, probes, recs):
            category = ni["category"]
            instruments = list(ni["instruments"])
            confidence = ni["confidence"]
            caption = None
            if rec is not None:
                # Content backend wins category when names had none; merge instruments.
                if category is None:
                    category = rec.category
                for ins in rec.instruments:
                    if ins not in instruments:
                        instruments.append(ins)
                confidence = max(confidence, rec.confidence)
                caption = rec.caption
            out.append(
                Tags(
                    filename=Path(display).name,
                    kind=probe.kind,
                    category=category,
                    instruments=instruments,
                    key=ni["key"],
                    scale=ni["scale"],
                    bpm=ni["bpm"],
                    duration=probe.duration,
                    confidence=round(confidence, 3),
                    caption=caption,
                    backend=backend,
                    path=None if tmpdir and str(real).startswith(tmpdir.name) else str(real),
                )
            )
        return out
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

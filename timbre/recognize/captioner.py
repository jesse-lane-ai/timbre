"""``AceCaptioner`` — ACE-Step's standalone music-captioner model.

``ACE-Step/acestep-captioner`` is a Qwen2.5-Omni-7B multimodal model that emits
a free-text description of a piece of audio. Unlike the generative ACE-Step
stack (DiT + planner LM), it's a plain ``transformers`` model, so it needs only
``torch`` + ``transformers`` and no ACE-Step checkpoint — a much lighter
dependency surface, which makes it the right backend for *recognition* (the
``ace-step`` library recognizer) rather than reusing the generation engine's
``understand_music`` path.

Lazy and opt-in like every other model backend: constructing this class is free;
the model is built on first ``caption()`` call, and a missing ``transformers``
raises an actionable ``BadInputError``.

Config (env):
  * ``ACESTEP_CAPTIONER_MODEL`` — HF repo id or local path
                                  (default ``ACE-Step/acestep-captioner``).
  * ``ACESTEP_DEVICE``          — shared with the generation engine
                                  (``cuda`` | ``mps`` | ``cpu`` | ``xpu``).
  * ``ACESTEP_CAPTIONER_LOAD``  — in-flight quantization: ``full`` (default),
                                  ``8bit``, or ``4bit``. The quantized modes use
                                  bitsandbytes (CUDA-only) to shrink the ~22 GB
                                  model to ~11 GB / ~6–7 GB, quantizing only the
                                  LLM tower so the audio encoder stays accurate.
  * ``ACESTEP_CAPTIONER_BATCH`` — files per ``generate()`` call (default ``1``).
                                  Higher values amortize per-call overhead and
                                  cut wall-clock on large scans, at the cost of
                                  more VRAM (longest clip in the batch sets the
                                  padded length). Try 4–8 on a 24 GB card.
  * ``ACESTEP_LOOP_CONTEXT``    — default on; set ``0`` to disable the
                                  loop-context disambiguation pass (see
                                  ``recognize.ace_step._apply_loop_context``):
                                  short *unpitched* one-shots captioned as tonal
                                  are re-captioned as a four-on-the-floor loop and
                                  overridden to a drum category when the loop
                                  reads as one.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys

from ..errors import BadInputError, EngineError


class _DropSystemPromptWarning(logging.Filter):
    """Drop Qwen2.5-Omni's "System prompt modified, audio output may not work"
    warning. We deliberately replace its verbose default system prompt and
    disable the talker (we only want the text caption, never audio), so the
    warning is pure noise emitted on every batch."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "System prompt modified" not in record.getMessage()


_warning_filter_installed = False


def _silence_system_prompt_warning() -> None:
    """Install the filter once on the root logger (where the model logs it)."""
    global _warning_filter_installed
    if not _warning_filter_installed:
        logging.getLogger().addFilter(_DropSystemPromptWarning())
        _warning_filter_installed = True


def _stdout_to_stderr():
    """Redirect stdout to stderr for the enclosed block.

    transformers' ``auto_docstring`` decorator unconditionally ``print()``s
    "🚨 … not documented" lines to *stdout* when it imports certain model
    modules (e.g. qwen2_5_omni). Our CLI puts the ``--json`` envelope on stdout,
    so that stray output would corrupt the JSON an agent parses. Routing it to
    stderr keeps stdout clean while still surfacing the messages in a terminal.
    """
    return contextlib.redirect_stdout(sys.stderr)

DEFAULT_CAPTIONER_MODEL = "ACE-Step/acestep-captioner"
# Prompt for the captioner. The model card documents "Describe this audio in
# detail"; we ask for one terse sentence naming the instrument + character,
# which is all the taxonomy keyword-matching needs (the model emits ~2 sentences
# either way, so this is about caption quality/brevity, not speed — the speed
# levers are batching and the audio-length cap). Override with
# ACESTEP_CAPTIONER_PROMPT.
DEFAULT_CAPTION_PROMPT = (
    "Describe this sound in one short sentence: name the instrument or sound "
    "type and its character. Be brief."
)


def _caption_prompt() -> str:
    return os.environ.get("ACESTEP_CAPTIONER_PROMPT", DEFAULT_CAPTION_PROMPT)

CAPTIONER_INSTALL_HINT = (
    "the ACE-Step captioner needs torch + transformers (<5). Install them with: "
    "pip install 'timbre[ace-step]'  (the model "
    "'ACE-Step/acestep-captioner' downloads on first use; override with "
    "ACESTEP_CAPTIONER_MODEL)."
)


class AceCaptioner:
    """Lazy handle on the ACE-Step (Qwen2.5-Omni) captioner."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._effective_mode = None  # set by _resolve_load to the mode actually used

    def _model_id(self) -> str:
        return os.environ.get("ACESTEP_CAPTIONER_MODEL", DEFAULT_CAPTIONER_MODEL)

    def _device(self) -> str:
        return os.environ.get("ACESTEP_DEVICE", "cuda")

    def _max_audio_seconds(self) -> float:
        """Cap on how many seconds of each clip are fed to the audio encoder
        (env ``ACESTEP_CAPTIONER_AUDIO_SECONDS``, default 30).

        This is the single biggest per-file lever. Qwen2.5-Omni's feature
        extractor pads every clip's mel to a fixed ~300-second window, so a
        half-second drum hit pays the full 300 s of conv-encoder compute. We
        instead pad the mel only to the longest clip actually in the batch
        (bounded by this cap), which for one-shots collapses the encoder cost to
        near zero while leaving captions bit-identical (the model already masks
        the padding, so trimming it changes nothing but speed). Raise the cap if
        you caption long loops and want the encoder to see all of them."""
        try:
            return max(0.5, float(os.environ.get("ACESTEP_CAPTIONER_AUDIO_SECONDS", "30")))
        except ValueError:
            return 30.0

    def _max_new_tokens(self) -> int:
        """Worst-case caption length cap (env ``ACESTEP_CAPTIONER_MAX_TOKENS``,
        default 96).

        Decode stops at the model's natural EOS (~40 tokens for these captions)
        well before this cap, so it bounds pathological runaways rather than
        typical speed — the real per-file levers are batching and the audio
        cap. Kept modest (vs the model's 256) as a guardrail."""
        try:
            return max(1, int(os.environ.get("ACESTEP_CAPTIONER_MAX_TOKENS", "96")))
        except ValueError:
            return 96

    def _load_mode(self) -> str:
        """In-flight quantization mode: ``4bit`` (default), ``8bit``, or
        ``full`` (fp16/bf16). The quantized modes shrink the ~22 GB captioner to
        roughly ~7 GB / ~11 GB of VRAM via bitsandbytes, quantizing only the LLM
        tower (the audio encoder stays full precision).

        4bit is the default deliberately: the full ~22 GB load maxes a 24 GB
        card, and (notably under WSL2) the driver then spills to system RAM,
        collapsing inference throughput ~15x (~60 s/file vs ~4 s/file measured
        on a 3090). 4bit stays comfortably resident with near-identical caption
        quality for coarse tagging. Falls back to full when its deps are absent
        — see :meth:`_resolve_load`."""
        mode = os.environ.get("ACESTEP_CAPTIONER_LOAD", "4bit").lower()
        if mode not in ("full", "8bit", "4bit"):
            raise BadInputError(
                f"ACESTEP_CAPTIONER_LOAD must be 'full', '8bit', or '4bit' (got '{mode}')"
            )
        return mode

    def _resolve_load(self):
        """Resolve ``(mode, quant_config)``, degrading a quantized *default* to
        full precision when bitsandbytes/accelerate are missing rather than
        hard-failing — so the tool still runs out of the box. An *explicit*
        ``ACESTEP_CAPTIONER_LOAD`` is honoured strictly (still errors if its deps
        are absent). Records the effective mode on ``self`` for analytics."""
        mode = self._load_mode()
        explicit = "ACESTEP_CAPTIONER_LOAD" in os.environ
        try:
            quant_config = self._quant_config(mode)
        except BadInputError:
            if explicit or mode == "full":
                raise
            print(
                f"[ace-step] '{mode}' load needs bitsandbytes + accelerate "
                f"(not installed) — falling back to full precision. Install them "
                f"(pip install bitsandbytes accelerate) for a ~15x speedup, or "
                f"set ACESTEP_CAPTIONER_LOAD=full to silence this.",
                file=sys.stderr,
            )
            mode, quant_config = "full", None
        self._effective_mode = mode
        return mode, quant_config

    def _quant_config(self, mode: str):
        """Build a bitsandbytes ``BitsAndBytesConfig`` for the quantized modes,
        or ``None`` for ``full``. bitsandbytes is CUDA-only, so this is the GPU
        path; raise an actionable error if the dep is missing."""
        if mode == "full":
            return None
        try:
            import accelerate  # noqa: F401  (device_map placement for quantized load)
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as err:
            raise BadInputError(
                f"ACESTEP_CAPTIONER_LOAD={mode} needs bitsandbytes + accelerate "
                f"(CUDA-only) — install them with: pip install bitsandbytes accelerate "
                f"(missing: {err.name})"
            )
        if mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

    def check_available(self) -> None:
        """Verify the import-time dependencies are present *without* loading or
        downloading the model — used at backend-selection time to fail fast."""
        try:
            with _stdout_to_stderr():
                import torch  # noqa: F401
                from transformers import (  # noqa: F401
                    AutoProcessor,
                    Qwen2_5OmniForConditionalGeneration,
                )
        except ImportError as err:
            raise BadInputError(f"{CAPTIONER_INSTALL_HINT} (missing: {err.name})")

    def _load(self):
        if self._model is not None:
            return self._model, self._processor

        _silence_system_prompt_warning()
        try:
            # Importing the qwen2_5_omni model module triggers transformers'
            # auto_docstring print()s to stdout — keep them off the JSON channel.
            with _stdout_to_stderr():
                import torch  # noqa: F401
                from transformers import (
                    AutoProcessor,
                    Qwen2_5OmniForConditionalGeneration,
                )
        except ImportError as err:
            raise BadInputError(f"{CAPTIONER_INSTALL_HINT} (missing: {err.name})")

        model_id = self._model_id()
        mode, quant_config = self._resolve_load()
        try:
          # Keep transformers' load-time chatter (incl. auto_docstring prints and
          # progress bars) off stdout so the CLI's JSON envelope stays clean.
          with _stdout_to_stderr():
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            kwargs = {"trust_remote_code": True}
            # Scaled-dot-product attention is much faster than the eager path on
            # the prefill (the dominant per-call cost for short clips). Honoured
            # by recent transformers; harmless string if an older build ignores
            # it. Override via ACESTEP_CAPTIONER_ATTN (e.g. flash_attention_2).
            attn = os.environ.get("ACESTEP_CAPTIONER_ATTN", "sdpa")
            if attn:
                kwargs["attn_implementation"] = attn
            if quant_config is not None:
                # bitsandbytes places weights on the GPU itself and forbids a
                # later .to(); let device_map handle placement.
                kwargs["quantization_config"] = quant_config
                kwargs["device_map"] = self._device()
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if quant_config is None:
                model.to(self._device())
            model.eval()
            # Qwen2.5-Omni ships a Talker (TTS) tower we never use — captioning
            # only needs the Thinker's text output. Dropping it frees VRAM and
            # removes it from the generate() path. Best-effort across builds.
            disable = getattr(model, "disable_talker", None)
            if callable(disable):
                try:
                    disable()
                except Exception:
                    pass
        except Exception as err:
            raise EngineError(f"failed to load ACE-Step captioner '{model_id}': {err}")

        self._model, self._processor = model, processor
        return model, processor

    def unload(self) -> None:
        """Drop the model + processor and free GPU memory.

        Used by long-running hosts (e.g. the library server) to release the
        ~6–22 GB of VRAM the captioner holds once a scan is done. The next
        ``caption()`` transparently reloads — so callers that import
        back-to-back should keep it warm instead. No-op if nothing is loaded."""
        if self._model is None and self._processor is None:
            return
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_audio(self, path: str):
        """Load audio at the processor's expected sampling rate (Qwen2.5-Omni
        uses 16 kHz)."""
        import librosa

        sr = getattr(getattr(self._processor, "feature_extractor", None),
                     "sampling_rate", 16000)
        waveform, _ = librosa.load(path, sr=sr, mono=True)
        return waveform

    def _conversation(self, audio):
        """Qwen2.5-Omni-style multimodal chat: one audio turn + the captioning
        instruction. Built via the chat template so it works across processor
        versions; the documented "<audio>" placeholder is supplied by the
        template's audio content part.

        An explicit, terse system turn suppresses the model's verbose default
        ("You are Qwen, a virtual human ... capable of perceiving auditory and
        visual inputs ..."), which would otherwise be prefilled on every call —
        pure overhead for a captioning task and a measurable slice of the
        per-file time on short clips."""
        return [
            {"role": "system", "content": [
                {"type": "text", "text": "You are an audio captioner."},
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": _caption_prompt()},
            ]},
        ]

    def caption(self, path: str) -> str:
        """Return a free-text caption for the audio file at ``path``."""
        return self.caption_batch([path])[0]

    def caption_batch(self, paths: list[str]) -> list[str]:
        """Caption several files in one ``generate()`` call.

        Batching amortizes the per-call Python/kernel-launch overhead across
        files, which is the main throughput lever for the captioner (a
        multi-hour library scan is dominated by thousands of single-file
        round-trips). Decoder generation is done with **left padding** so every
        row's prompt is the same length and the prompt-trim below is uniform.

        Returns one caption per input path, in order. An empty list in, empty
        list out.
        """
        if not paths:
            return []
        import torch

        model, processor = self._load()
        audios = [self._load_audio(p) for p in paths]
        texts = [
            processor.apply_chat_template(
                self._conversation(a), add_generation_prompt=True, tokenize=False
            )
            for a in audios
        ]

        # Left-pad so the (right-aligned) prompts share a common length; without
        # it batched generation would mis-trim and emit pad tokens mid-caption.
        tok = getattr(processor, "tokenizer", None)
        prev_side = getattr(tok, "padding_side", None) if tok is not None else None
        if tok is not None:
            tok.padding_side = "left"
        # Shrink the mel pad target from the model's fixed ~300 s window to the
        # longest clip actually in this batch (capped). The feature extractor
        # pads to nb_max_frames, so lowering it for the call's duration avoids
        # convolving hundreds of seconds of silence per file. Restored in the
        # finally below so nothing else sees the mutated extractor.
        fe = getattr(processor, "feature_extractor", None)
        prev_frames = prev_samples = None
        if fe is not None and getattr(fe, "hop_length", 0):
            sr = getattr(fe, "sampling_rate", 16000)
            hop = fe.hop_length
            longest = max((len(a) for a in audios), default=0)
            cap_samples = int(self._max_audio_seconds() * sr)
            target = min(longest, cap_samples)
            frames = max(1, target // hop + 4)  # +pad for conv edge frames
            if getattr(fe, "nb_max_frames", None) and frames < fe.nb_max_frames:
                prev_frames = fe.nb_max_frames
                prev_samples = getattr(fe, "n_samples", None)
                fe.nb_max_frames = frames
                fe.n_samples = frames * hop
        try:
            inputs = processor(
                text=texts, audio=audios, return_tensors="pt", padding=True
            ).to(self._device())
            with torch.no_grad():
                # Qwen2.5-Omni can also synthesize speech, in which case
                # generate() returns (text_ids, audio_waveform). Ask for
                # text-only; fall back if this build doesn't accept the kwarg.
                max_tokens = self._max_new_tokens()
                try:
                    generated = model.generate(
                        **inputs, max_new_tokens=max_tokens, return_audio=False
                    )
                except TypeError:
                    generated = model.generate(**inputs, max_new_tokens=max_tokens)
            # Unwrap a (text_ids, audio) tuple if the talker still fired.
            if isinstance(generated, (tuple, list)):
                generated = generated[0]
            # Drop the (uniform, left-padded) prompt tokens before decoding.
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        except Exception as err:
            raise EngineError(f"ACE-Step captioner inference error: {err}")
        finally:
            if tok is not None and prev_side is not None:
                tok.padding_side = prev_side
            if prev_frames is not None:
                fe.nb_max_frames = prev_frames
                if prev_samples is not None:
                    fe.n_samples = prev_samples
        return [(c or "").strip() for c in out]


_CAPTIONER: AceCaptioner | None = None


def get_captioner() -> AceCaptioner:
    global _CAPTIONER
    if _CAPTIONER is None:
        _CAPTIONER = AceCaptioner()
    return _CAPTIONER

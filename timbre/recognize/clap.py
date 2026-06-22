"""``clap`` recognizer — local zero-shot audio-text embedding (opt-in extra).

Lazy-imports ``torch`` + ``laion_clap`` only when this backend is actually
selected, so the base install never pays for them. Zero-shot: embeds each
file's audio plus a text prompt for every category/instrument in the
taxonomy, and:

  * ``category``    — the single highest-scoring category prompt for the
                       file's ``kind`` (one-shot vs loop vocabulary).
  * ``instruments`` — instrument prompts scoring at or above
                       ``INSTRUMENT_THRESHOLD_RATIO`` of the top instrument
                       score, capped at ``INSTRUMENT_CAP``, always keeping at
                       least the top-1.

Scaffolded — cannot be exercised in this environment (no ``torch``/
``laion_clap`` and no GPU/model download). Selecting this backend without the
extra installed raises an actionable ``BadInputError``.
"""

from __future__ import annotations

from ..errors import BadInputError
from .types import INSTRUMENT_VOCAB, FileProbe, Recognition, categories_for_kind

NAME = "clap"

# Keep an instrument label if its similarity score is within this fraction of
# the top instrument score (e.g. 0.5 == "at least half as confident as the
# best match").
INSTRUMENT_THRESHOLD_RATIO = 0.5
# Hard cap on how many instrument labels a single file can carry.
INSTRUMENT_CAP = 4

CLAP_INSTALL_HINT = (
    "the 'clap' recognizer needs the optional CLAP dependencies — "
    "install them with: pip install 'timbre[clap]'"
)


def _category_prompts(kind: str) -> tuple[str, ...]:
    return categories_for_kind(kind)


class ClapRecognizer:
    """Local zero-shot audio-text embedding via LAION-CLAP."""

    name = NAME

    def __init__(self) -> None:
        import sys

        # LAION-CLAP builds its training arg parser at *import* time and calls
        # ``parse_args()`` on the real ``sys.argv``, so it exits the moment it
        # sees our CLI flags. Shield it behind a bare argv during import.
        saved_argv = sys.argv
        sys.argv = [saved_argv[0]] if saved_argv else [""]
        try:
            import torch  # noqa: F401
            import laion_clap  # noqa: F401
        except ImportError as err:
            raise BadInputError(
                f"{CLAP_INSTALL_HINT} (missing: {err.name})"
            )
        finally:
            sys.argv = saved_argv
        # Model load is deferred to first `recognize()` call so constructing
        # the registry entry (e.g. for `library.search` introspection) never
        # forces a model download.
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import sys

            import laion_clap

            # ``CLAP_Module`` constructs LAION-CLAP's training arg parser, which
            # calls ``parse_args()`` on the *real* ``sys.argv`` and exits when it
            # sees our CLI's flags (e.g. ``library add ... --recognize clap``).
            # Shield it behind a bare argv while the model is built.
            import torch

            saved_argv = sys.argv
            sys.argv = [saved_argv[0]] if saved_argv else [""]
            # torch >= 2.6 defaults ``torch.load(weights_only=True)``, which
            # rejects the (trusted) LAION-CLAP checkpoint's numpy globals.
            # Force the legacy behaviour just for the checkpoint load.
            saved_load = torch.load

            def _load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return saved_load(*args, **kwargs)

            torch.load = _load
            try:
                model = laion_clap.CLAP_Module(enable_fusion=False)
                # Newer ``transformers`` drops buffers like
                # ``text_branch.embeddings.position_ids`` that exist in the
                # published checkpoint, so a strict load fails. Force the
                # inner module's ``load_state_dict`` to be non-strict.
                inner = model.model
                strict_load = inner.load_state_dict

                def _load_state_dict(state_dict, strict=True):
                    return strict_load(state_dict, strict=False)

                inner.load_state_dict = _load_state_dict
                model.load_ckpt()  # downloads/loads the pretrained checkpoint
            finally:
                sys.argv = saved_argv
                torch.load = saved_load
            self._model = model
        return self._model

    def recognize(self, items: list[FileProbe]) -> list[Recognition | None]:
        if not items:
            return []

        import numpy as np

        model = self._ensure_model()

        def _np(embed):
            # This ``laion_clap`` build returns torch tensors (no ``use_tensor``
            # kwarg); normalise everything to numpy for the scoring below.
            if hasattr(embed, "detach"):
                return embed.detach().cpu().numpy()
            return np.asarray(embed)

        paths = [str(item.path) for item in items]
        audio_embeds = _np(model.get_audio_embedding_from_filelist(x=paths))

        results: list[Recognition | None] = []
        for item, audio_embed in zip(items, audio_embeds):
            categories = _category_prompts(item.kind)
            category_embeds = _np(model.get_text_embedding(list(categories)))
            category_scores = audio_embed @ category_embeds.T
            best_idx = int(category_scores.argmax())
            category = categories[best_idx]
            category_confidence = float(category_scores[best_idx])

            instrument_embeds = _np(model.get_text_embedding(list(INSTRUMENT_VOCAB)))
            instrument_scores = audio_embed @ instrument_embeds.T
            top_score = float(instrument_scores.max())
            threshold = top_score * INSTRUMENT_THRESHOLD_RATIO
            order = instrument_scores.argsort()[::-1]
            instruments = []
            for idx in order:
                if len(instruments) >= INSTRUMENT_CAP:
                    break
                if len(instruments) == 0 or instrument_scores[idx] >= threshold:
                    instruments.append(INSTRUMENT_VOCAB[idx])

            results.append(
                Recognition(
                    category=category,
                    instruments=instruments,
                    source=NAME,
                    confidence=max(0.0, min(1.0, category_confidence)),
                )
            )
        return results

"""timbre — content + filename based audio sample classifier.

Public API::

    import timbre
    tags = timbre.classify("kick_01.wav")            # -> Tags
    tags = timbre.classify(raw_bytes, filename="x.wav", backend="ace-step")
    many = timbre.classify_many(paths, backend="heuristic")

Backends: ``heuristic`` (local, no extra deps), ``clap`` (extra: ``timbre[clap]``),
``ace-step`` (extra: ``timbre[ace-step]``). See :func:`timbre.list_backends`.
"""

from .api import (
    Tags,
    classify,
    classify_many,
    collection_add,
    collection_create,
    collection_delete,
    collection_remove,
    collection_rename,
    collections,
    delete,
    get,
    query,
    update,
)
from .recognize import FileProbe, Recognition, get_recognizer, list_backends

__all__ = [
    "Tags",
    "classify",
    "classify_many",
    "query",
    "get",
    "update",
    "delete",
    "collections",
    "collection_create",
    "collection_rename",
    "collection_delete",
    "collection_add",
    "collection_remove",
    "list_backends",
    "get_recognizer",
    "FileProbe",
    "Recognition",
]

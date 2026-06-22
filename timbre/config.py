"""User config — where the scan DB lives and whether it's used.

Persistence is **on by default**: with no flags, ``timbre scan`` writes to a
shared DB at the XDG data location. Resolution order for each setting is
env var > config file > built-in default.

Config file (TOML, written by ``timbre config set``)::

    [db]
    enabled = true
    path = "/home/me/.local/share/timbre/tags.db"

Env overrides: ``TIMBRE_CONFIG`` (config file path), ``TIMBRE_DB`` (db path),
``TIMBRE_DB_ENABLED`` (0/1/true/false).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

_FALSEY = {"0", "false", "no", "off", ""}


def config_path() -> Path:
    env = os.environ.get("TIMBRE_CONFIG")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "timbre" / "config.toml"


def default_db_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "timbre" / "tags.db"


def load() -> dict[str, Any]:
    cp = config_path()
    if cp.exists():
        with open(cp, "rb") as f:
            return tomllib.load(f)
    return {}


def save(cfg: dict[str, Any]) -> None:
    cp = config_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        tomli_w.dump(cfg, f)


def db_enabled() -> bool:
    env = os.environ.get("TIMBRE_DB_ENABLED")
    if env is not None:
        return env.lower() not in _FALSEY
    return bool(load().get("db", {}).get("enabled", True))


def db_path() -> Path:
    env = os.environ.get("TIMBRE_DB")
    if env:
        return Path(env).expanduser()
    cfg = load().get("db", {}).get("path")
    if cfg:
        return Path(cfg).expanduser()
    return default_db_path()


def effective() -> dict[str, Any]:
    """The fully-resolved settings actually in effect (env + file + defaults)."""
    return {"db": {"enabled": db_enabled(), "path": str(db_path())}}


# Known dotted keys and how to coerce their string values from `config set`.
_COERCE = {
    "db.enabled": lambda v: v.lower() not in _FALSEY,
    "db.path": lambda v: str(Path(v).expanduser()),
}


def set_value(dotted_key: str, value: str) -> Any:
    """Set a dotted config key (e.g. ``db.enabled``) and persist. Returns the
    stored value."""
    from .errors import BadInputError

    if dotted_key not in _COERCE:
        raise BadInputError(
            f"unknown config key '{dotted_key}' (known: {', '.join(sorted(_COERCE))})"
        )
    coerced = _COERCE[dotted_key](value)
    section, _, field = dotted_key.partition(".")
    cfg = load()
    cfg.setdefault(section, {})[field] = coerced
    save(cfg)
    return coerced

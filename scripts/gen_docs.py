#!/usr/bin/env python3
"""Generate ``docs/reference.md`` from code — the single source of truth.

The vocabularies, the CLI surface, and the HTTP routes all live in code; this
script imports them and renders the reference, so the docs can never drift from
what timbre actually does. Three sources:

  * **Vocabulary** — ``timbre/recognize/types.py`` (kinds, per-kind categories,
    instruments) plus the ``Tags`` dataclass.
  * **CLI** — introspected from the Click command tree in ``timbre/cli.py``.
  * **HTTP** — the ``ROUTES`` table in ``timbre/server.py``.

Usage::

    python scripts/gen_docs.py            # write docs/reference.md
    python scripts/gen_docs.py --check    # exit 1 if the file is stale (CI)

When ``--check`` fails, run the script with no args and commit the result.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "reference.md"

# Import from the package (the script runs against the installed/dev package).
from timbre.api import Tags  # noqa: E402
from timbre.cli import cli  # noqa: E402
from timbre.recognize.types import (  # noqa: E402
    INSTRUMENT_VOCAB,
    KINDS,
    LOOP_CATEGORIES,
    ONESHOT_CATEGORIES,
    RECORDING_CATEGORIES,
)
from timbre.server import ROUTES  # noqa: E402

_HEADER = """<!--
  GENERATED FILE — do not edit by hand.
  Source: scripts/gen_docs.py (imports the vocab, CLI, and HTTP route tables).
  Regenerate with:  python scripts/gen_docs.py
  CI fails if this file is out of date (python scripts/gen_docs.py --check).
-->

# timbre reference

Auto-generated from code. For tutorials and recipes see [`README.md`](../README.md);
for architecture and internals see [`CLAUDE.md`](../CLAUDE.md).
"""


def _md_list(items) -> str:
    return ", ".join(f"`{i}`" for i in items)


def _vocab_section() -> str:
    lines = ["## Vocabulary", ""]
    lines.append(f"**Kinds:** {_md_list(KINDS)}")
    lines.append("")
    lines.append("**Categories** (per kind — the valid `category` values):")
    lines.append("")
    lines.append("| Kind | Categories |")
    lines.append("|---|---|")
    for kind, cats in (
        ("one-shot", ONESHOT_CATEGORIES),
        ("loop", LOOP_CATEGORIES),
        ("recording", RECORDING_CATEGORIES),
    ):
        lines.append(f"| `{kind}` | {_md_list(cats)} |")
    lines.append("")
    lines.append(f"**Instruments** (shared across all kinds): {_md_list(INSTRUMENT_VOCAB)}")
    lines.append("")
    return "\n".join(lines)


def _tags_section() -> str:
    lines = ["## `Tags` schema", "", "| Field | Type | Default |", "|---|---|---|"]
    for f in dataclasses.fields(Tags):
        if f.default is not dataclasses.MISSING:
            default = repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f"{f.default_factory()!r}"
        else:
            default = "—"
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        lines.append(f"| `{f.name}` | `{type_name}` | `{default}` |")
    lines.append("")
    return "\n".join(lines)


def _format_param(p: click.Parameter) -> str | None:
    if isinstance(p, click.Argument):
        return f"`{p.name.upper()}` — positional argument"
    if isinstance(p, click.Option):
        flags = ", ".join(f"`{o}`" for o in p.opts)
        help_text = (p.help or "").strip()
        return f"{flags}{f' — {help_text}' if help_text else ''}"
    return None


def _walk_command(name: str, cmd: click.Command, depth: int, lines: list[str]) -> None:
    heading = "#" * min(depth, 6)
    full = name.strip()
    lines.append(f"{heading} `timbre {full}`")
    lines.append("")
    if cmd.help:
        lines.append(cmd.help.strip().split("\n\n")[0])
        lines.append("")
    if isinstance(cmd, click.Group):
        for sub_name in sorted(cmd.commands):
            _walk_command(f"{full} {sub_name}", cmd.commands[sub_name], depth + 1, lines)
        return
    params = [_format_param(p) for p in cmd.params]
    params = [p for p in params if p]
    if params:
        for p in params:
            lines.append(f"- {p}")
        lines.append("")


def _cli_section() -> str:
    lines = ["## CLI", ""]
    lines.append("Every command takes `--json` and returns "
                 "`{\"ok\": true, \"data\": …}` or `{\"ok\": false, \"error\": …}`.")
    lines.append("")
    for name in sorted(cli.commands):
        _walk_command(name, cli.commands[name], 3, lines)
    return "\n".join(lines)


def _http_section() -> str:
    lines = [
        "## HTTP API",
        "",
        "Served by `timbre serve`. CORS is wide open (`*`) — meant to run locally "
        "next to the client app.",
        "",
        "| Method | Endpoint | Description |",
        "|---|---|---|",
    ]
    for method, path, summary in ROUTES:
        lines.append(f"| `{method}` | `{path}` | {summary} |")
    lines.append("")
    return "\n".join(lines)


def render() -> str:
    parts = [
        _HEADER,
        _vocab_section(),
        _tags_section(),
        _cli_section(),
        _http_section(),
    ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if docs/reference.md is out of date.")
    args = ap.parse_args()

    content = render()
    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != content:
            print("docs/reference.md is out of date — run: python scripts/gen_docs.py",
                  file=sys.stderr)
            return 1
        print("docs/reference.md is up to date.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

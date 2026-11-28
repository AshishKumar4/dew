"""XLA's flags read as XLA reads XLA_FLAGS (xla/parse_flags_from_env.cc's `ParseArgvFromString`): the one
lexer Dew's flag readers (`dew._devices`) and the test suite's lane (tests/lane_environment.py, which loads
this file before anything imports Dew or JAX) take. Standard library alone, so importing it opens nothing."""

from __future__ import annotations

SPACE = " \t\r\n"
"""What XLA reads as space between flags (parse_flags_from_env.cc's `kWS`): these four alone, never another
character Python counts as whitespace."""


def spans(flags: str) -> list[tuple[str, str]]:
    """`flags` (an XLA_FLAGS value) as XLA splits it: each flag its name (`--xla_...`) and its whole span as
    written, a quoted value (`--name="a b"` or `'a b'`, a backslash escaping within double quotes) whole with
    its quotes, an unquoted one to the next of XLA's spaces (`SPACE`); read up to the first span that is no
    flag, as XLA gives up there, the rest kept as one span of no name."""
    found, at, end = [], 0, len(flags)
    while at < end and flags[at] in SPACE:
        at += 1
    while at < end and flags[at] == "-":
        start = at
        while at < end and flags[at].isascii() and (flags[at] in "-_" or flags[at].isalnum()):
            at += 1
        name = flags[start:at]
        if at + 1 < end and flags[at] == "=" and flags[at + 1] in "'\"":
            quote, at = flags[at + 1], at + 2
            while at < end and flags[at] != quote:
                at += 2 if quote == '"' and flags[at] == "\\" and at + 1 < end else 1
            at = min(at + 1, end)
        else:
            while at < end and flags[at] not in SPACE:
                at += 1
        found.append((name, flags[start:at]))
        while at < end and flags[at] in SPACE:
            at += 1
    if at < end:
        found.append(("", flags[at:]))
    return found


def value(name: str, span: str) -> str:
    """The value flag `name`'s `span` (`spans`') carries as XLA hands it on: a bare flag `true`, a quoted
    value without its quotes, a backslash within double quotes taking the character after it."""
    if not span.startswith(f"{name}="):
        return "true"
    written = span[len(name) + 1:]
    if written[:1] not in ("'", '"'):
        return written
    quote, body = written[0], written[1:]
    if body.endswith(quote):
        body = body[:-1]
    if quote == "'":
        return body
    out, at = [], 0
    while at < len(body):
        at += body[at] == "\\" and at + 1 < len(body)
        out.append(body[at])
        at += 1
    return "".join(out)


def flag(flags: str, name: str) -> str | None:
    """The value `--<name>` carries in `flags` as XLA reads it, its last occurrence winning, or None where it
    is absent: a quoted value naming another flag is no flag of its own."""
    found = None
    for spanned, span in spans(flags):
        if spanned == f"--{name}":
            found = value(spanned, span)
    return found


def without(flags: str, *names: str) -> str:
    """`flags` without each flag of `names` (`--xla_...`), whatever value it carries, each read whole."""
    return " ".join(span for name, span in spans(flags) if name not in names)


def each_once(flags: str) -> str:
    """`flags` with each flag named once, its last value: a process configured again states each flag it sets
    once, not again beside the inherited one."""
    named: dict[str, str] = {}
    for name, span in spans(flags):
        named.pop(name or span, None)
        named[name or span] = span
    return " ".join(named.values())

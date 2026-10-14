"""Published reference code, fetched at a pinned URL and run as written.

The fixture generators take a reference's own definitions rather than a
transcription of them: `selected` picks the named classes, functions and
assignments of a source file (or the members of one class in it) in file
order, and refuses a name the file no longer defines, so a moved or renamed
upstream definition fails the generator instead of leaving a stale one in
scope. `defined_in` (`definitions` for a URL) runs them in the generator's
scope; a generator that retargets a dtype, drops a decorator or builds a
class around them does so between `selected` and `ran`. `simple_statements`
takes single lines out of a published function. `verified` keeps a
reference file on disk, fetched once and checked against its pinned SHA-256
on every use.
"""

from __future__ import annotations

import ast
import hashlib
import urllib.request
from collections.abc import Iterable
from pathlib import Path

TIMEOUT = 120
"""Seconds a fetch may take."""
CACHE = Path.home() / ".cache" / "dew" / "upstream"
"""Where `verified` keeps fetched reference files, each under its own source's path."""


def fetched(url: str) -> str:
    """The text of the file at `url`."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read().decode()


def verified(url: str, local: Path, digest: str) -> Path:
    """`local`, the file at `url` fetched there if it is not yet, refused
    unless its SHA-256 is `digest`."""
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            local.write_bytes(response.read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != digest:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
    return local


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


def selected(text: str, names: Iterable[str], *, within: str | None = None,
             filename: str = "<reference>") -> list[ast.stmt]:
    """The top-level statements of `text` defining `names` (or the members of
    its class `within`), in file order. Every name must be defined there."""
    body = ast.parse(text).body
    if within is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == within).body
    wanted = set(names)
    chosen = [node for node in body if _defined(node) & wanted]
    missing = wanted - set().union(*(_defined(node) for node in chosen))
    assert not missing, f"{filename} defines none of {sorted(missing)}"
    return chosen


def simple_statements(text: str, node: ast.AST, starts: Iterable[str]) -> list[str]:
    """The source of the first statement under `node` (`text`'s), in walk
    order, holding no block of its own, that begins with each of `starts`:
    lines a generator runs out of a function too long to run whole. Every
    one must be there."""
    starts = list(starts)
    found: dict[str, str] = {}
    blocks = ast.For | ast.If | ast.With | ast.While | ast.Try
    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and not isinstance(child, blocks):
            source = ast.get_source_segment(text, child) or ""
            for start in starts:
                if source.startswith(start):
                    found.setdefault(start, source)
    missing = [start for start in starts if start not in found]
    assert not missing, f"the source no longer holds {missing}"
    return [found[start] for start in starts]


def ran(statements: Iterable[ast.stmt], scope: dict, filename: str = "<reference>") -> dict:
    """Run `statements` in `scope`, in order, and return `scope`."""
    module = ast.fix_missing_locations(ast.Module(body=list(statements), type_ignores=[]))
    exec(compile(module, filename, "exec"), scope)
    return scope


def defined_in(text: str, names: Iterable[str], scope: dict, *, within: str | None = None,
               filename: str = "<reference>") -> dict:
    """Run `selected(text, names, within=within)` in `scope`, and return it."""
    return ran(selected(text, names, within=within, filename=filename), scope, filename)


def definitions(url: str, names: Iterable[str], scope: dict, *, within: str | None = None) -> dict:
    """`defined_in` over the file at `url`."""
    return defined_in(fetched(url), names, scope, within=within, filename=url)

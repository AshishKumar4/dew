"""Published reference code, fetched at a pinned URL and run as written.

The fixture generators take a reference's own definitions rather than a
transcription of them: `definitions` (`defined_in` for text in hand) runs the named classes,
functions and assignments of a source file (or the members of one class in it) in the
generator's scope, in file order, and refuses a name the file no longer
defines, so a moved or renamed upstream definition fails the generator
instead of leaving a stale one in scope.
"""

from __future__ import annotations

import ast
import urllib.request


def fetched(url: str) -> str:
    """The text of the file at `url`."""
    return urllib.request.urlopen(url).read().decode()


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


def definitions(url: str, names: set[str] | tuple[str, ...], scope: dict, *, within: str | None = None) -> dict:
    """`defined_in` over the file at `url`."""
    return defined_in(fetched(url), names, scope, within=within, filename=url)


def defined_in(text: str, names: set[str] | tuple[str, ...], scope: dict, *, within: str | None = None,
               filename: str = "<reference>") -> dict:
    """Run the top-level definitions of `text` named `names` (or the members
    of its class `within`, as top-level definitions) in `scope`, in file
    order, and return `scope`. Every name must be defined there."""
    body = ast.parse(text).body
    if within is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == within).body
    wanted = set(names)
    chosen = [node for node in body if _defined(node) & wanted]
    missing = wanted - set().union(*(_defined(node) for node in chosen))
    assert not missing, f"{filename} defines none of {sorted(missing)}"
    for node in chosen:
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope

"""The duplicate-code gate: no code is written twice. It reads code two ways.

- As text, with pylint's similarity check: a block of more than six lines two files share, once imports,
  signatures, docstrings and comments are set aside, fails.
- As structure: two functions in different files whose bodies are equal once every name is renamed in the
  order it is first used, every constant is read as its type, and docstrings and annotations are dropped
  (at least `FUNCTION_NODES` syntax nodes), fail. So do six consecutive statements of two statement lists in
  different files, equal the same way (at least `WINDOW_NODES` nodes). A class's annotated fields and a run
  of calls to one function (a parser's arguments, a registry's entries) are declarations, not windows.

    python tools/duplicates.py [--base=REF] PATH...

Every hit is printed, and any hit fails. With `--base`, a structural hit the merge-base of REF and HEAD
already had, the same copies by file and enclosing function, is printed and does not fail: main still holds
the duplicates its first run found (ledger 896-950), and a head is held to adding none. CI's lint job and
the lanes' lint gate run it with main as the base until main's last one goes, when tools/check.sh runs it
with none. tests/test_duplicates.py runs it on copied code.

It runs the test environment's pylint, which pyproject.toml's test extra pins.
"""

import argparse
import ast
import hashlib
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pylint.lint import Run

ARGUMENTS = ("--disable=all", "--enable=duplicate-code", "--min-similarity-lines=6", "--ignore-imports=y",
             "--ignore-signatures=y", "--ignore-docstrings=y", "--ignore-comments=y", "--score=n")
"""The text reading's one statement of the check, which the gate and its test both run."""

FUNCTION_NODES = 40
"""The fewest syntax nodes a function's body has for its copies to count: below it, a body says too little
(a property, a one-line delegation) for two to be one logic."""
WINDOW = 6
"""Consecutive statements a window holds, as pylint's six lines."""
WINDOW_NODES = 60
"""The fewest syntax nodes six statements hold for their copies to count."""

BODIES = ("body", "orelse", "finalbody")
DROPPED = {"ctx", "type_comment", "decorator_list", "returns", "annotation", "type_params", "kind"}
NAME = "\0"
"""Marks an identifier among a statement's tokens, for each window to number in its own order."""


@dataclass(frozen=True)
class Place:
    file: str
    owner: str
    """The qualified name of the function or class the code is in; "" at a module's top level."""
    line: int


@dataclass(frozen=True)
class Hit:
    kind: str
    """"function" or "statements"."""
    places: tuple[Place, ...]

    def copies(self) -> frozenset[tuple[str, str]]:
        return frozenset((place.file, place.owner) for place in self.places)


def declaration(statement: ast.stmt) -> bool:
    """A docstring, or a bare string standing as one."""
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str))


def stripped(node: ast.AST, field: str, statements: list[ast.stmt]) -> list[ast.stmt]:
    """`node`'s statement list without its docstrings, nor a class's annotated fields."""
    fields = isinstance(node, ast.ClassDef) and field == "body"
    return [statement for statement in statements
            if not declaration(statement) and not (fields and isinstance(statement, ast.AnnAssign))]


def tokens(node: ast.AST, out: list[str]) -> int:
    """`node` as tokens into `out`, every identifier marked with `NAME` and every constant its type (None,
    booleans and the ellipsis as themselves); its count of syntax nodes."""
    out.append(type(node).__name__)
    count = 1
    for field, value in ast.iter_fields(node):
        if field in DROPPED:
            continue
        out.append(field)
        if field in BODIES and isinstance(value, list):
            value = stripped(node, field, value)
        if isinstance(node, ast.Constant) and field == "value":
            out.append(repr(value) if value is None or value is Ellipsis or isinstance(value, bool)
                       else type(value).__name__)
        elif isinstance(value, ast.AST):
            count += tokens(value, out)
        elif isinstance(value, list):
            out.append(f"[{len(value)}")
            for item in value:
                if isinstance(item, ast.AST):
                    count += tokens(item, out)
                elif isinstance(item, str):
                    out.append(NAME + item)
                else:
                    out.append(repr(item))
        elif isinstance(value, str):
            out.append(NAME + value)
        else:
            out.append(repr(value))
    return count


def digest(parts: Sequence[list[str]]) -> str:
    """The parts' tokens with each identifier numbered in the order it is first used."""
    numbered: dict[str, str] = {}
    text = []
    for part in parts:
        text.extend(numbered.setdefault(token, f"n{len(numbered)}") if token.startswith(NAME) else token
                    for token in part)
        text.append(";")
    return hashlib.sha256("\x1f".join(text).encode()).hexdigest()


def declarative(window: Sequence[ast.stmt]) -> bool:
    """A run of calls to one function: data, not logic."""
    calls = [statement.value for statement in window
             if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)]
    return len(calls) == len(window) and len({ast.dump(call.func) for call in calls}) == 1


def units(file: str, tree: ast.Module) -> Iterator[tuple[str, str, Place, int]]:
    """(kind, digest, place, nodes) for each function of `tree` and each window of its statement lists."""
    shapes: dict[int, tuple[list[str], int]] = {}

    def shape(statement: ast.stmt) -> tuple[list[str], int]:
        if id(statement) not in shapes:
            out: list[str] = []
            shapes[id(statement)] = (out, tokens(statement, out))
        return shapes[id(statement)]

    def visit(node: ast.AST, owner: str) -> Iterator[tuple[str, str, Place, int]]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = f"{owner}.{node.name}" if owner else node.name
        for field in BODIES:
            statements = getattr(node, field, None)
            if not (isinstance(statements, list) and statements and isinstance(statements[0], ast.stmt)):
                continue
            body = stripped(node, field, statements)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and field == "body" and body:
                nodes = sum(shape(statement)[1] for statement in body)
                if nodes >= FUNCTION_NODES:
                    place = Place(file, owner, node.lineno)
                    yield "function", digest([shape(s)[0] for s in body]), place, nodes
            for start in range(len(body) - WINDOW + 1):
                window = body[start:start + WINDOW]
                nodes = sum(shape(statement)[1] for statement in window)
                if nodes >= WINDOW_NODES and not declarative(window):
                    yield ("statements", digest([shape(s)[0] for s in window]),
                           Place(file, owner, window[0].lineno), nodes)
        for handler in getattr(node, "handlers", None) or ():
            yield from visit(handler, owner)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.match_case)):
                yield from visit(child, owner)

    yield from visit(tree, "")


def structural(sources: dict[str, str]) -> list[Hit]:
    """The functions and windows `sources` (file to text) write in two files or more."""
    found: dict[tuple[str, str], list[Place]] = defaultdict(list)
    for file, text in sorted(sources.items()):
        for kind, key, place, _ in units(file, ast.parse(text, filename=file)):
            found[kind, key].append(place)
    hits = [Hit(kind, tuple(places)) for (kind, _), places in found.items()
            if len({place.file for place in places}) > 1]
    # A window inside copies a function hit already names is that hit.
    functions = [hit.copies() for hit in hits if hit.kind == "function"]
    return [hit for hit in hits
            if hit.kind == "function" or not any(hit.copies() <= copies for copies in functions)]


def python_files(paths: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for path in map(Path, paths):
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return sorted(files)


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout


def new(hits: list[Hit], base: str) -> list[Hit]:
    """The hits `base`'s merge-base with HEAD did not have: a copy no group of the same kind there held, by
    file and enclosing function. A hit whose files are all as the merge-base has them it had."""
    merged = git("merge-base", "HEAD", base).strip()
    changed = set(git("diff", "--name-only", merged, "--").splitlines())
    candidates = [hit for hit in hits if any(place.file in changed for place in hit.places)]
    files = {place.file for hit in candidates for place in hit.places}
    present = set(git("ls-tree", "-r", "--name-only", merged, "--", *files).splitlines()) if files else set()
    before = structural({file: git("show", f"{merged}:{file}") for file in files & present})
    held = defaultdict(list)
    for hit in before:
        held[hit.kind].append(hit.copies())
    return [hit for hit in candidates if not any(hit.copies() <= copies for copies in held[hit.kind])]


def report(hit: Hit) -> str:
    places = "; ".join(f"{place.file}:{place.line} {place.owner or '<module>'}" for place in hit.places)
    return f"duplicate {hit.kind} in {len({place.file for place in hit.places})} files: {places}"


def main(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="a commit: a structural hit fails if new to its merge-base with HEAD")
    parser.add_argument("paths", nargs="+")
    options = parser.parse_args(arguments)
    text = Run([*ARGUMENTS, *options.paths], exit=False).linter.msg_status
    files = python_files(options.paths)
    hits = structural({str(file): file.read_text() for file in files})
    failing = new(hits, options.base) if options.base else hits
    for hit in hits:
        print(("" if hit in failing else "(held by the base) ") + report(hit))
    print(f"{len(hits)} structural duplicates over {len(files)} files, {len(failing)} failing")
    return 1 if text or failing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

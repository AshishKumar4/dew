"""The duplicate-code gate: no code is written twice. It reads code two ways.

- As text, with pylint's similarity check (`Symilar`, the `symilar` command's): a block of more than six
  lines two files share, once imports, signatures, docstrings and comments are set aside, fails. No pragma
  holds a copy: a `pylint: disable=duplicate-code` is not read. The script strips the files in a pool of
  processes, one a processor, and compares them in one, in the files' order, as pylint's own run does.
- As structure: two functions whose bodies are equal once every name is renamed in the order it is first
  used, every constant is read as its type, and docstrings and annotations are dropped (at least
  `FUNCTION_NODES` syntax nodes), fail, in one file or in two. So do six consecutive statements of two
  statement lists, equal the same way (at least `WINDOW_NODES` nodes). Each copy is named once: the
  windows of one copied run as the run, and a copy that lies inside a larger one as the larger.
  Imports, stubs (a protocol's members), functions that only store their fields (a constructor, as a
  dataclass declares it) and a class's annotated fields are declarations and are not read.
  A run of calls to one function (a parser's arguments, a registry's entries, a config's fields read one
  by one) declares data, so its constants are read as their values: runs alike in shape that declare
  other data are no copy, and a run copied with its data is one.

    python tools/duplicates.py [--base=REF] [--reading=text|structure] PATH...

Every copy is printed, and any fails. `--reading` reads one way alone: tools/check.sh runs the two as gates of
their own (tools/check_gates), side by side, and its `--changed` the structure alone, in seconds.

With `--base`, a structural copy fails only where it is written in more places than at the merge-base of REF
and HEAD, wherever those places are: a copy that moved is held, and one written beside it, or once more,
fails. Main still holds the copies the gate's runs and the
duplication audits found (ledger 896-1287, 1309), and a head is held to adding none. The text reading has no
base: main holds no text copy, so any fails. tools/check.sh (CI's lint job, the lanes' lint gate) runs the
gate held to origin/main until main's last copy goes, then with no base. tests/test_duplicates.py runs it
on copied code.

It runs the test environment's pylint, which pyproject.toml's test extra pins.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forked import forked_map

if TYPE_CHECKING:
    from pylint.checkers.symilar import LineSet

SIMILARITY = {"min_lines": 6, "ignore_comments": True, "ignore_docstrings": True, "ignore_imports": True,
              "ignore_signatures": True}
"""The text reading: pylint's similarity check, as the gate runs it."""

FUNCTION_NODES = 20
"""The fewest syntax nodes a function's body has for its copies to count: below it, a body says too little
(a property, a one-line delegation) for two to be one logic. The smallest copy found that shares knowledge,
two context managers' cleanup (training/distributed.py's and tracker.py's), has 21; two constructors that
only store their fields (ledger 934) are declarations, not logic."""
WINDOW = 6
"""Consecutive statements a window holds, as pylint's six lines."""
WINDOW_NODES = 28
"""The fewest syntax nodes six statements hold for their copies to count; the first run's smallest copy of
six statements (ledger 920) has 28."""

BODIES = ("body", "orelse", "finalbody")
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
DROPPED = {"ctx", "type_comment", "decorator_list", "returns", "annotation", "type_params", "kind"}
NAME = "\0"
"""Marks an identifier among a statement's tokens, for each unit to number in its own order."""


@dataclass(frozen=True)
class Place:
    file: str
    owner: str
    """The qualified name of the function or class the code is in; "" at a module's top level."""
    line: int
    end: int
    """The last line the code spans."""

    def holds(self, other: "Place") -> bool:
        return self.file == other.file and self.line <= other.line and other.end <= self.end

    def meets(self, other: "Place") -> bool:
        return self.file == other.file and self.line <= other.end and other.line <= self.end


@dataclass(frozen=True)
class Unit:
    kind: str
    """"function" or "statements"."""
    key: str
    """Its digest: the units equal up to renaming share it."""
    place: Place


@dataclass(frozen=True)
class Hit:
    kind: str
    key: str
    places: tuple[Place, ...]
    """Where the code is written, in file and line order, no two overlapping."""


def docstring(statement: ast.stmt) -> bool:
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str))


def data(value: ast.expr) -> bool:
    """A name, a constant, an empty container, or a tuple of them: a value stored as it is."""
    if isinstance(value, ast.Tuple):
        return all(data(element) for element in value.elts)
    if isinstance(value, (ast.List, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Dict):
        return not value.keys
    return isinstance(value, (ast.Name, ast.Constant))


def field(target: ast.expr) -> bool:
    """`self.name`, or a tuple of them."""
    if isinstance(target, ast.Tuple):
        return all(field(element) for element in target.elts)
    return (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id == "self")


def stores(statement: ast.stmt) -> bool:
    """A statement that only stores values in the instance's fields, or hands its arguments, as they are, to
    the parent class's constructor."""
    if isinstance(statement, ast.Assign):
        return all(field(target) for target in statement.targets) and data(statement.value)
    call = statement.value if isinstance(statement, ast.Expr) else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "__init__"):
        return False
    parent = call.func.value
    return (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "super"
            and all(isinstance(argument, (ast.Name, ast.Attribute)) for argument in parent.args)
            and not parent.keywords and all(data(argument) for argument in call.args)
            and all(data(keyword.value) for keyword in call.keywords))


def declaration(statement: ast.stmt, in_class: bool) -> bool:
    """A docstring or a bare string standing as one, an import, a stub (a function whose body is only `...`
    or `pass`), a function that only stores its fields (a constructor as a dataclass declares it), a bare
    annotation, or a class's annotated field."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = [line for line in statement.body if not docstring(line)]
        stub = len(body) <= 1 and all(
            isinstance(line, ast.Pass) or (isinstance(line, ast.Expr) and isinstance(line.value, ast.Constant)
                                           and line.value.value is Ellipsis) for line in body)
        return stub or all(stores(assigned(line)) for line in body)
    if isinstance(statement, ast.AnnAssign):
        return in_class or statement.value is None
    return docstring(statement) or isinstance(statement, (ast.Import, ast.ImportFrom))


def assigned(statement: ast.stmt) -> ast.stmt:
    """A valued annotated assignment as the assignment it is: an annotation hides no copy."""
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return ast.copy_location(ast.Assign(targets=[statement.target], value=statement.value), statement)
    return statement


def callee(statement: ast.stmt) -> str | None:
    """The function a statement that is only a call, or only names a call's result, calls, as written."""
    if isinstance(statement, (ast.Expr, ast.Assign)) and isinstance(statement.value, ast.Call):
        return ast.dump(statement.value.func)
    return None


def normalized(tree: ast.Module) -> set[int]:
    """`tree` as the gate reads it, rewritten in place: its declarations dropped and its valued annotated
    assignments plain. Returns the ids of the statements in runs of calls to one function, which declare
    data."""
    data: set[int] = set()
    for node in ast.walk(tree):
        for field in BODIES:
            statements = getattr(node, field, None)
            if not (isinstance(statements, list) and statements and isinstance(statements[0], ast.stmt)):
                continue
            kept = [assigned(statement) for statement in statements
                    if not declaration(statement, isinstance(node, ast.ClassDef))]
            setattr(node, field, kept)
            for first, second in pairwise(kept):
                if callee(first) is not None and callee(first) == callee(second):
                    data.update((id(first), id(second)))
    return data


class Shapes:
    """Each statement of one normalized tree as tokens, with its count of syntax nodes, read once."""

    def __init__(self, data: set[int]):
        self.data = data
        self.read: dict[int, tuple[list[str], int]] = {}

    def of(self, statement: ast.stmt) -> tuple[list[str], int]:
        if id(statement) not in self.read:
            out: list[str] = []
            self.read[id(statement)] = (out, self.tokens(statement, out, id(statement) in self.data))
        return self.read[id(statement)]

    def tokens(self, node: ast.AST, out: list[str], data: bool) -> int:
        """`node` as tokens into `out`, every identifier marked with `NAME` and every constant its type
        (None, booleans and the ellipsis as themselves), or its value where `node` declares data; its count
        of syntax nodes."""
        out.append(type(node).__name__)
        count = 1
        for field, value in ast.iter_fields(node):
            if field in DROPPED:
                continue
            out.append(field)
            if isinstance(node, ast.Constant) and field == "value":
                literal = data or value is None or value is Ellipsis or isinstance(value, bool)
                out.append(repr(value) if literal else type(value).__name__)
            elif isinstance(value, ast.AST):
                count += self.tokens(value, out, data)
            elif isinstance(value, list):
                out.append(f"[{len(value)}")
                for item in value:
                    if isinstance(item, ast.stmt):
                        part, nodes = self.of(item)
                        out.extend(part)
                        count += nodes
                    elif isinstance(item, ast.AST):
                        count += self.tokens(item, out, data)
                    elif isinstance(item, str):
                        out.append(NAME + item)
                    else:
                        out.append(repr(item))
            elif isinstance(value, str):
                out.append(NAME + value)
            else:
                out.append(repr(value))
        return count

    def digest(self, statements: Sequence[ast.stmt]) -> tuple[str, int]:
        """The statements' tokens, each identifier numbered in the order it is first used, hashed; and
        their count of syntax nodes."""
        numbered: dict[str, str] = {}
        text = []
        nodes = 0
        for statement in statements:
            part, count = self.of(statement)
            nodes += count
            text.extend(numbered.setdefault(token, f"n{len(numbered)}") if token.startswith(NAME) else token
                        for token in part)
            text.append(";")
        return hashlib.sha256("\x1f".join(text).encode()).hexdigest(), nodes


def segments(statements: list[ast.stmt]) -> Iterator[list[ast.stmt]]:
    """`statements`' runs between definitions: a window reads logic, and a definition is a unit of its own."""
    run: list[ast.stmt] = []
    for statement in statements:
        if isinstance(statement, DEFINITIONS):
            if run:
                yield run
            run = []
        else:
            run.append(statement)
    if run:
        yield run


def units(file: str, tree: ast.Module) -> Iterator[Unit]:
    """Each function of `tree` and each window of six statements of its statement lists, at its size."""
    shapes = Shapes(normalized(tree))

    def visit(node: ast.AST, owner: str) -> Iterator[Unit]:
        if isinstance(node, DEFINITIONS):
            owner = f"{owner}.{node.name}" if owner else node.name
        for field in BODIES:
            statements = getattr(node, field, None)
            if not (isinstance(statements, list) and statements and isinstance(statements[0], ast.stmt)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and field == "body":
                key, nodes = shapes.digest(statements)
                if nodes >= FUNCTION_NODES:
                    end = node.end_lineno or node.lineno
                    yield Unit("function", key, Place(file, owner, node.lineno, end))
            for run in segments(statements):
                for start in range(len(run) - WINDOW + 1):
                    window = run[start:start + WINDOW]
                    key, nodes = shapes.digest(window)
                    if nodes >= WINDOW_NODES:
                        last = window[-1]
                        yield Unit("statements", key,
                                   Place(file, owner, window[0].lineno, last.end_lineno or last.lineno))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                yield from visit(child, owner)

    yield from visit(tree, "")


def distinct(places: list[Place]) -> list[Place]:
    """`places` in file and line order, less each that overlaps the one kept before it: a window that recurs
    within one run of statements is written once there."""
    kept: list[Place] = []
    for place in sorted(places, key=lambda place: (place.file, place.line)):
        if not (kept and kept[-1].meets(place)):
            kept.append(place)
    return kept


def found(sources: dict[str, str]) -> list[Hit]:
    """Every unit `sources` (file to text) write at two places or more: each a copy, unit by unit."""
    grouped: dict[tuple[str, str], list[Place]] = defaultdict(list)
    for file, text in sorted(sources.items()):
        for unit in units(file, ast.parse(text, filename=file)):
            grouped[unit.kind, unit.key].append(unit.place)
    hits = [Hit(kind, key, tuple(apart)) for (kind, key), places in grouped.items()
            if len(apart := distinct(places)) > 1]
    return sorted(hits, key=lambda hit: [(place.file, place.line) for place in hit.places])


def covers(outer: Hit, inner: Hit) -> bool:
    """Every place of `inner` lies inside one of `outer`'s."""
    return all(any(place.holds(each) for place in outer.places) for each in inner.places)


def named(hits: list[Hit]) -> list[Hit]:
    """`hits` (as `found` orders them) as the copies they are: the windows of one copied run, each place
    overlapping its counterpart's, as one; and a copy whose every place lies inside another's as that one.
    Of copies on the same lines (a function written on one line and its statements), the function is named."""
    copies: list[Hit] = []
    for hit in hits:
        for index, run in enumerate(copies):
            if hit.kind == run.kind == "statements" and len(hit.places) == len(run.places):
                pairs = list(zip(run.places, hit.places, strict=True))
                if all(a.owner == b.owner and a.meets(b) for a, b in pairs):
                    copies[index] = Hit(run.kind, run.key, tuple(
                        Place(a.file, a.owner, min(a.line, b.line), max(a.end, b.end)) for a, b in pairs))
                    break
        else:
            copies.append(hit)

    def inside(index: int, hit: Hit) -> bool:
        return any(covers(other, hit) and (not covers(hit, other) or (other.kind, at) < (hit.kind, index))
                   for at, other in enumerate(copies) if at != index)

    return [hit for index, hit in enumerate(copies) if not inside(index, hit)]


def structural(sources: dict[str, str]) -> list[Hit]:
    """The copies `sources` (file to text) write, named."""
    return named(found(sources))


def python_files(paths: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for path in map(Path, paths):
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return sorted(files)


def git(top: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(top), *arguments], stdout=subprocess.PIPE, text=True,
                          check=True).stdout


def merge_base(top: Path, base: str) -> str:
    return git(top, "merge-base", "HEAD", base).strip()


def new(hits: list[Hit], merged: str, top: Path) -> list[Hit]:
    """The copies of `hits` (as `found` gives them, each file named from the repository's top `top`)
    written in more places than at the commit `merged`. A file the checkout leaves as `merged` has it holds
    the places it holds now; a changed, deleted or new one is read as `merged` has it."""
    # --no-renames: a moved file is named at both ends, so its old content is read as `merged` has it.
    present = set(git(top, "ls-tree", "-r", "--name-only", merged).splitlines())
    # A file the merge-base lacks is new however git sees it now: untracked, ignored or added.
    changed = set(git(top, "diff", "--no-renames", "--name-only", merged, "--").splitlines()) | {
        place.file for hit in hits for place in hit.places if place.file not in present}
    before: dict[tuple[str, str], list[Place]] = defaultdict(list)
    for file in sorted(changed & present):
        if file.endswith(".py"):
            for unit in units(file, ast.parse(git(top, "show", f"{merged}:{file}"), filename=file)):
                before[unit.kind, unit.key].append(unit.place)
    return [hit for hit in hits if len(hit.places) > len(distinct(
        [place for place in hit.places if place.file not in changed] + before[hit.kind, hit.key]))]


def report(hit: Hit) -> str:
    places = "; ".join(f"{place.file}:{place.line}-{place.end} {place.owner or '<module>'}"
                       for place in hit.places)
    return f"duplicate {hit.kind} at {len(hit.places)} places: {places}"


def judged(hits: list[Hit], merged: str, top: Path) -> tuple[list[Hit], list[Hit]]:
    """`hits` (as `found` gives them) as the copies held to the commit `merged` fails and those it holds."""
    failing = new(hits, merged, top)
    return failing, [hit for hit in hits if hit not in failing]


def stripped(file: Path) -> tuple["LineSet", set[int]]:
    """`file`'s lines as the text reading compares them (`SIMILARITY`), and the hashes of their windows of one
    line past `min_lines`, as pylint hashes a window (`hash_lineset`), read in a process of its own (a fork,
    which hashes text as its parent does). pylint reports a run of lines two files share only past
    `min_lines` lines, so the two share such a window."""
    from pylint.checkers.symilar import Symilar, hash_lineset

    reading = Symilar(**SIMILARITY)
    with file.open(encoding="utf-8") as stream:
        reading.append_stream(str(file), stream)
    [lineset] = reading.get_map_data()
    windows, _ = hash_lineset(lineset, SIMILARITY["min_lines"] + 1)
    return lineset, {hash(window) for window in windows}


def text_copies(files: list[Path], forked: bool) -> int:
    """Print pylint's similarity report on the files of `files` that share a window, and return how many
    lines the copies it finds repeat.

pylint compares every two files, hashing both each time, which is quadratic in the files; it reports a
    copy only where two files share a window one line past its minimum (`stripped`). So each file is stripped
    and hashed once, in forked processes when `forked`; files are grouped where they share such a window,
    transitively; and pylint compares each group's files, in the files' order, as it compares them in a run
    over all of them: a group shares no file and no line with another, so no copy in one changes what pylint
    reports in another.
    """
    from pylint.checkers.symilar import Symilar  # its import alone takes seconds, which the structure skips

    read = forked_map(stripped, files) if forked else [stripped(file) for file in files]
    # Each file's group, as the first file it shares a window with names it, merged where two groups meet.
    group = list(range(len(read)))

    def root(index: int) -> int:
        while group[index] != index:
            group[index] = group[group[index]]
            index = group[index]
        return index

    first: dict[int, int] = {}
    for index, (_, windows) in enumerate(read):
        for window in windows:
            group[root(index)] = root(first.setdefault(window, index))
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(read)):
        members[root(index)].append(index)
    repeated = 0
    for indexes in sorted(members.values()):
        if len(indexes) < 2:
            continue
        reading = Symilar(**SIMILARITY)
        reading.combine_mapreduce_data([[read[index][0]] for index in indexes])
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            reading.run()
        # The report's last line: "TOTAL lines=N duplicates=M percent=P".
        totals = re.search(r"^TOTAL lines=\d+ duplicates=(\d+) ", report.getvalue(), re.MULTILINE)
        assert totals is not None, "pylint's similarity report ends with its totals"
        if int(totals.group(1)):
            print(report.getvalue(), end="")
            repeated += int(totals.group(1))
    return repeated


def main(arguments: Sequence[str], forked: bool = False) -> int:
    """The gate, the text read in forked processes when `forked` (`forked_map`), as the script reads it; a
    caller in a process that may hold threads (a test, under JAX) reads it in its own."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="a commit: a structural copy fails if written in more places than at"
                                       " its merge-base with HEAD")
    parser.add_argument("--reading", choices=("text", "structure"),
                        help="one reading alone (tools/check_gates' two gates); both by default")
    parser.add_argument("paths", nargs="+")
    options = parser.parse_args(arguments)
    files = python_files(options.paths)
    # Held to a base, files are named from the repository's top, as git names them, and a base this checkout
    # shares no history with fails here, before either reading.
    top = Path.cwd()
    merged = ""
    if options.base:
        try:
            top = Path(git(top, "rev-parse", "--show-toplevel").strip()).resolve()
            merged = merge_base(top, options.base)
        except subprocess.CalledProcessError:
            # An armada or CI checkout is shallow and has no remote branches: check.sh fetches main and its
            # history first (tools/base_history.sh), as a checkout run by hand does.
            print(f"cannot hold this checkout to the base {options.base} (git's reason on stderr); a shallow"
                  " checkout fetches it first: sh tools/base_history.sh")
            return 2
    started = time.monotonic()
    text = 0
    if options.reading != "structure":
        text = text_copies(files, forked)
    read = time.monotonic()
    hits = [] if options.reading == "text" else found({
        os.path.relpath(file.resolve(), top) if options.base else str(file): file.read_text()
        for file in files})
    failing, held = judged(hits, merged, top) if options.base else (hits, [])
    for hit in named(failing):
        print(report(hit))
    for hit in named(held):
        print("(held by the base) " + report(hit))
    print(f"{len(named(failing))} structural copies failing, {len(named(held))} held by the base, over"
          f" {len(files)} files; text read in {read - started:.0f} s,"
          f" structure in {time.monotonic() - read:.0f} s")
    # The verdict is the copies' own, whatever naming them shows.
    return 1 if text or failing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], forked=True))

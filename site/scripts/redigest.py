"""Keep the landing page's recordings across a change to the code they ran that only renames.

A recording names the sha256 of the program it ran, and the page refuses one whose
program has changed (src/data/recordings.mjs). Recorded outputs are re-recorded
only when a change alters what they show (the bible's rule for renames). So an
API rename that reaches a recorded program updates that program's source and its
recording's digest, and keeps the outputs:

    python site/scripts/redigest.py [--base origin/main] [--head HEAD] --rename OLD=NEW...

where OLD and NEW are Dew's names as an import reaches them, with what holds them:
dew.config.OptimConfig.learning_rate, a class's field or constructor parameter (or a
method), a function's parameter, or an object a module defines (dew.sampling.TextToImage).

The programs and the captures that name their digests are site/snippets/cells.py's,
the page's one declaration of both. For each recording whose program differs from
what its digest names, this checks two things. First, the recording is of `base`'s
program: its digest matches that program. Second, the program is that one renamed,
read as Python parses it:
- every name, value, string, f-string conversion and literal text, operator and
  keyword is unchanged, but for Dew's names the program reaches through an import in
  the scope that uses them (a call's keyword, an attribute, an imported name, a name
  an import binds): a parameter, an assignment or a definition of the same name in
  that scope, or a global or nonlocal declaration of it anywhere, stands for no
  import;
- each of those changes only as the change declares it (`--rename`), a rename of
  Dew's API that the two commits show: its old name defined in src/dew at `base` and
  its new one at `head`, or what holds it renamed so; and none is left as it was
  where the head's Dew no longer defines it;
- where any changes, nothing in the program reads a name from a string or a mapping,
  which no rename follows: no reflective builtin (getattr, globals), no mapping
  unpacked into a call, no string spelling a renamed name.
It then prints each renaming and writes the new digest, and nothing else in the
recording changes, byte for byte. Whether the library's rename keeps what the
recording shows (a label, a value it prints) is the head's own claim. Its reviewer
reads the renamings printed here and the --rename it declares.

A change that is not a renaming (a pinned model moved to another revision included: its run record,
not only its weights, says what the program samples), or a recording that was already of other code,
writes nothing and says to record again (site/scripts/capture_snippets.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import griffe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import cells
import gen_api
import lint_scopes as scopes

SITE = cells.SITE
REPO = SITE.parent
REFLECTION = frozenset({"getattr", "setattr", "delattr", "hasattr", "vars", "globals", "locals", "exec",
                        "eval", "__import__"})
"""The builtins that read or bind a name spelled as a string, which no rename follows."""


def tables(program: ast.Module) -> dict[int, Mapping[str, str]]:
    """What each name imported in the scope of each node of `program` stands for (lint_scopes.scoped: a
    parameter, an assignment or a definition there stands for no import), keyed by the node's identity. A
    name the program declares global or nonlocal anywhere stands for none: the scope that rebinds it is
    another's."""
    rebound = {name for node in ast.walk(program) if isinstance(node, ast.Global | ast.Nonlocal)
               for name in node.names}
    return {id(node): {name: path for name, path in table.items() if name not in rebound}
            for node, table in scopes.scoped(program.body, {}, ())}


class Renaming:
    """How the program at the head renames base's: every name, value and form the same, but for the names of
    Dew's API it uses, each reached through an import in the scope that uses it, which change only as a
    declared rename gives them (`changes`), and in full."""

    def __init__(self, before: ast.Module, after: ast.Module, declared: Declared):
        self.declared = declared
        self.tables = (tables(before), tables(after))
        self.changes: list[tuple[str, str, int]] = []
        """Each name of Dew's API the program changes, as the dotted path it reaches on each side, and its
        line."""

    def reached(self, was: ast.expr, now: ast.expr) -> tuple[str, str]:
        """The dotted paths `was` and `now` reach through the imports of their scopes; "" for one reaching
        none."""
        return (scopes.resolved(was, self.tables[0].get(id(was), {})),
                scopes.resolved(now, self.tables[1].get(id(now), {})))

    def use(self, spelled: tuple[str, str], paths: tuple[str, str], line: int) -> None:
        """One use of a name, spelled on each side as `spelled`, reaching the dotted paths `paths`: a changed
        one is a name of Dew's API renamed as declared; an unchanged one is not a name of base's Dew that the
        head's no longer defines."""
        if spelled[0] != spelled[1] or paths[0] != paths[1]:
            if not self.declared.accounts_for(*paths):
                raise ValueError(f"line {line}: {paths[0] or spelled[0]} became {paths[1] or spelled[1]}, "
                                 "which no --rename of Dew's API declares")
            self.changes.append((*paths, line))
        elif self.declared.vanished(paths[0]):
            raise ValueError(f"line {line}: {paths[0]} is left as it was, which the head's Dew no longer "
                             "defines")

    def compare(self, was: object, now: object, line: int = 0) -> None:
        """`now` is `was` up to the declared renames; a ValueError at the first other difference."""
        if type(was) is not type(now):
            raise ValueError(f"line {line}: {type(was).__name__} became {type(now).__name__}")
        if isinstance(was, list):
            if len(was) != len(now):
                raise ValueError(f"line {line}: {len(was)} items became {len(now)}")
            for before, after in zip(was, now, strict=True):
                self.compare(before, after, line)
        elif isinstance(was, ast.Name):
            line = was.lineno
            self.compare(was.ctx, now.ctx, line)
            self.use((was.id, now.id), self.reached(was, now), line)
        elif isinstance(was, ast.Attribute):
            line = was.lineno
            self.compare([was.value, was.ctx], [now.value, now.ctx], line)
            self.use((was.attr, now.attr), self.reached(was, now), line)
        elif isinstance(was, ast.Call):
            line = was.lineno
            self.compare([was.func, was.args], [now.func, now.args], line)
            if len(was.keywords) != len(now.keywords):
                raise ValueError(f"line {line}: {len(was.keywords)} keywords became {len(now.keywords)}")
            owners = self.reached(was.func, now.func)
            for one, other in zip(was.keywords, now.keywords, strict=True):
                if (one.arg is None) != (other.arg is None):
                    raise ValueError(f"line {line}: a keyword became a mapping unpacked, or one the other")
                if one.arg is not None:
                    paths = tuple(f"{owner}.{name}" if owner else ""
                                  for owner, name in zip(owners, (one.arg, other.arg), strict=True))
                    self.use((one.arg, other.arg), paths, line)
                self.compare(one.value, other.value, line)
        elif isinstance(was, ast.Import | ast.ImportFrom):
            line = was.lineno
            modules = (getattr(was, "module", None) or "", getattr(now, "module", None) or "")
            if getattr(was, "level", 0) != getattr(now, "level", 0) or len(was.names) != len(now.names):
                raise ValueError(f"line {line}: an import became another")
            for one, other in zip(was.names, now.names, strict=True):
                if one.asname != other.asname:
                    raise ValueError(f"line {line}: the name an import binds became another")
                paths = tuple(".".join(part for part in (module, alias.name) if part)
                              for module, alias in zip(modules, (one, other), strict=True))
                self.use((one.name, other.name), paths, line)
        elif isinstance(was, ast.AST):
            line = getattr(was, "lineno", line)
            for field, before in ast.iter_fields(was):
                self.compare(before, getattr(now, field, None), line)
        elif was != now:
            raise ValueError(f"line {line}: {was!r} became {now!r}, not a rename")

    def check(self, after: ast.Module) -> None:
        """Where the program's names of Dew's change, nothing in it reads a name from a string, which no
        rename follows: no reflective builtin, no mapping unpacked into a call, no string spelling a renamed
        name."""
        renamed = {was.rpartition(".")[2] for was, _, _ in self.changes}
        for node in ast.walk(after) if self.changes else ():
            if ((isinstance(node, ast.Name) and node.id in REFLECTION)
                    or (isinstance(node, ast.keyword) and node.arg is None)
                    or (isinstance(node, ast.Constant) and node.value in renamed)):
                raise ValueError(f"line {node.lineno}: {ast.unparse(node)} reads a name from a string or a "
                                 "mapping, which the rename does not follow")

    def printed(self) -> str:
        return ", ".join(sorted({f"{was} -> {now}" for was, now, _ in self.changes}))


class Declared:
    """The renames of Dew's API a change declares (`--rename OLD=NEW`), each name a dotted path an import
    reaches (dew.config.OptimConfig.learning_rate), read in Dew's own API model (site/scripts/gen_api.py)
    at each commit: its old name defined at `base` and its new one at `head`, both where the model says they
    are defined, aliases followed. A ValueError names the first that is not."""

    def __init__(self, renames: list[str], base: str, head: str):
        self.models = (api_model(base), api_model(head)) if renames else None
        self.pairs: set[tuple[str, str]] = set()
        for rename in renames:
            old, _, new = rename.partition("=")
            found = self.at(old, new)
            if not new or found[0] is None or found[1] is None:
                missing = old if found[0] is None else new
                raise ValueError(f"--rename {rename}: {missing} is no name of Dew's API at "
                                 f"{base if missing is old else head}")
            self.pairs.add(found)

    def at(self, old: str, new: str) -> tuple[str | None, str | None]:
        """Where `old` is defined at the base and `new` at the head."""
        return gen_api.canonical(self.models[0], old), gen_api.canonical(self.models[1], new)

    def accounts_for(self, was: str, now: str) -> bool:
        """Whether a use reaching `was` at the base and `now` at the head is a declared rename: of the name
        itself, or of what holds it, the name kept (a renamed class's field)."""
        if not (self.models and was and now) or None in (found := self.at(was, now)):
            return False
        held, _, name = was.rpartition(".")
        holder, _, kept = now.rpartition(".")
        return found in self.pairs or (name == kept and bool(held) and self.accounts_for(held, holder))

    def vanished(self, path: str) -> bool:
        """Whether `path`, a name of base's Dew, is no name of the head's."""
        return bool(self.models and path) and gen_api.canonical(self.models[0], path) is not None and (
            gen_api.canonical(self.models[1], path) is None)


def api_model(revision: str) -> griffe.Module:
    """Dew's API model (gen_api's) of `revision`'s src/, read from its commit."""
    with tempfile.TemporaryDirectory() as root:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", revision, "src"], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", root], input=archive, check=True)
        return gen_api.load(Path(root) / "src")


def renamed(before: str, after: str, declared: Declared) -> Renaming:
    """How `after` renames `before`; a ValueError, naming the first other difference, when it differs
    otherwise."""
    try:
        was, now = ast.parse(before), ast.parse(after)
        renaming = Renaming(was, now, declared)
        renaming.compare(was, now)
        renaming.check(now)
    except SyntaxError as error:
        raise ValueError(f"a program does not parse: {error}") from error
    return renaming


def digest_of(capture: bytes, key: str) -> str:
    """The digest a capture names under `key` (dotted: `meta.cell_sha256`)."""
    value = json.loads(capture)
    for part in key.split("."):
        value = value[part]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main",
                        help="the revision whose programs the recordings ran")
    parser.add_argument("--head", default="HEAD", help="the revision whose Dew the programs now use")
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                        help="a rename of Dew's API the change makes, each name qualified by what holds it")
    options = parser.parse_args()
    try:
        renames = Declared(options.rename, options.base, options.head)
    except ValueError as refusal:
        print(f"{refusal}: nothing written", file=sys.stderr)
        return 1

    def at_base(path: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{options.base}:site/{path}"], check=True,
                              capture_output=True, text=True).stdout

    now, before = cells.programs(), cells.programs(at_base)
    captures = {capture: (SITE / capture).read_bytes() for capture, _ in cells.recordings().values()}
    written = dict(captures)
    refused = []
    for name, (capture, key) in cells.recordings().items():
        recorded, digest = digest_of(captures[capture], key), cells.digest(now[name])
        if recorded == digest:
            continue
        if cells.digest(before[name]) != recorded:
            refused.append(f"{name}: the recording is not of {options.base}'s program either")
            continue
        try:
            renaming = renamed(before[name], now[name], renames)
        except ValueError as difference:
            refused.append(f"{name}: {difference}")
            continue
        field = f'"{key.rsplit(".", 1)[-1]}": "{recorded}"'.encode()
        if written[capture].count(field) != 1:
            refused.append(f"{name}: {capture} names {key} {recorded} other than once")
            continue
        written[capture] = written[capture].replace(field, field.replace(recorded.encode(), digest.encode()))
        print(f"{name}: {renaming.printed()}")
    if refused:
        print("not a rename, so record again (site/scripts/capture_snippets.py):\n  "
              + "\n  ".join(refused), file=sys.stderr)
        return 1
    changed = [capture for capture in written if written[capture] != captures[capture]]
    for capture in changed:
        (SITE / capture).write_bytes(written[capture])
    print(f"{len(changed)} captures re-digested, their outputs kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())

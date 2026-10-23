"""Keep the landing page's recordings across a change to the code they ran that only renames.

A recording names the sha256 of the program it ran, and the page refuses one whose
program has changed (src/data/recordings.mjs). Recorded outputs are re-recorded
only when a change alters what they show (the bible's rule for renames). So an
API rename that reaches a recorded program updates that program's source and its
recording's digest, and keeps the outputs:

    python site/scripts/redigest.py [--base origin/main] [--head HEAD] --rename OLD=NEW...

where OLD and NEW are Dew's names, qualified by what holds them: OptimConfig.learning_rate,
a class's field, method or constructor parameter, a function's parameter, or a module's
own name (TextToImage).

The programs and the captures that name their digests are site/snippets/cells.py's,
the page's one declaration of both. For each recording whose program differs from
what its digest names, this checks two things. First, the recording is of `base`'s
program: its digest matches that program. Second, the program is that one renamed,
read as Python parses it:
- every value, string, f-string conversion and literal text, operator and keyword
  is unchanged;
- an API's names (a call's keyword argument, an attribute, an imported name) may be
  renamed where they are used, each only as the change declares it (`--rename`), a
  rename of Dew's API that the two commits show: its old name defined in src/dew at
  `base` and its new one at `head`;
- a variable (a name, a parameter, a function or class it defines, an import's
  binding) is renamed everywhere, each to its own name, which no other variable of
  the program had, so every use still means what it meant;
- the result compiles.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import cells

SITE = cells.SITE
REPO = SITE.parent
VARIABLE, MEMBER = "variable", "member"
ROLES = {(ast.Name, "id"): VARIABLE, (ast.arg, "arg"): VARIABLE, (ast.FunctionDef, "name"): VARIABLE,
         (ast.AsyncFunctionDef, "name"): VARIABLE, (ast.ClassDef, "name"): VARIABLE,
         (ast.Global, "names"): VARIABLE, (ast.Nonlocal, "names"): VARIABLE,
         (ast.ExceptHandler, "name"): VARIABLE, (ast.MatchAs, "name"): VARIABLE,
         (ast.MatchStar, "name"): VARIABLE, (ast.MatchMapping, "rest"): VARIABLE,
         (ast.keyword, "arg"): MEMBER, (ast.Attribute, "attr"): MEMBER,
         (ast.ImportFrom, "module"): MEMBER, (ast.MatchClass, "kwd_attrs"): MEMBER}
"""Each identifier of the syntax, by node and field, and what it names: a variable of the program, or an API's
name, which the program uses but does not bind."""


class Renaming:
    """The renamings between two parsed programs: each variable to its one new name, injectively, and each
    API name where it is used."""

    def __init__(self, before: Callables, after: Callables):
        self.variables: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.before, self.after = before, after
        """What each program defines that a call's keyword can bind."""

    def name(self, role: str, was: str | None, now: str | None, line: int) -> None:
        if (was is None) != (now is None):
            raise ValueError(f"line {line}: {was!r} became {now!r}")
        if was is None:
            return
        if role == MEMBER:
            if was != now:
                self.members.setdefault(was, set()).add(now)
            return
        if self.variables.setdefault(was, now) != now:
            raise ValueError(f"line {line}: the variable {was!r} became both {self.variables[was]!r} and "
                             f"{now!r}")

    def compare(self, was: object, now: object, line: int = 0) -> None:
        """`now` is `was` up to the renamings; a ValueError at the first other difference."""
        if type(was) is not type(now):
            raise ValueError(f"line {line}: {type(was).__name__} became {type(now).__name__}")
        if isinstance(was, list):
            if len(was) != len(now):
                raise ValueError(f"line {line}: {len(was)} items became {len(now)}")
            for before, after in zip(was, now, strict=True):
                self.compare(before, after, line)
        elif isinstance(was, ast.Call):
            line = was.lineno
            self.compare([was.func, was.args], [now.func, now.args], line)
            if len(was.keywords) != len(now.keywords):
                raise ValueError(f"line {line}: {len(was.keywords)} keywords became {len(now.keywords)}")
            callees = (was.func.id if isinstance(was.func, ast.Name) else None,
                       now.func.id if isinstance(now.func, ast.Name) else None)
            for one, other in zip(was.keywords, now.keywords, strict=True):
                self.keyword(one.arg, other.arg, callees, line)
                self.compare(one.value, other.value, line)
        elif isinstance(was, ast.AST):
            line = getattr(was, "lineno", line)
            if isinstance(was, ast.Import):
                # `import a.b` binds the package a, `import a.b as m` the module a.b: an alias there is no
                # rename.
                for one, other in zip(was.names, now.names, strict=False):
                    if "." in one.name and (one.asname is None) != (other.asname is None):
                        raise ValueError(f"line {line}: an alias on import {one.name} binds another module")
            for field, before in ast.iter_fields(was):
                after = getattr(now, field, None)
                role = ROLES.get((type(was), field))
                if role is not None:
                    names, renames = (item if isinstance(item, list) else [item] for item in (before, after))
                    if len(names) != len(renames):
                        raise ValueError(f"line {line}: {names} became {renames}")
                    for one, other in zip(names, renames, strict=True):
                        self.name(role, one, other, line)
                elif isinstance(was, ast.alias):
                    if field == "name":
                        self.alias(was, now, line)
                else:
                    self.compare(before, after, line)
        elif was != now:
            raise ValueError(f"line {line}: {was!r} became {now!r}, not a rename")

    def keyword(self, was: str | None, now: str | None, callees: tuple[str | None, str | None],
                line: int) -> None:
        """A call's keyword, on each side: a parameter it binds (a variable, which renames with the
        parameter), a key a **kwargs collects (kept as it is), or an API's name. A callable the program
        defines and calls by its name binds its own parameters; one it calls otherwise (a method, an alias)
        may bind any parameter it defines. A keyword binds on both sides or on neither."""
        if was is None or now is None:  # a mapping unpacked into the call: **options
            self.name(MEMBER, was, now, line)
            return
        callee, renamed = callees
        if callee in self.before.signatures:
            bound = (was in self.before.signatures[callee], now in self.after.signatures.get(renamed, set()))
        else:
            bound = (was in self.before.parameters, now in self.after.parameters)
            if not any(bound):
                self.name(MEMBER, was, now, line)
                return
        if bound[0] != bound[1]:
            raise ValueError(f"line {line}: the keyword {was!r} became {now!r}, which binds a parameter "
                             "on one side only")
        if bound[0]:
            self.name(VARIABLE, was, now, line)
        elif was != now:
            raise ValueError(f"line {line}: the key {was!r} a **kwargs collects became {now!r}")

    def alias(self, was: ast.alias, now: ast.alias, line: int) -> None:
        """An import: its name an API name, and the variable it binds (its `as`, else its name's first
        part)."""
        self.name(MEMBER, was.name, now.name, line)
        self.name(VARIABLE, was.asname or was.name.split(".")[0], now.asname or now.name.split(".")[0], line)

    def check(self, after: str, declared: dict[str, str]) -> None:
        """Every variable renamed to a name of its own; every API name renamed as a rename `declared`
        (old name to new, unqualified) gives it; the renamed program compiles."""
        bound = list(self.variables.values())
        if len(set(bound)) != len(bound):
            twice = sorted({name for name in bound if bound.count(name) > 1})
            raise ValueError(f"two variables became one: {', '.join(twice)}")
        for was, nows in sorted(self.members.items()):
            if nows != {declared.get(was)}:
                raise ValueError(f"the API name {was!r} became {' and '.join(sorted(nows))}, which no "
                                 "--rename of Dew's API declares")
        compile(after, "<program>", "exec")

    def printed(self) -> str:
        names = {f"{was} -> {now}" for was, now in self.variables.items() if was != now}
        names |= {f"{was} -> {now}" for was, nows in self.members.items() for now in nows}
        return ", ".join(sorted(names))


class Callables:
    """What a program defines that a call's keyword can bind: each function, class and lambda it names, with
    the parameters a keyword binds (`signatures`: a function's own, but those only by position; a class's
    __init__'s, but its first; a class with no __init__ of its own, none; a name defined twice, those both
    definitions share), and every parameter of every callable it defines (`parameters`)."""

    def __init__(self, program: ast.Module):
        self.signatures: dict[str, set[str]] = {}
        for node in ast.walk(program):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                named, names = node.name, set(keywords(node.args, bound=False))
            elif isinstance(node, ast.ClassDef):
                init = next((item for item in node.body if isinstance(item, ast.FunctionDef)
                             and item.name == "__init__"), None)
                named, names = node.name, set() if init is None else set(keywords(init.args, bound=True))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                for target in assigned(node):
                    self.define(target, set(keywords(node.value.args, bound=False)))
                continue
            else:
                continue
            self.define(named, names)
        self.parameters = {node.arg for node in ast.walk(program) if isinstance(node, ast.arg)}

    def define(self, name: str, names: set[str]) -> None:
        self.signatures[name] = self.signatures[name] & names if name in self.signatures else names


def api(revision: str) -> set[str]:
    """Dew's names at `revision`, read from its src/dew as Python parses it: each module's own (a class,
    function or assignment), each class's (`Class.name`: a method, a field, its constructor's
    parameters) and each function's parameters (`function.name`)."""
    listing = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", revision, "src/dew"],
                             check=True, capture_output=True, text=True).stdout.split()
    asked = "".join(f"{revision}:{path}\n" for path in listing if path.endswith(".py")).encode()
    sources = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], input=asked, check=True,
                             capture_output=True).stdout
    names: set[str] = set()
    while sources:
        header, _, sources = sources.partition(b"\n")
        size = int(header.split()[2])
        tree, sources = ast.parse(sources[:size]), sources[size + 1:]
        names |= {target for node in tree.body for target in assigned(node)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names |= {node.name, *(f"{node.name}.{name}" for name in keywords(node.args, bound=False))}
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
                for item in node.body:
                    held = assigned(item)
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        held = [item.name]
                        if item.name == "__init__":
                            held += keywords(item.args, bound=True)
                    names |= {f"{node.name}.{name}" for name in held}
    return names


def assigned(node: ast.stmt) -> list[str]:
    """The names an assignment statement binds; none for another statement."""
    targets = (node.targets if isinstance(node, ast.Assign)
               else [node.target] if isinstance(node, ast.AnnAssign) else [])
    return [target.id for target in targets if isinstance(target, ast.Name)]


def keywords(arguments: ast.arguments, bound: bool) -> list[str]:
    """The parameters a keyword passes a callable of `arguments`, its first left out where it is `bound`."""
    positional = arguments.args[1:] if bound else arguments.args
    return [arg.arg for arg in (*positional, *arguments.kwonlyargs)]


def declared(renames: list[str], base: str, head: str) -> dict[str, str]:
    """`renames` (OLD=NEW, each a qualified name of Dew's API) as old name to new, unqualified: each one's old
    name defined at `base` and its new one at `head`; a ValueError naming the first that is not."""
    if not renames:
        return {}
    at_base, at_head, found = api(base), api(head), {}
    for rename in renames:
        old, _, new = rename.partition("=")
        if not new or old == new:
            raise ValueError(f"--rename {rename}: OLD=NEW, two names")
        if old not in at_base:
            raise ValueError(f"--rename {rename}: {old} is no name of Dew's API at {base}")
        if new not in at_head:
            raise ValueError(f"--rename {rename}: {new} is no name of Dew's API at {head}")
        was, now = old.rsplit(".", 1)[-1], new.rsplit(".", 1)[-1]
        if found.setdefault(was, now) != now:
            raise ValueError(f"--rename {rename}: {was} is already renamed {found[was]}")
    return found


def renamed(before: str, after: str, renames: dict[str, str]) -> Renaming:
    """How `after` renames `before`; a ValueError, naming the first other difference, when it differs
    otherwise."""
    try:
        was, now = ast.parse(before), ast.parse(after)
        renaming = Renaming(Callables(was), Callables(now))
        renaming.compare(was, now)
        renaming.check(after, renames)
    except SyntaxError as error:
        raise ValueError(f"the renamed program does not compile: {error}") from error
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
        renames = declared(options.rename, options.base, options.head)
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

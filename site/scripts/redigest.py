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
import tempfile
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
VARIABLE, MEMBER = "variable", "member"
ROLES = {(ast.Name, "id"): VARIABLE, (ast.arg, "arg"): VARIABLE, (ast.FunctionDef, "name"): VARIABLE,
         (ast.AsyncFunctionDef, "name"): VARIABLE, (ast.ClassDef, "name"): VARIABLE,
         (ast.Global, "names"): VARIABLE, (ast.Nonlocal, "names"): VARIABLE,
         (ast.ExceptHandler, "name"): VARIABLE, (ast.MatchAs, "name"): VARIABLE,
         (ast.MatchStar, "name"): VARIABLE, (ast.MatchMapping, "rest"): VARIABLE,
         (ast.MatchClass, "kwd_attrs"): MEMBER}
"""Each identifier of the syntax, by node and field, and what it names: a variable of the program, or an API's
name, which the program uses but does not bind. An attribute, an import and a call's keyword are read apart,
each with what holds it."""


class Renaming:
    """The renamings between two parsed programs: each variable to its one new name, injectively, and each
    changed API name with what holds it on each side (`changes`), which a declared rename must account for."""

    def __init__(self, before: ast.Module, after: ast.Module):
        self.variables: dict[str, str] = {}
        self.changes: list[tuple[str, str, int]] = []
        """Each API name the program changes, as the dotted path it reaches on each side ("" where the program
        does not show what holds it), and its line."""
        self.loose: list[tuple[str, int]] = []
        """Each keyword passed to a callable the program does not show, kept as it was, and its line."""
        self.before, self.after = Callables(before), Callables(after)
        self.held = (scopes.imports(before, []), scopes.imports(after, []))

    def name(self, role: str, was: str | None, now: str | None, line: int) -> None:
        if (was is None) != (now is None):
            raise ValueError(f"line {line}: {was!r} became {now!r}")
        if was is None:
            return
        if role == MEMBER:
            if was != now:
                self.changes.append(("", "", line))
            return
        if self.variables.setdefault(was, now) != now:
            raise ValueError(f"line {line}: the variable {was!r} became both {self.variables[was]!r} and "
                             f"{now!r}")

    def member(self, paths: tuple[str, str], spelled: tuple[str, str], line: int) -> None:
        """An API name, spelled on each side as `spelled`, at the dotted path it reaches on each (`paths`)."""
        if spelled[0] != spelled[1] or paths[0].rsplit(".", 1)[0] != paths[1].rsplit(".", 1)[0]:
            self.changes.append((*paths, line))

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
            for one, other in zip(was.keywords, now.keywords, strict=True):
                self.keyword(one.arg, other.arg, (was.func, now.func), line)
                self.compare(one.value, other.value, line)
        elif isinstance(was, ast.Attribute):
            line = was.lineno
            self.compare([was.value, was.ctx], [now.value, now.ctx], line)
            self.member(self.reached(was, now), (was.attr, now.attr), line)
        elif isinstance(was, (ast.Import, ast.ImportFrom)):
            line = was.lineno
            modules = (getattr(was, "module", None) or "", getattr(now, "module", None) or "")
            if getattr(was, "level", 0) != getattr(now, "level", 0) or len(was.names) != len(now.names):
                raise ValueError(f"line {line}: an import became another")
            for one, other in zip(was.names, now.names, strict=True):
                # `import a.b` binds the package a, `import a.b as m` the module a.b: an alias there is no
                # rename.
                if "." in one.name and (one.asname is None) != (other.asname is None):
                    raise ValueError(f"line {line}: an alias on import {one.name} binds another module")
                paths = tuple(".".join(part for part in (module, alias.name) if part)
                              for module, alias in zip(modules, (one, other), strict=True))
                self.member(paths, paths, line)
                bound = (alias.asname or alias.name.split(".")[0] for alias in (one, other))
                self.name(VARIABLE, *bound, line)
        elif isinstance(was, ast.AST):
            line = getattr(was, "lineno", line)
            for field, before in ast.iter_fields(was):
                after = getattr(now, field, None)
                role = ROLES.get((type(was), field))
                if role is not None:
                    names, renames = (item if isinstance(item, list) else [item] for item in (before, after))
                    if len(names) != len(renames):
                        raise ValueError(f"line {line}: {names} became {renames}")
                    for one, other in zip(names, renames, strict=True):
                        self.name(role, one, other, line)
                else:
                    self.compare(before, after, line)
        elif was != now:
            raise ValueError(f"line {line}: {was!r} became {now!r}, not a rename")

    def reached(self, was: ast.expr, now: ast.expr) -> tuple[str, str]:
        """The dotted paths `was` and `now` reach through the programs' imports; "" for one reaching none."""
        return scopes.resolved(was, self.held[0]), scopes.resolved(now, self.held[1])

    def keyword(self, was: str | None, now: str | None, callees: tuple[ast.expr, ast.expr],
                line: int) -> None:
        """A call's keyword, on each side. To a callable the program defines and calls by its name, a
        parameter it binds (a variable, which renames with the parameter) or a key its **kwargs collects
        (kept as it is), binding on both sides or on neither. To one an import reaches, an API's name, under
        that callable. To any other (a method, an alias, a builtin), kept as it is: which parameter it binds
        is not shown."""
        if was is None or now is None:  # a mapping unpacked into the call: **options
            self.name(MEMBER, was, now, line)
            return
        callee, renamed = (node.id if isinstance(node, ast.Name) else None for node in callees)
        if callee in self.before.signatures and callee not in self.held[0]:
            bound = (was in self.before.signatures[callee], now in self.after.signatures.get(renamed, set()))
            if bound[0] != bound[1]:
                raise ValueError(f"line {line}: the keyword {was!r} became {now!r}, which binds a parameter "
                                 "on one side only")
            if bound[0]:
                self.name(VARIABLE, was, now, line)
            elif was != now:
                raise ValueError(f"line {line}: the key {was!r} a **kwargs collects became {now!r}")
            return
        owners = self.reached(*callees)
        if all(owners):
            self.member((f"{owners[0]}.{was}", f"{owners[1]}.{now}"), (was, now), line)
        elif was != now:
            raise ValueError(f"line {line}: the keyword {was!r} became {now!r}, to a callable the program "
                             "does not show")
        else:
            self.loose.append((was, line))

    def check(self, after: str, declared: Declared) -> None:
        """Every variable renamed to a name of its own; every API name changed as a rename `declared` gives
        it, at what holds it; no keyword left to an unshown callable spells a parameter the program renames;
        the renamed program compiles."""
        bound = list(self.variables.values())
        if len(set(bound)) != len(bound):
            twice = sorted({name for name in bound if bound.count(name) > 1})
            raise ValueError(f"two variables became one: {', '.join(twice)}")
        for was, now, line in self.changes:
            if not declared.accounts_for(was, now):
                raise ValueError(f"line {line}: {was or 'an API name'} became {now or 'another'}, which no "
                                 "--rename of Dew's API declares")
        renamed = {name for was, now in self.variables.items() if was != now and was in self.before.parameters
                   for name in (was, now)}
        for was, line in self.loose:
            if was in renamed:
                raise ValueError(f"line {line}: the keyword {was!r} to a callable the program does not show "
                                 "spells a parameter it renames")
        compile(after, "<program>", "exec")

    def printed(self) -> str:
        names = {f"{was} -> {now}" for was, now in self.variables.items() if was != now}
        names |= {f"{was} -> {now}" for was, now, _ in self.changes}
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
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.define(target.id, set(keywords(node.value.args, bound=False)))
                continue
            else:
                continue
            self.define(named, names)
        self.parameters = {node.arg for node in ast.walk(program) if isinstance(node, ast.arg)}

    def define(self, name: str, names: set[str]) -> None:
        self.signatures[name] = self.signatures[name] & names if name in self.signatures else names


def keywords(arguments: ast.arguments, bound: bool) -> list[str]:
    """The parameters a keyword passes a callable of `arguments`, its first left out where it is `bound`."""
    positional = arguments.args[1:] if bound else arguments.args
    return [arg.arg for arg in (*positional, *arguments.kwonlyargs)]


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
        return bool(self.models) and bool(was) and bool(now) and self.at(was, now) in self.pairs


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
        renaming = Renaming(was, now)
        renaming.compare(was, now)
        renaming.check(after, declared)
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

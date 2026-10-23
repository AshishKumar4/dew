"""Keep the landing page's recordings across a change to the code they ran that only renames.

A recording names the sha256 of the program it ran, and the page refuses one whose
program has changed (src/data/recordings.mjs). Recorded outputs are re-recorded
only when a change alters what they show (the bible's rule for renames). So an
API rename that reaches a recorded program updates that program's source and its
recording's digest, and keeps the outputs:

    python site/scripts/redigest.py [--base origin/main]

The programs and the captures that name their digests are site/snippets/cells.py's,
the page's one declaration of both. For each recording whose program differs from
what its digest names, this checks two things. First, the recording is of `base`'s
program: its digest matches that program. Second, the program is that one renamed,
read as Python parses it:
- every value, string, f-string conversion and literal text, operator and keyword
  is unchanged;
- an API's names (a call's keyword argument, an attribute, an imported name) may be
  renamed where they are used;
- a variable (a name, a parameter, a function or class it defines, an import's
  binding) is renamed everywhere, each to its own name, which no other variable of
  the program had, so every use still means what it meant;
- the result compiles.
It then prints each renaming and writes the new digest, and nothing else in the
recording changes, byte for byte. Whether the library's rename keeps what the
recording shows (a label, a value it prints) is the head's own claim. Its reviewer
reads the renamings printed here.

One other change keeps a recording too: the pinned model (site/live/container/
text-to-image, which site/scripts/gen_pin.py writes everywhere) moved to a revision
with the same weights. A string that names the base's pinned revision may name the
new one when the Hub (HF_ENDPOINT) lists the two revisions with the same files and
every weight object (each LFS file) the same; the other files that differ, a run's
record for one, are printed for the reviewer.

A change that is not one of these, or a recording that was already of other code,
writes nothing and says to record again (site/scripts/capture_snippets.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import cells

SITE = cells.SITE
REPO = SITE.parent
PIN = "live/container/text-to-image"
VARIABLE, MEMBER = "variable", "member"
ROLES = {(ast.Name, "id"): VARIABLE, (ast.arg, "arg"): VARIABLE, (ast.FunctionDef, "name"): VARIABLE,
         (ast.AsyncFunctionDef, "name"): VARIABLE, (ast.ClassDef, "name"): VARIABLE,
         (ast.Global, "names"): VARIABLE, (ast.Nonlocal, "names"): VARIABLE,
         (ast.ExceptHandler, "name"): VARIABLE, (ast.MatchAs, "name"): VARIABLE,
         (ast.MatchStar, "name"): VARIABLE, (ast.MatchMapping, "rest"): VARIABLE,
         (ast.alias, "asname"): VARIABLE, (ast.keyword, "arg"): MEMBER, (ast.Attribute, "attr"): MEMBER,
         (ast.ImportFrom, "module"): MEMBER, (ast.MatchClass, "kwd_attrs"): MEMBER}
"""Each identifier of the syntax, by node and field, and what it names: a variable of the program, or an API's
name, which the program uses but does not bind."""


class Renaming:
    """The renamings between two parsed programs: each variable to its one new name, injectively, and each
    API name where it is used; and the pinned revision, where a string names its new one."""

    def __init__(self, moved: tuple[str, str]):
        self.moved = moved
        self.variables: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.pin = False

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
                elif isinstance(was, ast.alias) and field == "name":
                    self.alias(was, now, line)
                else:
                    self.compare(before, after, line)
        elif was != now:
            if isinstance(was, str) and self.moved[0] in was and was.replace(*self.moved) == now:
                self.pin = True
                return
            raise ValueError(f"line {line}: {was!r} became {now!r}, not a rename")

    def alias(self, was: ast.alias, now: ast.alias, line: int) -> None:
        """An import's name: an API name, and with no `as`, the variable it binds too (a module's first
        part)."""
        self.name(MEMBER, was.name, now.name, line)
        if was.asname is None and now.asname is None:
            self.name(VARIABLE, was.name.split(".")[0], now.name.split(".")[0], line)

    def check(self, after: str) -> None:
        """Every variable renamed to a name of its own; the renamed program compiles."""
        bound = list(self.variables.values())
        if len(set(bound)) != len(bound):
            twice = sorted({name for name in bound if bound.count(name) > 1})
            raise ValueError(f"two variables became one: {', '.join(twice)}")
        compile(after, "<program>", "exec")

    def printed(self) -> str:
        names = {f"{was} -> {now}" for was, now in self.variables.items() if was != now}
        names |= {f"{was} -> {now}" for was, nows in self.members.items() for now in nows}
        pin = [f"@{self.moved[0][:9]} -> @{self.moved[1][:9]}"] if self.pin else []
        return ", ".join(sorted(names) + pin)


def renamed(before: str, after: str, moved: tuple[str, str]) -> Renaming:
    """How `after` renames `before`; a ValueError, naming the first other difference, when it differs
    otherwise."""
    renaming = Renaming(moved)
    try:
        was, now = ast.parse(before), ast.parse(after)
        renaming.compare(was, now)
        renaming.check(after)
    except SyntaxError as error:
        raise ValueError(f"the renamed program does not compile: {error}") from error
    return renaming


def weights_kept(repository: str, old: str, new: str) -> list[str]:
    """The files other than weights that differ between `repository`'s revisions `old` and `new`, which hold
    the same files and the same weight objects; a ValueError where they do not."""
    def tree(revision: str) -> dict[str, dict]:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = f"{endpoint}/api/models/{repository}/tree/{revision}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as answer:
            return {item["path"]: item for item in json.load(answer) if item["type"] == "file"}

    def oid(item: dict) -> str | None:
        return (item.get("lfs") or {}).get("oid")

    was, now = tree(old), tree(new)
    if set(was) != set(now):
        raise ValueError(f"{repository}@{new[:9]} holds other files than @{old[:9]}: "
                         f"{sorted(set(was) ^ set(now))[:5]}")
    weights = [path for path in was if "lfs" in was[path] or "lfs" in now[path]]
    changed = [path for path in weights if oid(was[path]) != oid(now[path])]
    if changed:
        raise ValueError(f"{repository}@{new[:9]} has other weights than @{old[:9]}: {changed[:5]}")
    return sorted(path for path in was if path not in weights and was[path]["oid"] != now[path]["oid"])


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
    options = parser.parse_args()

    def at_base(path: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{options.base}:site/{path}"], check=True,
                              capture_output=True, text=True).stdout

    now, before = cells.programs(), cells.programs(at_base)
    repository, old = at_base(PIN).strip().split("@")
    new = (SITE / PIN).read_text().strip().split("@")[1]
    captures = {capture: (SITE / capture).read_bytes() for capture, _ in cells.recordings().values()}
    written = dict(captures)
    refused, kept = [], False
    for name, (capture, key) in cells.recordings().items():
        recorded, digest = digest_of(captures[capture], key), cells.digest(now[name])
        if recorded == digest:
            continue
        if cells.digest(before[name]) != recorded:
            refused.append(f"{name}: the recording is not of {options.base}'s program either")
            continue
        try:
            renaming = renamed(before[name], now[name], (old, new))
            if renaming.pin and not kept:
                differing = weights_kept(repository, old, new)
                print(f"{repository}: @{old[:9]} -> @{new[:9]}, the same weights; other files differing: "
                      f"{', '.join(differing) or 'none'}")
                kept = True
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

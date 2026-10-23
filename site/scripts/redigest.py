"""Keep the landing page's recordings across a change to the code they ran that only renames.

A recording names the sha256 of the program it ran, and the page refuses one whose
program has changed (src/data/recordings.mjs). Recorded outputs are re-recorded
only when a change alters what they show (the bible's rule for renames). So an
API rename that reaches a recorded program updates that program's source and its
recording's digest, and keeps the outputs:

    python site/scripts/redigest.py [--base origin/main]

For each recording whose program differs from what its digest names, this checks
two things. First, the recording is of `base`'s program: its digest matches that
program. Second, the program differs from it only by identifiers: the same tokens
in the same order, with numbers, strings, operators and keywords unchanged. An
API's names (an attribute after `.`, a keyword argument before `=`) may be renamed
where they are used. A variable's name is renamed everywhere it appears, to one
name the program did not use, so every use still means what it meant. It then
prints each renaming and writes the new digest. Nothing else in the recording changes.
Whether the library's rename keeps what the recording shows (a label, a value it
prints) is the head's own claim. Its reviewer reads the renamings printed here. A
change that is not a renaming, or a recording that was already of other code,
writes nothing and says to record again (site/scripts/capture_snippets.py).
"""

from __future__ import annotations

import argparse
import io
import json
import keyword
import subprocess
import sys
import tokenize
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import cells

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
IGNORED = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER)


def recordings(read: Callable[[str], str]) -> dict[tuple[str, str], str]:
    """Each recording's digest, (its capture.json, the digest's key there), to the program it names, as `read`
    (a path in the repository to its text) has the page's sources (src/pages/index.astro checks the same)."""
    framework, hero = read("site/snippets/framework.py"), read("site/src/data/hero.py")
    cell = "site/public/examples/framework/{}/capture.json"
    return {**{(cell.format(name), "script_sha256"): cells.cell(name, framework)
               for name in [*cells.CELLS["pool"], *cells.CELLS["colab"]]},
            ("site/public/examples/text/capture.json", "script_sha256"): read("site/src/data/text.py"),
            ("site/public/examples/sampler/capture.json", "script_sha256"): read("site/src/data/sampler.py"),
            ("site/src/data/capture.json", "script_sha256"): hero,
            ("site/src/data/capture.json", "cell_sha256"): cells.training_example(hero)}


def stored(capture: dict, key: str) -> str:
    """The digest `key` of a capture: at its top level for a cell's recording, in its meta for the hero's."""
    return capture[key] if key in capture else capture["meta"][key]


def tokens(source: str) -> list[tuple[tokenize.TokenInfo, bool]]:
    """`source`'s tokens, each with whether it names an API's member: an attribute (after `.`) or a keyword
    argument (before `=` in a call's parentheses), rather than a variable."""
    found = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
             if token.type not in IGNORED]
    marked, brackets = [], []
    for index, token in enumerate(found):
        following = found[index + 1].string if index + 1 < len(found) else ""
        attribute = index > 0 and found[index - 1].string == "."
        argument = brackets[-1:] == ["("] and following == "="
        member = token.type == tokenize.NAME and (attribute or argument)
        marked.append((token, member))
        if token.string in "([{" and token.type == tokenize.OP:
            brackets.append(token.string)
        elif token.string in ")]}" and token.type == tokenize.OP:
            brackets.pop()
    return marked


def renamings(before: str, after: str) -> dict[str, str]:
    """The identifiers `after` renames from `before`, old to new; a ValueError, naming the first other
    difference, when it differs otherwise."""
    old, new = tokens(before), tokens(after)
    if len(old) != len(new):
        raise ValueError(f"{len(old)} tokens became {len(new)}")
    renamed: dict[str, str] = {}
    variables = {token.string for token, member in old if token.type == tokenize.NAME and not member}
    for (was, member), (now, _) in zip(old, new, strict=True):
        if (was.type, was.string) == (now.type, now.string):
            continue
        if not (was.type == now.type == tokenize.NAME) or any(
                keyword.iskeyword(name) or keyword.issoftkeyword(name) for name in (was.string, now.string)):
            raise ValueError(f"line {was.start[0]}: {was.string!r} became {now.string!r}, "
                             "not an identifier's rename")
        if renamed.setdefault(was.string, now.string) != now.string:
            raise ValueError(f"line {was.start[0]}: {was.string!r} became both {renamed[was.string]!r} and "
                             f"{now.string!r}")
        if not member and now.string in variables:
            raise ValueError(f"line {was.start[0]}: the variable {was.string!r} became {now.string!r}, "
                             "a name the program already used")
    for (was, member), (now, _) in zip(old, new, strict=True):
        if not member and was.string in renamed and was.string == now.string:
            raise ValueError(f"line {was.start[0]}: the variable {was.string!r} kept its name here and "
                             f"became {renamed[was.string]!r} elsewhere")
    return renamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main",
                        help="the revision whose programs the recordings ran")
    options = parser.parse_args()

    def at_base(path: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{options.base}:{path}"], check=True,
                              capture_output=True, text=True).stdout

    now, before = recordings(lambda path: (REPO / path).read_text()), recordings(at_base)
    captures = {path: json.loads((REPO / path).read_text()) for path, _ in now}
    digests, refused = {}, []
    for (path, key), program in now.items():
        recorded, digest = stored(captures[path], key), cells.digest(program)
        if recorded == digest:
            continue
        name = f"{Path(path).parent.name} ({key})"
        if cells.digest(before[path, key]) != recorded:
            refused.append(f"{name}: the recording is not of {options.base}'s program either")
            continue
        try:
            renamed = renamings(before[path, key], program)
        except ValueError as difference:
            refused.append(f"{name}: {difference}")
            continue
        print(f"{name}: " + ", ".join(f"{old} -> {new}" for old, new in sorted(renamed.items())))
        digests[path, key] = (recorded, digest)
    if refused:
        print("not a rename, so record again (site/scripts/capture_snippets.py):\n  "
              + "\n  ".join(refused), file=sys.stderr)
        return 1
    for (path, key), (recorded, digest) in digests.items():
        text = (REPO / path).read_text()
        written = f'"{key}": "{recorded}"'
        if text.count(written) != 1:
            raise ValueError(f"{path} names {key} {recorded} other than once")
        (REPO / path).write_text(text.replace(written, f'"{key}": "{digest}"'))
    print(f"{len(digests)} recordings re-digested, their outputs kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())

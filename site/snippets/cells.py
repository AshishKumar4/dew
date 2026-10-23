"""The landing page's programs: each marked region of framework.py, as the page shows it, the hero's
script and cell, and the live text and sampler programs; and which recording names each one's digest.

This is the one place the page's programs and their recordings are declared. Run it to write them to
site/src/data/programs.json, which the page and the Colab notebooks read, and
site/src/data/recordings.json, each recorded program to the capture (under site/) that names its sha256
and the key it is under there, which the page checks every recording against (src/data/recordings.mjs)
and site/scripts/digest.py reads; `--check` writes nothing and fails where either is stale
(tools/check_gates). The recorder and the live container call `cell` and `training_example` directly.

    python site/snippets/cells.py [--check]
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
CELLS = json.loads((HERE / "cells.json").read_text())
PROGRAMS = SITE / "src/data/programs.json"
RECORDINGS = SITE / "src/data/recordings.json"


def cell(name: str, source: str | None = None) -> str:
    """The code of cell `name` in `source`, framework.py by default."""
    source = (HERE / "framework.py").read_text() if source is None else source
    begin, end = f"# Begin snippet: {name}\n", f"# End snippet: {name}"
    if begin not in source:
        raise KeyError(f"framework.py has no cell {name!r}")
    lines = source.split(begin, 1)[1].split(end, 1)[0].split("\n")
    indent = min(len(line) - len(line.lstrip(" ")) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines).strip()


def digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def training_example(source: str) -> str:
    """site/src/data/hero.py as the page shows it, without its command line."""
    before, command = source.split("# Begin command line\n", 1)
    _, after = command.split("# End command line\n", 1)
    return before + after


def programs() -> dict[str, str]:
    """Every program the page shows: each cell of cells.json, the hero's cell and its script, and the live
    text and sampler."""
    hero = (SITE / "src/data/hero.py").read_text()
    return {**{name: cell(name) for name in [*CELLS["pool"], *CELLS["colab"]]},
            "hero": training_example(hero), "hero.py": hero,
            "text": (SITE / "src/data/text.py").read_text(), "sampler": (SITE / "src/data/sampler.py").read_text()}


def recordings() -> dict[str, list[str]]:
    """Each recorded program to [the capture under site/ that names its sha256, the key it is under there]."""
    return {**{name: [f"public/examples/framework/{name}/capture.json", "script_sha256"]
               for name in [*CELLS["pool"], *CELLS["colab"]]},
            "hero": ["src/data/capture.json", "meta.cell_sha256"],
            "hero.py": ["src/data/capture.json", "meta.script_sha256"],
            "text": ["public/examples/text/capture.json", "script_sha256"],
            "sampler": ["public/examples/sampler/capture.json", "script_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Check the files without writing them.")
    args = parser.parse_args()
    written = {PROGRAMS: json.dumps(programs(), indent=1) + "\n",
               RECORDINGS: json.dumps(recordings(), indent=1) + "\n"}
    stale = [path for path, text in written.items() if not path.exists() or path.read_text() != text]
    if args.check and stale:
        raise SystemExit(f"cells: {', '.join(str(path.relative_to(SITE)) for path in stale)} stale; "
                         "run python site/snippets/cells.py")
    if not args.check:
        for path in stale:
            path.write_text(written[path])


if __name__ == "__main__":
    main()

"""The landing page's cells: each marked region of framework.py, as the page shows it.

This is the one place the page's programs come from. Run it to write them to
site/src/data/programs.json, which the page and the Colab notebooks read
(tests/test_landing_examples.py checks it is current); the recorder and the
live container call `cell` and `training_example` directly. A recording's
script_sha256 is the hash of the program the page shows.

    python site/snippets/cells.py
"""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = json.loads((HERE / "cells.json").read_text())
PROGRAMS = HERE.parent / "src/data/programs.json"


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
    source = source.replace("import argparse\n\n", "", 1)
    return re.sub(r"parser = argparse.ArgumentParser\(\)[\s\S]*?steps = parser.parse_args\(\).steps",
                  "steps = 1000", source, count=1)


def programs() -> dict[str, str]:
    """Every program the page shows: each cell of cells.json, and the hero."""
    names = [*CELLS["pool"], *CELLS["colab"]]
    return {**{name: cell(name) for name in names},
            "hero": training_example((HERE.parent / "src/data/hero.py").read_text())}


if __name__ == "__main__":
    PROGRAMS.write_text(json.dumps(programs(), indent=1) + "\n")

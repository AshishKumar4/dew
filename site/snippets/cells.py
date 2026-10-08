"""The landing page's cells: each marked region of framework.py, as the page shows it.

`cell` dedents a region exactly as site/src/data/framework-examples.mjs does, so
a recording's script_sha256 is the hash of the code on the page.
"""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = json.loads((HERE / "cells.json").read_text())


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
    """site/src/data/hero.py as the page shows it, without its command line
    (trainingExample in framework-examples.mjs)."""
    source = source.replace("import argparse\n\n", "", 1)
    return re.sub(r"parser = argparse.ArgumentParser\(\)[\s\S]*?steps = parser.parse_args\(\).steps",
                  "steps = 1000", source, count=1)

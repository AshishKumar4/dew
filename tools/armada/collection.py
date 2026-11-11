"""What pytest collects of test files on the CPU, for tools/armada/ci.py: how many tests of each file a mark
expression selects, and the GPU generations each file's selected tests are pinned to, which the plugin pins.py
beside this file reads off their marks."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def collected(python: str, marks: str, files: list[str], within: float | None = None, tolerant: bool = False,
              pinned: bool = False) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Each of test files `files` with tests pytest's `-m marks` selects, and how many, as `python`'s
    environment collects it on the CPU; and, where `pinned`, each such file's GPU generations (pins.py): those
    its selected tests are pinned to, none where a test of it runs on every one. `tolerant`, a file that does
    not collect has no tests and stops no other's reading (its own run says its error, as every lane's does; a
    lane that died of one file's import ran none of its files); a collection interrupted is no reading. Within
    `within` seconds where its caller has a deadline."""
    if not files:
        return {}, {}
    with tempfile.TemporaryDirectory() as scratch:
        pins = Path(scratch) / "pins.json"
        # The plugin beside this file, on the path of the collection that loads it.
        here = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent),
                                             os.environ.get("PYTHONPATH")]))
        done = subprocess.run([f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q",
                               "-m", marks, *(["--continue-on-collection-errors"] if tolerant else []),
                               *(["-p", "pins", f"--pins-out={pins}"] if pinned else []), *files],
                              capture_output=True, text=True, timeout=within,
                              env={**os.environ, "JAX_PLATFORMS": "cpu",
                                   **({"PYTHONPATH": here} if pinned else {})})
        # Told to go on past a file that does not collect, pytest exits 1 for those; 2 is a collection
        # interrupted, no reading at all.
        if done.returncode not in ((0, 1, 5) if tolerant else (0, 5)):
            raise SystemExit(f"collecting the tests marked {marks} exited {done.returncode}\n"
                             f"{done.stdout[-4000:]}{done.stderr[-2000:]}")
        counts: dict[str, int] = {}
        for node in done.stdout.splitlines():
            if "::" in node:
                counts[node.split("::")[0]] = counts.get(node.split("::")[0], 0) + 1
        return counts, json.loads(pins.read_text()) if pinned and counts else {}

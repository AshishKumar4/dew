"""Execute every tutorial notebook top to bottom at smoke size, on a CPU.

Each notebook's Settings cell shrinks its own sizes when DEW_TUTORIAL_SMOKE=1,
so this script holds no per-notebook knowledge. It skips the `%pip install`
cells, since Dew comes from the checkout, and runs the notebooks in file order
in one working directory, which lets notebook 04 read the checkpoint that 02
writes. A cell that raises fails the run, unless the cell carries the
`raises-exception` tag that marks a deliberate error.

    python tools/run_tutorials.py                 # every notebook
    python tools/run_tutorials.py 02 04           # the ones whose names start so
    python tools/run_tutorials.py --workdir /tmp/tutorials

The committed outputs come from full-size runs on a GPU; this only proves the
notebooks still run against the current code.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def smoke_copy(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells
                      if not (cell.cell_type == "code" and cell.source.lstrip().startswith(("%pip", "!pip")))]
    return notebook


def execute(path: Path, workdir: Path, timeout: int) -> float:
    notebook = smoke_copy(path)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(workdir)}})
    started = time.monotonic()
    client.execute()
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefixes", nargs="*", help="run only the notebooks whose names start with these")
    parser.add_argument("--workdir", type=Path, help="where the notebooks write their runs (default: a new temporary directory)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds one cell may take")
    options = parser.parse_args()

    notebooks = sorted((ROOT / "tutorials").glob("*.ipynb"))
    if options.prefixes:
        notebooks = [path for path in notebooks if path.name.startswith(tuple(options.prefixes))]
    workdir = options.workdir or Path(tempfile.mkdtemp(prefix="dew-tutorials-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["DEW_TUTORIAL_SMOKE"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MPLBACKEND", "Agg")

    failed = []
    for path in notebooks:
        print(f"{path.name}: running in {workdir}", flush=True)
        try:
            seconds = execute(path, workdir, options.timeout)
        except CellExecutionError as error:
            failed.append(path.name)
            print(f"{path.name}: FAILED\n{error}", flush=True)
            continue
        print(f"{path.name}: ok in {seconds:.0f} s", flush=True)
    if failed:
        print(f"{len(failed)} of {len(notebooks)} notebooks failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"all {len(notebooks)} notebooks ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())

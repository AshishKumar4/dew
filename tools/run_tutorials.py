"""Execute every tutorial notebook top to bottom, at smoke size on a CPU by default.

Each notebook's Settings cell shrinks its own sizes when DEW_TUTORIAL_SMOKE=1,
so this script holds no per-notebook knowledge. `--full` runs the notebooks as
committed, on whatever accelerator JAX finds. Either way the script skips the
`%pip install` cells, since Dew comes from the checkout, and runs the
notebooks in file order in one working directory, which lets notebook 04 read
the checkpoint that 02 writes. A cell that raises, runs past `--timeout`, or
kills its kernel fails its notebook, unless the cell carries the
`raises-exception` tag that marks a deliberate error; the other notebooks
still run, and the script exits nonzero if any failed.

`--save DIR` writes each notebook to DIR as the repository keeps it, with the
run's outputs in place of the old ones and its install cells unexecuted; that
is how the committed outputs are made, by copying DIR over tutorials/. A cell
the run did not reach is saved with no output, and the site build refuses it.
A notebook that ran through also gets `dew.outputs` in its metadata: the last
commit that changed src/dew or pyproject.toml in this checkout (so the record
survives a rebase of a branch that leaves the library and its dependencies
alone), the date, the device and the JAX version. The site prints it under the notebook's outputs. It assumes the
interpreter running this script has Dew installed from this checkout.

`--imports FILE` writes, for each notebook that ran through, the Dew modules
its kernel had imported by the end. tools/check_tutorial_outputs.py compares
them with each notebook's recorded commit, to catch outputs made before a
change to the code they ran.

    python tools/run_tutorials.py                 # every notebook, smoke size
    python tools/run_tutorials.py 02 04           # the ones whose names start so
    python tools/run_tutorials.py --workdir /tmp/tutorials --imports /tmp/imports.json
    python tools/run_tutorials.py --full --save /tmp/executed --timeout 14400
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from nbformat.v4.rwbase import split_lines

ROOT = Path(__file__).resolve().parent.parent

# Run in each kernel after the notebook's last cell: what it imported from Dew, and where it ran.
PROBE_MARK = "dew-tutorial-probe "
PROBE = f"""\
import json as _json, sys as _sys
import jax as _jax
print({PROBE_MARK!r} + _json.dumps({{
    "modules": sorted(name for name in _sys.modules if name == "dew" or name.startswith("dew.")),
    "jax": _jax.__version__,
    "device": _jax.devices()[0].device_kind,
}}))
"""


def is_install(cell) -> bool:
    return cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith(("%pip", "!pip"))


def smoke_copy(path: Path) -> nbformat.NotebookNode:
    """The notebook without its install cells, and without the outputs of an earlier run."""
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not is_install(cell)]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    return notebook


def committed_form(path: Path, executed: nbformat.NotebookNode, record: dict | None) -> str:
    """The notebook at `path` in the repository's own JSON, with the outputs of `executed`, matched by cell id.

    `record` describes the run for the notebook's metadata; without one (the run failed) an
    earlier run's record is dropped, since it no longer describes the outputs.
    """
    notebook = json.loads(path.read_text())
    dew = notebook["metadata"].pop("dew", {})
    dew.pop("outputs", None)
    if record is not None:
        dew["outputs"] = record
    if dew:
        notebook["metadata"]["dew"] = dew
    runs = {cell["id"]: cell for cell in split_lines(copy.deepcopy(executed)).cells if cell["cell_type"] == "code"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        run = runs.get(cell["id"], {"outputs": [], "execution_count": None})
        cell["outputs"], cell["execution_count"] = run["outputs"], run["execution_count"]
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def this_python_kernel(directory: Path) -> str:
    """A kernel spec that runs this interpreter, so the notebooks import the Dew this script sees.

    Jupyter's own "python3" spec can name another Python, such as the system one on a
    Colab VM, whose Dew is not the checkout's.
    """
    from ipykernel.kernelspec import write_kernel_spec

    name = "dew-tutorials"
    # Written on every run: a --workdir an earlier run used may hold a spec for its interpreter.
    shutil.rmtree(directory / "kernels" / name, ignore_errors=True)
    write_kernel_spec(directory / "kernels" / name)
    os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [str(directory), os.environ.get("JUPYTER_PATH")]))
    return name


LIBRARY = ("src/dew", "pyproject.toml")  # the code and the pinned dependencies the outputs come from


def library_commit() -> str:
    """The last commit that changed the library or its dependencies in this checkout, which the saved outputs come from."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

    if git("status", "--porcelain", "--", *LIBRARY):
        raise SystemExit(f"{' or '.join(LIBRARY)} has uncommitted changes, so no commit describes the outputs; "
                         "commit them before --save")
    return git("log", "-1", "--format=%H", "--", *LIBRARY)


def execute(path: Path, workdir: Path, timeout: int, kernel: str):
    """Run one notebook. Returns the executed copy, what the probe found (None when a cell failed),
    the seconds it took and the error that stopped it, if one did."""
    notebook = smoke_copy(path)
    notebook.cells.append(nbformat.v4.new_code_cell(PROBE))
    client = NotebookClient(notebook, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(workdir)}})
    started = time.monotonic()
    error = None
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as caught:
        error = caught
    seconds = time.monotonic() - started
    probe_cell = notebook.cells.pop()
    if error is not None:
        return notebook, None, seconds, error
    for output in probe_cell.outputs:
        for line in output.get("text", "").splitlines():
            if line.startswith(PROBE_MARK):
                return notebook, json.loads(line[len(PROBE_MARK):]), seconds, None
    raise SystemExit(f"{path.name}: the probe after the last cell printed nothing: {probe_cell.outputs}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefixes", nargs="*", help="run only the notebooks whose names start with these")
    parser.add_argument("--workdir", type=Path, help="where the notebooks write their runs (default: a new temporary directory)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds one cell may take")
    parser.add_argument("--full", action="store_true", help="run at the committed sizes, on the accelerator JAX finds")
    parser.add_argument("--save", type=Path, help="write each notebook, with this run's outputs, to this directory")
    parser.add_argument("--imports", type=Path, help="write the Dew modules each notebook imported, as JSON, to this file")
    options = parser.parse_args()

    notebooks = sorted((ROOT / "tutorials").glob("*.ipynb"))
    if options.prefixes:
        notebooks = [path for path in notebooks if path.name.startswith(tuple(options.prefixes))]
    workdir = options.workdir or Path(tempfile.mkdtemp(prefix="dew-tutorials-"))
    workdir.mkdir(parents=True, exist_ok=True)
    if options.save:
        options.save.mkdir(parents=True, exist_ok=True)
    if options.full:
        os.environ.pop("DEW_TUTORIAL_SMOKE", None)
        # Figures become image outputs only with the inline backend.
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    else:
        os.environ["DEW_TUTORIAL_SMOKE"] = "1"
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("MPLBACKEND", "Agg")

    kernel = this_python_kernel(workdir / ".jupyter")
    commit = library_commit() if options.save else None
    imports = {}
    failed = []
    for path in notebooks:
        print(f"{path.name}: running in {workdir}", flush=True)
        executed, probe, seconds, error = execute(path, workdir, options.timeout, kernel)
        if options.save:
            # A notebook that fails is saved too, with the outputs up to the failing cell.
            record = None if probe is None else {
                "commit": commit, "date": time.strftime("%Y-%m-%d"), "device": probe["device"], "jax": probe["jax"]}
            (options.save / path.name).write_text(committed_form(path, executed, record))
        if error is not None:
            failed.append(path.name)
            print(f"{path.name}: FAILED\n{error}", flush=True)
            continue
        imports[path.name] = probe["modules"]
        print(f"{path.name}: ok in {seconds:.0f} s", flush=True)
    if options.imports:
        options.imports.write_text(json.dumps(imports, indent=1) + "\n")
    if failed:
        print(f"{len(failed)} of {len(notebooks)} notebooks failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"all {len(notebooks)} notebooks ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())

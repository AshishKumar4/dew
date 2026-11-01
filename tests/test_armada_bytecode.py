"""What an armada environment leaves of Dew's own modules for a task (tools/armada/compiled.sh, which
tools/armada/install.sh runs for each Python): bytecode no checkout can make stale.

A container's boot checks the task's commit out over the environment's (`git checkout -f`), which on the hosts
CI runs on gives every tracked file a new mtime, so a pyc that records its source's mtime is stale for every
module of src/ and the container's first interpreters compile them again. The script compiles src/ with the
hash of each source in its pyc instead, which Python compares at every import: a module the task's commit
leaves alone loads from the pyc, one it edits compiles again, whatever its mtime and size (ledger 3215)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

COMPILED = Path(__file__).resolve().parents[1] / "tools/armada/compiled.sh"

PROBE = """
import importlib._bootstrap_external as external
import json
compiled = []
original = external.SourceLoader.source_to_code


def spy(self, data, path, *, _optimize=-1):
    compiled.append(path.rsplit("/", 1)[-1])
    return original(self, data, path, _optimize=_optimize)


external.SourceLoader.source_to_code = spy
import pkg.edited, pkg.kept
try:
    import pkg.gone
    gone = "imported"
except ModuleNotFoundError:
    gone = "missing"
print(json.dumps({"value": pkg.edited.VALUE, "compiled": sorted(compiled), "gone": gone}))
"""
"""A fresh interpreter's report: the value the edited module holds, the modules it compiled from source (an
import that found no usable pyc), and whether the deleted module imports."""


def interpreter(root: Path) -> dict:
    """What `PROBE` prints in a fresh interpreter run in `root` over its src/, with the two variables that
    keep a pyc from being written or read where it lies unset."""
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"}}
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True,
                          env={**environment, "PYTHONPATH": "src"})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def environment_built(root: Path) -> Path:
    """A tree under `root` with a package of three modules, compiled as an environment's install leaves it: by
    tools/armada/compiled.sh, with this interpreter."""
    package = root / "src/pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name, source in (("edited", "VALUE = 1\n"), ("kept", "KEPT = 1\n"), ("gone", "GONE = 1\n")):
        (package / f"{name}.py").write_text(source)
    built = subprocess.run(["sh", str(COMPILED), sys.executable], cwd=root, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    return package


def task_checkout(package: Path, mtimes: str) -> None:
    """What a task's commit leaves over the environment's tree: one module edited to a source of its size, one
    deleted. With `mtimes` "boot", every file left has a new mtime, as a boot's checkout gives; with "kept",
    the edited module has the mtime it had and the others new ones."""
    edited = package / "edited.py"
    before = edited.stat()
    edited.write_text("VALUE = 2\n")
    assert edited.stat().st_size == before.st_size
    (package / "gone.py").unlink()
    later = before.st_mtime_ns + 3_000_000_000
    for path in (package / "__init__.py", package / "kept.py", *([edited] if mtimes == "boot" else [])):
        os.utime(path, ns=(later, later))
    if mtimes == "kept":
        os.utime(edited, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (edited.stat().st_mtime_ns == before.st_mtime_ns) == (mtimes == "kept")


@pytest.mark.parametrize("mtimes", ["boot", "kept"])
def test_an_edited_module_is_compiled_again_and_an_unedited_one_is_loaded_from_its_pyc(tmp_path, mtimes):
    """Whatever the edit's mtime and size: the pyc the environment left holds the old source's hash, which
    Python compares with the source's at each import, so only the edit compiles, a boot's new mtimes recompile
    nothing else, and the module a task's commit deletes is gone with its pyc still in __pycache__; the pyc
    Python writes for the edit serves the next interpreter."""
    task_checkout(environment_built(tmp_path), mtimes)
    assert interpreter(tmp_path) == {"value": 2, "compiled": ["edited.py"], "gone": "missing"}
    assert interpreter(tmp_path) == {"value": 2, "compiled": [], "gone": "missing"}

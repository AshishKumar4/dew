#!/usr/bin/env python3
"""Install a built wheel into a new, empty virtual environment with pip and use it.

    python tools/smoke_wheel.py dist/dewml-*.whl

The wheel is installed from PyPI's index and Dew's package index, which serves Dew's JAX build, dewml's
releases and the forks Dew carries (`INDEX`), as a user's pip installs it, so a requirement neither can
serve fails here. pip takes the newest release each requirement admits, live: this is the base requirements'
highest set, and CI's package job runs it on every push, a train's among them. Then, from outside the
checkout so nothing of it is importable:

- every module the wheel installed compiles (`compileall`), so a syntax error anywhere in the package
  fails, imported or not;
- tools/bounds_workflow.py runs in the environment: every module imports, the console entry point answers,
  a decoder runs forward, a language model trains and resumes from a checkpoint, a decoder directory is
  saved and loaded, records are read. It is the workflow `tools/check_bounds.py` runs at each bound of
  the declared ranges (.github/workflows/ci.yml's bounds job), with uv, which pip cannot do.

Exits nonzero on the first failure. The release workflow runs this on the artifact it then uploads.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

INDEX = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/dew/builds.py"))["PACKAGE_INDEX"]
"""Dew's package index, the one place Dew's JAX build is served (dew.builds.PACKAGE_INDEX)."""
WORKFLOW = Path(__file__).with_name("bounds_workflow.py")


def run(*command: str | Path, cwd: Path) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="dew-wheel-") as scratch:
        root = Path(scratch)
        venv.create(root / "env", with_pip=True)
        python = root / "env" / "bin" / "python"
        # tools/pip.txt's pip first: older pips trust the roots of eMudhra's compromised CAs (2026-10-20).
        run(python, "-m", "pip", "install", "--quiet", "-r", Path(__file__).with_name("pip.txt"), cwd=root)
        run(python, "-m", "pip", "install", "--quiet", "--index-url", "https://pypi.org/simple",
            "--extra-index-url", INDEX, wheel, cwd=root)
        # find_spec locates the package without importing it, so a module
        # that does not compile fails at compileall, named.
        locate = ("import importlib.util, pathlib; "
                  "print(pathlib.Path(importlib.util.find_spec('dew').origin).parent)")
        package = subprocess.run([str(python), "-c", locate], cwd=root, check=True, capture_output=True,
                                 text=True).stdout.strip()
        run(python, "-m", "compileall", "-q", package, cwd=root)
        run(python, "-P", WORKFLOW, "--scratch", root / "work", cwd=root)
    print(f"{wheel.name}: installs, compiles and runs the bounds workflow")


if __name__ == "__main__":
    sys.exit(main())

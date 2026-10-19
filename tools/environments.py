#!/usr/bin/env python3
"""A checkout's own locked environment, built once for each lock and shared.

    python tools/environments.py TREE [--lock cpu|cuda12|cuda13] [--cache DIR]

prints the Python of TREE's environment: TREE's `locks/<lock>.txt` installed
by TREE's own tools/install_environment.sh at TREE's .python-version, into
CACHE/<lock>-<digest of those three files>. The lock defaults to the one this
machine installs: the newest of JAX's CUDA builds its driver and GPUs run, by
TREE's own table (`dew.telemetry.devices.CUDA_BUILDS`, read as its literal, as
site/scripts/gen_install.py reads it, since no environment exists yet to
import Dew from), or cpu where no NVIDIA GPU answers. Checkouts whose three
files agree share one environment; a caller runs each checkout's own code
with its `src` first on PYTHONPATH, ahead of the Dew the environment holds.

The standard library alone, so it runs on a machine that holds no environment
yet: tools/perf_gate.py builds each tree's with it, and so do GPU sessions.
"""

import argparse
import ast
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dew" / "environments"
"""Where the environments are kept by default, beside Dew's other caches (`dew.cache.dew_cache_dir`)."""


def gpus() -> tuple[str, tuple[int, ...]] | None:
    """The driver version and the least SM of this machine's NVIDIA GPUs, or
    None where nvidia-smi is absent or answers none."""
    if shutil.which("nvidia-smi") is None:
        return None
    listed = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
                            capture_output=True, text=True)
    lines = [line for line in listed.stdout.splitlines() if line.strip()]
    rows = [[field.strip() for field in line.split(",")] for line in lines]
    if listed.returncode != 0 or not rows:
        return None
    return rows[0][0], min(tuple(int(part) for part in row[1].split(".")) for row in rows)


def machine_lock(tree: Path) -> str:
    """The lock of `tree` this machine installs (the module's docstring)."""
    found = gpus()
    if found is None:
        return "cpu"
    source = ast.parse((tree / "src" / "dew" / "telemetry" / "devices.py").read_text())
    builds = next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                  and any(isinstance(name, ast.Name) and name.id == "CUDA_BUILDS" for name in node.targets))
    driver, sm = found
    runs = [name for name, least, least_sm in builds if int(driver.split(".")[0]) >= least and sm >= least_sm]
    if not runs:
        raise SystemExit(f"no CUDA build of JAX runs on driver {driver} and SM {sm}: {builds}")
    return runs[0]


def environment(tree: Path, lock: str, cache: Path = CACHE) -> Path:
    """The Python of `tree`'s environment for `lock`, built first if no
    finished build exists. Builders of one environment take turns on a lock
    file beside it, and a build that never finished is started over."""
    tree, cache = tree.resolve(), cache.resolve()
    files = [tree / "locks" / f"{lock}.txt", tree / "tools" / "install_environment.sh",
             tree / ".python-version"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()[:12]
    venv = cache / f"{lock}-{digest}"
    python = venv / "bin" / "python"
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / f"{venv.name}.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        if not (venv / "complete").is_file():
            if venv.exists():
                shutil.rmtree(venv)
            # Each uv call names its interpreter, so no UV_PYTHON or environment of the caller's is used.
            subprocess.run(["uv", "venv", "-q", "--python", files[2].read_text().strip(), str(venv)],
                           check=True)
            subprocess.run(["sh", "tools/install_environment.sh", lock, "-q", "--python", str(python)],
                           cwd=tree, check=True)
            (venv / "complete").touch()
    return python


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/environments.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("tree", type=Path)
    parser.add_argument("--lock", default=None, help="cpu, cuda12 or cuda13; by default this machine's")
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    print(environment(args.tree, args.lock or machine_lock(args.tree), args.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())

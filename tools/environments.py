#!/usr/bin/env python3
"""A checkout's own locked environment, built once for each lock and shared.

    python tools/environments.py TREE [--lock cpu|cuda12|cuda13] [--cache DIR]

prints the Python of TREE's environment: TREE's `locks/<lock>.txt` installed
by TREE's own tools/install_environment.sh at TREE's .python-version, into
CACHE/<lock>-<digest of those three files>. The lock defaults to the one this
machine installs (`machine_lock`). Checkouts whose three files agree share one
environment; a caller runs each checkout's own code with its `src` first on
PYTHONPATH, ahead of the Dew the environment holds.

The standard library alone, so it runs on a machine that holds no environment
yet: tools/perf_gate.py builds each tree's with it, and so do GPU sessions.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dew" / "environments"
"""Where the environments are kept by default, beside Dew's other caches (`dew.cache.dew_cache_dir`)."""


class Unlocked(Exception):
    """A checkout without the files its environment is built from: one older than the locks."""


def _builds():
    """dew.telemetry.builds, this checkout's own, loaded by its path: Dew's
    package imports JAX, which no environment holds yet."""
    path = Path(__file__).resolve().parents[1] / "src" / "dew" / "telemetry" / "builds.py"
    spec = importlib.util.spec_from_file_location("dew_telemetry_builds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine_lock() -> str:
    """The lock this machine installs: the newest of JAX's CUDA builds its
    NVIDIA driver and GPUs run (`dew.telemetry.builds`), else cpu, where
    Dew's own runtime check speaks for a GPU no build runs on
    (`dew.telemetry.devices.cpu_only_refusal`)."""
    builds = _builds()
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    answer = subprocess.run(builds.QUERY, capture_output=True, text=True)
    return builds.newest_build(builds.listed_gpus(answer.stdout if answer.returncode == 0 else None)) or "cpu"


def environment(tree: Path, lock: str, cache: Path = CACHE) -> Path:
    """The Python of `tree`'s environment for `lock`, built first if no
    finished build exists. Builders of one environment take turns on a lock
    file beside it, and a build that never finished is started over."""
    tree, cache = tree.resolve(), cache.resolve()
    files = [tree / "locks" / f"{lock}.txt", tree / "tools" / "install_environment.sh",
             tree / ".python-version"]
    if missing := [str(path.relative_to(tree)) for path in files if not path.is_file()]:
        raise Unlocked(f"{tree} has no {', '.join(missing)}")
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
    print(environment(args.tree, args.lock or machine_lock(), args.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())

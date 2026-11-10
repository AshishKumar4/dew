"""What a GPU lane (tools/armada/ci.py `gpu-lane`) needs to run each test file as measured: the most any
GPU class's last whole run of it held with its children (`ci.gpu_needs`), in bytes, as JSON by file; a file
no class measured is absent. A GPU tier sizes each part's job by it, so the lane admits every file at its need
(2026-11-10: a part given a default 8 GiB had a 4 GiB lane for a file measured at 5.9 GiB).

    DEW_XLA_CACHE_URL=... DEW_XLA_CACHE_TOKEN=... python3 tools/armada/gpu_holds.py --tests "FILE..."

A group (`<file>#G/N`) is its file's: its whole measured run holds any group of it."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path


def holds(tests: list[str]) -> dict[str, float]:
    """Each of `tests`' files by the most a GPU class's last whole run of it held (`ci.borrowed_records`, over
    every class)."""
    # ci.py, beside this file, loaded by its path as ci.py loads its neighbours.
    spec = importlib.util.spec_from_file_location("armada_ci", Path(__file__).resolve().with_name("ci.py"))
    assert spec is not None and spec.loader is not None
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    files = sorted({name.partition("#")[0] for name in tests})
    token = os.environ.get("DEW_XLA_CACHE_TOKEN", "")
    return {name: ci.gpu_needs(found, math.inf)[1]
            for name, found in ci.borrowed_records(files, ci.PYTHONS[0], "", token).items()}


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/armada/gpu_holds.py")
    parser.add_argument("--tests", required=True, help="the test files, space-separated, a group as its file")
    print(json.dumps(holds(parser.parse_args().tests.split())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

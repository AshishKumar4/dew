"""What a GPU lane (tools/armada/ci.py `gpu-lane`) needs to run each test file as measured: the most any
GPU class's last run of it held with its children (`ci.gpu_needs`), whole or a group's, in bytes, as JSON by
file, or what a lane filed one of its units as needing by its prepared programs where it could not hold it
(ci.py's `filed_needs`), whichever is more; a file neither names is absent. A GPU tier sizes each part's job
by it, so the lane admits every file at its need (2026-11-10: a part given a default 8 GiB had a 4 GiB lane
for a file measured at 5.9 GiB; rows 12893 and after: a unit no lane held, never measured, sized none).

    DEW_XLA_CACHE_URL=... DEW_XLA_CACHE_TOKEN=... python3 tools/armada/gpu_holds.py --tests "FILE..."

A group (`<file>#G/N`) is its file's: a tier cuts a file's groups anew, so one's measure is the most any run
of its file held, which is what its lane admits it at."""

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path


def holds(tests: list[str]) -> dict[str, float]:
    """Each of `tests`' files by the most a GPU class's last run of it held, whole or a group's
    (`ci.borrowed_records`, over every class)."""
    # ci.py, beside this file, loaded by its path as ci.py loads its neighbours.
    spec = importlib.util.spec_from_file_location("armada_ci", Path(__file__).resolve().with_name("ci.py"))
    assert spec is not None and spec.loader is not None
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    files = sorted({name.partition("#")[0] for name in tests})
    token = os.environ.get("DEW_XLA_CACHE_TOKEN", "")
    # Read whole or not at all (`strict`): a read the Worker does not answer fails this tool, whose caller
    # sizes a job by it, and never reads as a file no class measured. A test node (`file::test`) holds what
    # its file does.
    found = ci.borrowed_records(sorted({ci.grading.owning(name) for name in files}), ci.PYTHONS[0], "", token,
                                strict=True)
    measured = {name: ci.gpu_needs(found[whole], math.inf)[1] for name in files
                if (whole := ci.grading.owning(name)) in found}
    needs = filed(ci, files, token)
    return {name: max(measured.get(name, 0.0), needs.get(name, 0.0)) for name in files
            if name in measured or name in needs}


def filed(ci, files: list[str], token: str) -> dict[str, float]:
    """Each of `files` by the most a lane filed one of its units as needing, on any class, within
    `NEED_DAYS` (records.py's `need_key`); read whole or not at all, as the measures are."""
    prefix = f"{ci.CACHE_PREFIX}-need/{ci.PYTHONS[0]}/"
    most: dict[str, float] = {}
    for key in ci.listed(prefix, token):
        file = ci.grading.owning(ci.key_parts(key)[1])
        if file in files and (found := ci.store(token).get(prefix + key)) is not None:
            need = json.loads(found)
            if time.time() - need["at"] <= ci.records.NEED_DAYS * 86400:
                most[file] = max(most.get(file, 0.0), float(need["need"]))
    return most


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/armada/gpu_holds.py")
    parser.add_argument("--tests", required=True, help="the test files, space-separated, a group as its file")
    print(json.dumps(holds(parser.parse_args().tests.split())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE

The plan prints a task matrix for each Python CI proves. Whole test files are
packed heaviest first into the lightest of the tasks, each near `target`
seconds, so a task imports only its own modules. A file heavier than that runs
as that many pytest-split groups of itself. A file weighs armada's median of
its last green runs, else its tests' sum in tests/test_durations.json, else
the mean. Each entry names its rows, one a file (or a file's group), so armada
grades a task that leaves one out as not green.

A task runs its files under its Python's environment (tools/armada/install.sh)
with CI's selection, and writes a verdict row for each. A row is red on a
failed or erroring test, on a test skipped because its import failed (a file
CI would never have run), or when pytest left no report of it.
"""

import argparse
import heapq
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PYTHONS = ("3.12", "3.14")
"""The compatibility floor and the newest release CI proves."""
SKIPPED = {"tests/test_gen_api.py"}
"""CI runs it in the lint job, under the griffe it pins."""
DURATIONS = Path("tests/test_durations.json")


def weights(files: list[str], timings: dict, python: str) -> dict[str, float]:
    """Each file's seconds on armada at `python`: its median, a split file's
    as the sum of its last complete set of groups; else its tests' sum in
    tests/test_durations.json, scaled by how much slower armada ran the
    files measured both ways; else the mean."""
    recorded: dict[str, float] = {}
    if DURATIONS.is_file():
        for node, seconds in json.loads(DURATIONS.read_text()).items():
            recorded[node.split("::")[0]] = recorded.get(node.split("::")[0], 0.0) + seconds
    measured = dict(timings.get("files", {}))
    groups: dict[tuple[str, int], dict[int, float]] = {}
    for name, seconds in timings.get("rows", {}).items():
        own, _, rest = name.partition(":")
        file, _, split = rest.partition("#")
        if own == python and split:
            group, count = (int(part) for part in split.split("/"))
            groups.setdefault((file, count), {})[group] = seconds
    for (file, count), seconds in sorted(groups.items()):
        if len(seconds) == count:
            measured[file] = sum(seconds.values())
    ratios = sorted(measured[name] / recorded[name] for name in files
                    if name in measured and recorded.get(name, 0) > 1)
    slower = ratios[len(ratios) // 2] if ratios else 1.0
    known = {name: measured.get(name, recorded[name] * slower if name in recorded else None)
             for name in files}
    present = [seconds for seconds in known.values() if seconds is not None]
    mean = sum(present) / len(present) if present else 1.0
    return {name: mean if seconds is None else seconds for name, seconds in known.items()}


def row(python: str, name: str, split: str) -> str:
    return f"{python}:{name}" + (f"#{split}" if split else "")


def plan(target: float, timings: dict) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python."""
    files = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    entries = []
    for python in PYTHONS:
        weighed = weights(files, timings, python)
        tasks: list[tuple[list[str], str]] = []
        for name, seconds in weighed.items():
            if seconds > target:
                groups = math.ceil(seconds / target)
                tasks += [([name], f"{group}/{groups}") for group in range(1, groups + 1)]
        light = {name: seconds for name, seconds in weighed.items() if seconds <= target}
        bins: list[tuple[float, int, list[str]]] = [
            (0.0, index, []) for index in range(max(math.ceil(sum(light.values()) / target), 1))]
        for name, seconds in sorted(light.items(), key=lambda item: (-item[1], item[0])):
            held, index, names = heapq.heappop(bins)
            heapq.heappush(bins, (held + seconds, index, [*names, name]))
        tasks += [(sorted(names), "") for _, _, names in sorted(bins, key=lambda bin_: bin_[1]) if names]
        for index, (names, split) in enumerate(tasks, start=1):
            share = int(split.split("/")[1]) if split else 1
            entries.append({"name": f"{python}-{index}", "python": python, "tests": " ".join(names),
                            "split": split, "rows": [row(python, name, split) for name in names],
                            "weight": sum(weighed[name] for name in names) / share})
    return entries


def file_of(case: ET.Element) -> str:
    """The test file a JUnit case is from: its class path, or a collection error's own name."""
    parts = (case.get("classname") or case.get("name") or "").split(".")
    module = next((index for index, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "/".join(parts[:module + 1]) + ".py"


def task(python: str, tests: list[str], split: str, out: Path) -> int:
    """Run `tests` and write one verdict row a file."""
    report = out.with_name(f"{out.name}.junit.xml")
    report.unlink(missing_ok=True)
    grouping = ["--splits", split.split("/")[1], "--group", split.split("/")[0], "--splitting-algorithm",
                "duration_based_chunks", "--durations-path", str(DURATIONS)] if split else []
    # faulthandler prints every thread's stack when a test runs ten minutes.
    command = [f".venv-{python}/bin/python", "-m", "pytest", "-q", "-m", "not network", "-rfE", "--tb=short",
               "--continue-on-collection-errors", "-o", "faulthandler_timeout=600",
               "-p", "no:cacheprovider", f"--junitxml={report}", *tests, *grouping]
    # Streamed as it comes, so a task armada stops at its timeout still shows where it was.
    lines = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as done:
        for line in done.stdout or ():
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
    output = "".join(lines)
    cases = list(ET.parse(report).getroot().iter("testcase")) if report.is_file() else []
    rows = []
    for name in tests:
        own = [case for case in cases if file_of(case) == name]
        red = [f"{kind.upper()} {case.get('classname')}::{case.get('name')}: {node.get('message')}"
               for case in own for kind in ("failure", "error") if (node := case.find(kind)) is not None]
        red += [f"SKIPPED on a missing import {case.get('classname')}::{case.get('name')}: "
                f"{node.get('message')}" for case in own for node in case.iter("skipped")
                if "could not import" in f"{node.get('message')} {node.text}"]
        if not own and done.returncode not in (0, 5):
            red.append(f"pytest exited {done.returncode} with no report of {name}")
        seconds = sum(float(case.get("time") or 0) for case in own)
        rows.append({"name": row(python, name, split), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([output[-4000:]] if red else [])),
                     **({} if split else {"timings": {name: seconds}})})
    out.write_text(json.dumps({"rows": rows}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/armada/ci.py")
    operations = parser.add_subparsers(dest="operation", required=True)
    planning = operations.add_parser("plan")
    planning.add_argument("--target", type=float, required=True)
    planning.add_argument("--timings", type=Path, required=True)
    running = operations.add_parser("task")
    running.add_argument("--python", choices=PYTHONS, required=True)
    running.add_argument("--tests", required=True)
    running.add_argument("--split", default="")
    running.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "plan":
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps({"include": plan(args.target, timings)}))
        return 0
    return task(args.python, args.tests.split(), args.split, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

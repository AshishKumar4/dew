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
CI would never have run), or when pytest left no report of it. A task still
running at its deadline, short of armada's own timeout, is interrupted and
every row it holds is red, so a hang is graded with its stack dump rather
than leaving the task without a verdict.
"""

import argparse
import contextlib
import heapq
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PYTHONS = ("3.12", "3.14")
"""The compatibility floor, which runs every test, and the newest release CI proves."""
NEWEST_FILES = (
    # Annotations, which 3.14 evaluates lazily (PEP 649): records and their config (326e40fd5).
    "tests/test_config.py", "tests/test_encoders.py", "tests/test_registry_records.py",
    "tests/test_records.py", "tests/test_record_keys.py", "tests/test_linen_records.py",
    "tests/test_run_records.py",
    # The command line, through tyro's reading of those annotations.
    "tests/test_config_cli.py", "tests/test_tpu_cli.py",
    # Serialization.
    "tests/test_pickle_sources.py",
    # Data loading: providers without TensorFlow (54a52a57d, 1a276e6a5, d3cc21fe6) and a
    # fetcher pool under 3.14's forkserver start (c9dbce717).
    "tests/test_data.py", "tests/test_providers.py", "tests/test_hf_data.py", "tests/test_data_mixtures.py",
    "tests/test_online_loader.py",
    # Threads at interpreter exit, which 3.14 ends differently (13a271177, d00ed72cd), and the
    # compilation cache's codec (23f5adffe).
    "tests/test_iterator_lifecycle.py", "tests/test_native_rollout_server.py", "tests/test_distribution.py",
    "tests/test_instrumentation.py",
)
"""The files the newest Python runs whole: those its changes broke before, and the ones around
the same machinery. It imports every other test file (`COLLECT`). Across 34 suite runs no file
was red on 3.14 alone."""
COLLECT = "collect"
"""The split of a task that imports test files without running them."""
COLLECT_TASKS = 4
SKIPPED = {"tests/test_gen_api.py", "tests/test_multihost.py"}
"""CI runs the first in the lint job, under the griffe it pins, and the second across a gang (`MULTIHOST`)."""
MULTIHOST = {"name": "multihost", "gang": 4, "python": "3.12", "tests": "tests/test_multihost.py",
             "split": "", "rows": ["3.12:tests/test_multihost.py"], "weight": 1800}
"""The multi-host pools, run on demand (`plan --multihost`): four containers of a gang, one host each."""
DURATIONS = Path("tests/test_durations.json")


def weights(files: list[str], timings: dict, python: str) -> dict[str, float | None]:
    """Each file's seconds on armada at `python`: its median, a split file's
    as the sum of its last complete set of groups; else its tests' sum in
    tests/test_durations.json, scaled by how much slower armada ran the
    files measured both ways; else None, for a file nothing has timed."""
    recorded: dict[str, float] = {}
    if DURATIONS.is_file():
        for node, seconds in json.loads(DURATIONS.read_text()).items():
            recorded[node.split("::")[0]] = recorded.get(node.split("::")[0], 0.0) + seconds
    measured = dict(timings.get("files", {}))
    groups: dict[tuple[str, int], dict[int, float]] = {}
    for name, seconds in timings.get("rows", {}).items():
        own, _, rest = name.partition(":")
        file, _, split = rest.partition("#")
        if own == python and split and split != COLLECT:
            group, count = (int(part) for part in split.split("/"))
            groups.setdefault((file, count), {})[group] = seconds
    for (file, count), seconds in sorted(groups.items()):
        if len(seconds) == count:
            measured[file] = sum(seconds.values())
    ratios = sorted(measured[name] / recorded[name] for name in files
                    if name in measured and recorded.get(name, 0) > 1)
    slower = ratios[len(ratios) // 2] if ratios else 1.0
    return {name: measured.get(name, recorded[name] * slower if name in recorded else None) for name in files}


def row(python: str, name: str, split: str) -> str:
    return f"{python}:{name}" + (f"#{split}" if split else "")


def plan(target: float, timings: dict) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has
    timed runs alone, so however long it takes it holds up no other file, and its time is known from
    then on."""
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    entries = []
    for python in PYTHONS:
        files = every if python == PYTHONS[0] else [name for name in every if name in NEWEST_FILES]
        timed = weights(files, timings, python)
        weighed = {name: target if seconds is None else seconds for name, seconds in timed.items()}
        tasks: list[tuple[list[str], str]] = [
            ([name], "") for name, seconds in timed.items() if seconds is None]
        for name, seconds in timed.items():
            if seconds is not None and seconds > target:
                groups = math.ceil(seconds / target)
                tasks += [([name], f"{group}/{groups}") for group in range(1, groups + 1)]
        light = {name: seconds for name, seconds in timed.items()
                 if seconds is not None and seconds <= target}
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
    rest = [name for name in every if name not in NEWEST_FILES]
    for index in range(COLLECT_TASKS):
        names = rest[index::COLLECT_TASKS]
        entries.append({"name": f"{PYTHONS[1]}-{COLLECT}-{index + 1}", "python": PYTHONS[1],
                        "tests": " ".join(names), "split": COLLECT,
                        "rows": [row(PYTHONS[1], name, COLLECT) for name in names], "weight": target})
    return entries


def file_of(case: ET.Element) -> str:
    """The test file a JUnit case is from: its class path, or a collection error's own name."""
    parts = (case.get("classname") or case.get("name") or "").split(".")
    module = next((index for index, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "/".join(parts[:module + 1]) + ".py"


def signal_group(leader: int, number: signal.Signals) -> None:
    """`number` to every process left in the group `leader` led."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(leader, number)


def shown(output: str) -> str:
    """The end of a task's output, and the stack faulthandler printed for a test that hung, if one did."""
    stack = output.rfind("Timeout (")
    hung = output[stack:stack + 6000] + "\n...\n" if 0 <= stack < len(output) - 4000 else ""
    return hung + output[-4000:]


def collect(python: str, tests: list[str], out: Path, deadline: float) -> int:
    """Import `tests` under `python` without running them and write one verdict row a file: red
    where collecting it errs or skips on a missing import, which running it would have."""
    began = time.monotonic()
    command = [f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q", "-rs", "-p",
               "no:cacheprovider", *tests]
    done = subprocess.run(command, capture_output=True, text=True, timeout=deadline)
    output = done.stdout + done.stderr
    errors = dict.fromkeys(re.findall(r"ERROR collecting (tests/\S+\.py)", output), "")
    section = r"_+ ERROR collecting (tests/\S+\.py) _+\n(.*?)(?=\n_{3,} |\n={3,} )"
    for header in re.finditer(section, output, re.S):
        errors[header.group(1)] = header.group(2)[-4000:]
    for name, reason in re.findall(r"SKIPPED \[\d+\] (tests/\S+\.py):\d+: (could not import[^\n]*)", output):
        errors.setdefault(name, f"SKIPPED on a missing import: {reason}")
    if done.returncode not in (0, 2, 5) and not errors:
        errors = dict.fromkeys(tests, f"pytest --collect-only exited {done.returncode}\n{output[-4000:]}")
    seconds = (time.monotonic() - began) / len(tests)
    rows = [{"name": row(python, name, COLLECT), "exitCode": 1 if name in errors else 0, "seconds": seconds,
             "output": errors.get(name, "")} for name in tests]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, grace: float = 60.0) -> int:
    """Run `tests` and write one verdict row a file, interrupting them past `deadline` seconds and
    killing them `grace` seconds after that.

    A row's seconds are its share of the task's wall time, in proportion to its tests' own times:
    the plan weighs a file by what it costs a container (its import, its collection, the
    container's pace), which pytest's case times leave out. Weighed by case times, tasks planned
    at 300 seconds ran a median of 392."""
    began = time.monotonic()
    report = out.with_name(f"{out.name}.junit.xml")
    report.unlink(missing_ok=True)
    grouping = ["--splits", split.split("/")[1], "--group", split.split("/")[0], "--splitting-algorithm",
                "duration_based_chunks", "--durations-path", str(DURATIONS)] if split else []
    # faulthandler prints every thread's stack when a test runs ten minutes. Unbuffered (-u), so the
    # log holds every test that finished when a task is cut short.
    # A gang's ranks run the same tests, which name their files under one root on every host.
    ganged = ["--basetemp=/tmp/dew-gang"] if int(os.environ.get("ARMADA_WORLD", "1")) > 1 else []
    command = [f".venv-{python}/bin/python", "-u", "-m", "pytest", "-q", "-m", "not network", "-rfE",
               "--tb=short", "--continue-on-collection-errors", "-o", "faulthandler_timeout=600",
               "-p", "no:cacheprovider", f"--junitxml={report}", *ganged, *tests, *grouping]
    # pytest leads a process group of its own, so what its tests start ends with it. Its output is
    # streamed as it comes by a reader of its own, which nothing waits on past pytest's end: a test's
    # child holding the pipe open (a pool worker, a server) cannot keep the verdict from being written.
    lines: list[str] = []
    done = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)

    def read() -> None:
        for line in done.stdout or ():
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    expired = False
    try:
        done.wait(deadline)
    except subprocess.TimeoutExpired:
        # ^C first, so pytest reports what ran; a test stuck in C ignores it, and the group is killed.
        expired = True
        signal_group(done.pid, signal.SIGINT)
        try:
            done.wait(grace)
        except subprocess.TimeoutExpired:
            signal_group(done.pid, signal.SIGKILL)
            done.wait()
    signal_group(done.pid, signal.SIGKILL)
    reader.join(10)
    output = "".join(lines)
    cases = list(ET.parse(report).getroot().iter("testcase")) if report.is_file() else []
    rows = []
    own_seconds = {name: sum(float(case.get("time") or 0) for case in cases if file_of(case) == name)
                   for name in tests}
    wall = time.monotonic() - began
    pace = wall / sum(own_seconds.values()) if sum(own_seconds.values()) > 0 else 0.0
    for name in tests:
        own = [case for case in cases if file_of(case) == name]
        red = [f"{kind.upper()} {case.get('classname')}::{case.get('name')}: {node.get('message')}"
               for case in own for kind in ("failure", "error") if (node := case.find(kind)) is not None]
        red += [f"SKIPPED on a missing import {case.get('classname')}::{case.get('name')}: "
                f"{node.get('message')}" for case in own for node in case.iter("skipped")
                if "could not import" in f"{node.get('message')} {node.text}"]
        if not own and done.returncode not in (0, 5):
            red.append(f"pytest exited {done.returncode} with no report of {name}")
        if expired:
            red.append(f"the task ran past its deadline of {deadline:.0f} s and was interrupted")
        seconds = own_seconds[name] * pace if pace else wall / len(tests)
        rows.append({"name": row(python, name, split), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([shown(output)] if red else [])),
                     **({} if split else {"timings": {name: seconds}})})
    out.write_text(json.dumps({"rows": rows}))
    # A gang's task reports rank 0's rows; another rank says its own red by its exit, which the gang's
    # outcome takes.
    other_rank = int(os.environ.get("ARMADA_RANK", "0")) > 0
    return 1 if other_rank and any(entry["exitCode"] for entry in rows) else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/armada/ci.py")
    operations = parser.add_subparsers(dest="operation", required=True)
    planning = operations.add_parser("plan")
    planning.add_argument("--target", type=float, required=True)
    planning.add_argument("--timings", type=Path, required=True)
    planning.add_argument("--multihost", action="store_true", help="plan the multi-host pools alone")
    running = operations.add_parser("task")
    running.add_argument("--python", choices=PYTHONS, required=True)
    running.add_argument("--tests", required=True)
    running.add_argument("--split", default="")
    running.add_argument("--out", type=Path, required=True)
    running.add_argument("--deadline", type=float, default=1680.0,
                         help="seconds before armada's own task timeout (.armada.json) to interrupt at")
    args = parser.parse_args()
    if args.operation == "plan":
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps({"include": [MULTIHOST] if args.multihost else plan(args.target, timings)}))
        return 0
    if args.split == COLLECT:
        return collect(args.python, args.tests.split(), args.out, args.deadline)
    return task(args.python, args.tests.split(), args.split, args.out, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT...

The plan prints a task matrix for each Python CI proves. Whole test files are
packed heaviest first into the lightest of the tasks, each near `target`
seconds, so a task imports only its own modules. A file heavier than that runs
as groups of its tests (`groups`, tools/armada/split.py): runs of consecutive
tests in node-id order, as many as it takes to hold each to `target` by their
times in tests/test_durations.json, a heavier test alone, and each as light as
that many runs let it be. A file weighs armada's
median of its last green runs, else its tests' sum in
tests/test_durations.json, else the mean. Each entry names its rows, one a
file (or a file's group), so armada grades a task that leaves one out as not
green, and armada queues the heaviest entries first.

A task runs its files under its Python's environment (tools/armada/install.sh)
with CI's selection, and writes a verdict row for each. A row is red on a
failed or erroring test, on a test skipped because its import failed (a file
CI would never have run), or when pytest left no report of it. A task still
running at its deadline, short of armada's own timeout, is interrupted and
every row it holds is red, so a hang is graded with its stack dump rather
than leaving the task without a verdict. Each row also carries its tests' own
times, from which `durations` rewrites tests/test_durations.json: given green
runs' verdicts (`armada verdict <sha> --json`), the newest first, it records
each test of the newest at its least time as the floor's Python ran it.
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
"""CI runs the first in the lint job, under the griffe it pins, and the second in the multihost job,
across the containers of an armada gang."""
DURATIONS = Path("tests/test_durations.json")


def recorded_tests() -> dict[str, list[float]]:
    """Each file's tests' times in tests/test_durations.json, in node-id order."""
    tests: dict[str, list[float]] = {}
    if DURATIONS.is_file():
        for node, seconds in sorted(json.loads(DURATIONS.read_text()).items()):
            tests.setdefault(node.split("::")[0], []).append(seconds)
    return tests


def weights(files: list[str], timings: dict, python: str = PYTHONS[0]) -> dict[str, float | None]:
    """Each file's seconds on armada under `python`: its median, which armada
    sums over a split file's groups (`task`); else its tests' sum in
    tests/test_durations.json, scaled by how much slower armada ran the files
    measured both ways; else None, for a file nothing has timed."""
    recorded = {name: sum(tests) for name, tests in recorded_tests().items()}
    measured = {name: timings.get("files", {})[row(python, name, "")] for name in files
                if row(python, name, "") in timings.get("files", {})}
    ratios = sorted(measured[name] / recorded[name] for name in files
                    if name in measured and recorded.get(name, 0) > 1)
    slower = ratios[len(ratios) // 2] if ratios else 1.0
    return {name: measured.get(name, recorded[name] * slower if name in recorded else None) for name in files}


def row(python: str, name: str, split: str) -> str:
    return f"{python}:{name}" + (f"#{split}" if split else "")


def starts(weights: list[int], capacity: int) -> list[int]:
    """Where each run of consecutive `weights` begins, each run filled up to `capacity`, a heavier
    weight alone."""
    begins, held = [0], 0
    for index, weight in enumerate(weights):
        if held and held + weight > capacity:
            begins.append(index)
            held = 0
        held += weight
    return begins


def milliseconds(seconds: list[float]) -> list[int]:
    return [round(max(value, 0.0) * 1000) for value in seconds]


def groups(seconds: list[float], count: int) -> list[range]:
    """`count` runs of consecutive `seconds`, each filled up to the least capacity that needs no more
    runs, a heavier test alone: how every task of a file split `count` ways cuts the file's tests
    (split.py), in whole milliseconds, so each container cuts them alike. The runs beside a test
    heavier than the rest are no heavier than they must be. A run past the last test is empty."""
    weights = milliseconds(seconds)
    low, high = 0, sum(weights)
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if len(starts(weights, middle)) <= count else (middle + 1, high)
    bounds = [*starts(weights, low), len(weights)]
    bounds += [len(weights)] * (count + 1 - len(bounds))
    return [range(bounds[index], bounds[index + 1]) for index in range(count)]


def split(tests: list[float], seconds: float, target: float) -> list[float]:
    """The weights of the groups a file of `seconds` runs as: its recorded `tests`' times scaled to
    `seconds`, cut (`groups`) into as many runs as it takes to hold each to `target`, a heavier test
    alone. A file with no recorded time runs as equal counts of its tests."""
    if sum(tests) <= 0:
        count = math.ceil(seconds / target)
        return [seconds / count] * count
    scale = seconds / sum(tests)
    count = len(starts(milliseconds(tests), round(target / scale * 1000)))
    return [sum(tests[run.start:run.stop]) * scale for run in groups(tests, count) if run]


def plan(target: float, timings: dict) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has
    timed runs alone, so however long it takes it holds up no other file, and its time is known from
    then on."""
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    recorded = recorded_tests()
    entries = []
    for python in PYTHONS:
        files = every if python == PYTHONS[0] else [name for name in every if name in NEWEST_FILES]
        timed = weights(files, timings, python)
        tasks: list[tuple[list[str], str, float]] = [
            ([name], "", target) for name, seconds in timed.items() if seconds is None]
        for name, seconds in timed.items():
            if seconds is not None and seconds > target:
                cut = split(recorded.get(name, []), seconds, target)
                tasks += [([name], f"{group}/{len(cut)}", weight)
                          for group, weight in enumerate(cut, start=1)]
        light = {name: seconds for name, seconds in timed.items()
                 if seconds is not None and seconds <= target}
        bins: list[tuple[float, int, list[str]]] = [
            (0.0, index, []) for index in range(max(math.ceil(sum(light.values()) / target), 1))]
        for name, seconds in sorted(light.items(), key=lambda item: (-item[1], item[0])):
            held, index, names = heapq.heappop(bins)
            heapq.heappush(bins, (held + seconds, index, [*names, name]))
        tasks += [(sorted(names), "", held) for held, _, names in sorted(bins, key=lambda bin_: bin_[1])
                  if names]
        for index, (names, group, weight) in enumerate(tasks, start=1):
            entries.append({"name": f"{python}-{index}", "python": python, "tests": " ".join(names),
                            "split": group, "rows": [row(python, name, group) for name in names],
                            "weight": weight})
    rest = [name for name in every if name not in NEWEST_FILES]
    for index in range(COLLECT_TASKS):
        names = rest[index::COLLECT_TASKS]
        entries.append({"name": f"{PYTHONS[1]}-{COLLECT}-{index + 1}", "python": PYTHONS[1],
                        "tests": " ".join(names), "split": COLLECT,
                        "rows": [row(PYTHONS[1], name, COLLECT) for name in names], "weight": target})
    return entries


def node_of(case: ET.Element) -> str:
    """A JUnit case's pytest node id, as tests/test_durations.json keys it."""
    parts = (case.get("classname") or "").split(".")
    module = next((index for index, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "/".join(parts[:module + 1]) + ".py::" + "::".join([*parts[module + 1:], case.get("name") or ""])


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
    grouping = ["-p", "split", f"--ci-group={split}"] if split else []
    # faulthandler prints every thread's stack when a test runs ten minutes. Unbuffered (-u), so the
    # log holds every test that finished when a task is cut short.
    command = [f".venv-{python}/bin/python", "-u", "-m", "pytest", "-q", "-m", "not network", "-rfE",
               "--tb=short", "--continue-on-collection-errors", "-o", "faulthandler_timeout=600",
               "-p", "no:cacheprovider", f"--junitxml={report}", *tests, *grouping]
    # pytest leads a process group of its own, so what its tests start ends with it. Its output is
    # streamed as it comes by a reader of its own, which nothing waits on past pytest's end: a test's
    # child holding the pipe open (a pool worker, a server) cannot keep the verdict from being written.
    lines: list[str] = []
    # split.py, the plugin that keeps a split file's group, imports from beside this file.
    path = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")]))
    done = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, env={**os.environ, "PYTHONPATH": path})

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
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart.
        rows.append({"name": row(python, name, split), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([shown(output)] if red else [])),
                     "tests": {node_of(case): float(case.get("time") or 0) for case in own},
                     "timings": {row(python, name, ""): seconds}})
    out.write_text(json.dumps({"rows": rows}))
    return 0


def durations(verdicts: list[dict]) -> dict[str, float]:
    """Each test of the first of green runs' `verdicts`, the newest, at its least time over them all,
    as their floor's Python ran it: what it takes where nothing slows it. A container can run three
    times as slowly as the rest of its run, and a run's compilations can miss the shared cache, and
    neither moves a test's time. A test the newest run did not have is gone."""
    times: dict[str, list[float]] = {}
    for verdict in verdicts:
        rows = verdict["rows"]
        red = [row["name"] for row in rows if row["exitCode"] != 0]
        if red:
            raise SystemExit(f"{len(red)} rows are red, {red[0]} first: record durations from green runs")
        floor = [row for row in rows if row["name"].startswith(f"{PYTHONS[0]}:")]
        if not all("tests" in row for row in floor):
            raise SystemExit("a verdict's rows carry no test times; record them from runs of this version")
        for row in floor:
            for node, seconds in row["tests"].items():
                times.setdefault(node, []).append(seconds)
    newest = {node for row in verdicts[0]["rows"] if row["name"].startswith(f"{PYTHONS[0]}:")
              for node in row["tests"]}
    return {node: round(min(seconds), 3) for node, seconds in sorted(times.items()) if node in newest}


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
    running.add_argument("--deadline", type=float, default=1680.0,
                         help="seconds before armada's own task timeout (.armada.json) to interrupt at")
    recording = operations.add_parser("durations")
    recording.add_argument("verdicts", type=Path, nargs="+",
                           help="green runs' verdicts, the newest first: armada verdict <sha> --json")
    args = parser.parse_args()
    if args.operation == "durations":
        recorded = durations([json.loads(path.read_text()) for path in args.verdicts])
        DURATIONS.write_text(json.dumps(dict(sorted(recorded.items())), indent=0) + "\n")
        print(f"{len(recorded)} tests, {sum(recorded.values()):.0f} s, in {DURATIONS}")
        return 0
    if args.operation == "plan":
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps({"include": plan(args.target, timings)}))
        return 0
    if args.split == COLLECT:
        return collect(args.python, args.tests.split(), args.out, args.deadline)
    return task(args.python, args.tests.split(), args.split, args.out, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT...
    python3 tools/armada/ci.py gpu-tier [--marked] [--evidence] [--shards=N]
    python3 tools/armada/ci.py gpu-lane --tests="tests/a.py tests/b.py" --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py coverage items --target=SECONDS --timings=FILE > ITEMS
    python3 tools/armada/ci.py coverage report DIR

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

The test cache reuses a file's row when its inputs are unchanged since it ran green. A
green row records the checkout's Python files its tests executed
(tools/armada/executed.py) and its key, a hash of those files, the file
itself and what every file shares (`shared`: the environment's locks, the
CI tools, every non-Python file under tests/ and src/, and the names of every
Python file there, so a file a scanner would find changes every key). A task
recomputes each of its files' keys from the record made on its own class
(`device_class`: float32 results differ by CPU microarchitecture, and by GPU)
and reports the recorded green row
instead of running a file whose key matches, which ran green on exactly these
inputs there. A file that started another Python process is never reused, since
no event saw that process's code. The records live in the dew-xla-cache Worker
under `CACHE_PREFIX`. With DEW_TEST_CACHE=off where the plan runs, which its
entries carry to their tasks (`--cache`), or without the Worker's URL and
token, a task runs every file.

A row's record also says where its tests read the backend and which are
marked to run on a GPU, from which `gpu-tier` names the files that take
another path on a GPU (all of them daily, the marked ones per batch). A GPU
node runs them with `gpu-lane`, each as a task of a CUDA lane, whose rows are
recorded and reused on the GPU's class as a CPU task's are on the CPU's.

`coverage` measures src/dew's branch coverage over the suite, on demand and never
per batch: `items` prints the plan's tasks for the oldest Python as armada map
items, each running its files under coverage.py (pyproject.toml's
[tool.coverage.run], the Python processes its tests start included) with the
test cache off, since a reused row covers nothing, and writing its data among
the task's artifacts; a traced task records nothing in the cache either.
`report` combines the artifacts' data under DIR and writes DIR/coverage.json and
DIR/coverage.md, a module a line, the least covered first:

    armada map --artifacts=DIR --items=ITEMS -- bash -lc {item}
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_CLASSIFIERS = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["classifiers"]
_PREFIX = "Programming Language :: Python :: "
PYTHONS = tuple(sorted((classifier.removeprefix(_PREFIX) for classifier in _CLASSIFIERS
                        if re.fullmatch(re.escape(_PREFIX) + r"\d+\.\d+", classifier)),
                       key=lambda version: tuple(int(part) for part in version.split("."))))
"""The tested versions pyproject.toml's classifiers declare, oldest first: the floor runs every
test, the newest the files whose behaviour changed with it (`NEWEST_FILES`). `pythons` prints them
for tools/armada/install.sh, and `pythons --check` holds .python-version and the environment's
smoke (.armada.json) to them."""
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
SHARED = ("pyproject.toml", "constraints.txt", ".armada.json", "tools/armada/ci.py", "tools/armada/split.py",
          "tools/armada/executed.py", str(DURATIONS))
"""The files every test file's key covers whole, beside tests/ and src/, every lock (a GPU node
builds its environment from its CUDA lock) and the environment's own inputs, which .armada.json
names (`environment_files`)."""


def environment_files() -> list[str]:
    """The files .armada.json builds the tasks' environments from: each environment's setup and
    install scripts and its snapshot key's files. A shell script runs no Python a record could see."""
    if not Path(".armada.json").is_file():  # a project armada does not run, as the CI tool's tests make
        return []
    config = json.loads(Path(".armada.json").read_text())
    environments = [config["environment"], *config.get("environments", {}).values()]
    return sorted({name for environment in environments
                   for name in (environment.get("setup"), environment.get("install"),
                                *environment.get("key", ()))
                   if name})
CACHE_PREFIX = "testcache/" + hashlib.sha256(
    Path(__file__).read_bytes() + Path(__file__).with_name("executed.py").read_bytes()).hexdigest()[:16]
"""Where records live: one namespace per version of the tools that write and read them, so a branch
grading with a changed record format never overwrites or breaks main's records."""


def shared(python: str) -> str:
    """The hash of what every test file's run shares: `python`, the `SHARED` files, every lock, every
    non-Python file under tests/ and src/, and the names of every Python file there."""
    digest = hashlib.sha256(python.encode())
    for top in ("tests", "src"):
        for path in sorted(Path(top).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path).encode())
            if path.suffix != ".py":
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    for name in sorted({*SHARED, *map(str, Path("locks").glob("*.txt")), *environment_files()}):
        digest.update(name.encode() + contents(name))
    return digest.hexdigest()


def contents(name: str) -> bytes:
    """The hash of file `name`'s bytes, or b"missing"."""
    return hashlib.sha256(Path(name).read_bytes()).digest() if Path(name).is_file() else b"missing"


def cache_key(common: str, name: str, depends: list[str]) -> str:
    """The key of test file `name` whose run executed `depends`: those files' contents and `common`."""
    digest = hashlib.sha256(common.encode())
    for path in sorted({name, *depends}):
        digest.update(path.encode() + contents(path))
    return digest.hexdigest()


def cpu_class() -> str:
    """This host's CPU microarchitecture (vendor, family, model). XLA:CPU emits the same code on every
    host, but hardware estimate instructions (vrsqrtps) differ between microarchitectures, so float32
    results, and thus numerical verdicts, belong to the class that produced them."""
    fields = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("vendor_id", "cpu family", "model") and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    return "-".join(fields.get(key, "unknown") for key in ("vendor_id", "cpu family", "model"))


def device_class() -> str:
    """The class a task's results belong to: on a CUDA lane (`JAX_PLATFORMS`, as
    tests/lane_environment.py reads it) the models of the GPUs it sees as the driver names them,
    before the CPU's class, whose float32 the lane's CPU devices compute (`cpu_class`); else the
    CPU's class alone."""
    if "cuda" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        return cpu_class()
    return "-".join([*(re.sub(r"[^A-Za-z0-9.]+", "-", model) for model in sorted(set(gpu_query("name")))),
                     cpu_class()])


def gpu_query(field: str) -> list[str]:
    """`nvidia-smi`'s `field` of each GPU this process sees (`CUDA_VISIBLE_DEVICES`, when set)."""
    given = os.environ.get("CUDA_VISIBLE_DEVICES")
    return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                           *(["-i", given] if given else [])], capture_output=True, text=True,
                          check=True).stdout.split("\n")[:-1]


def worker() -> str:
    """The URL of the Worker that holds the records."""
    return os.environ["DEW_XLA_CACHE_URL"].rstrip("/")


def cache_url(python: str, row_name: str, held: str, inputs: str | None = None) -> str:
    """Where the record of `row_name` on class `held` (`device_class`) lives: the last run's, or with
    `inputs`, the last of the runs on those shared inputs (`shared`), apart from the last runs' list
    (`listed`). Any run of another commit replaces the last record, so a commit's retry reads its own
    under its inputs first (`reused`). The Worker's key alphabet has no colon or hash."""
    _, name, split = parsed_row(row_name)
    at = f"{CACHE_PREFIX}/{python}" if inputs is None else f"{CACHE_PREFIX}-inputs/{inputs[:16]}/{python}"
    return f"{worker()}/{at}/{held}/{name}/" + ("-of-".join(split.split("/")) if split else "whole")


def cache_request(url: str, token: str, data: bytes | None = None) -> bytes | None:
    """GET (or with `data`, PUT) `url`: None on a miss, and on a failure, which it reports and which
    costs only a rerun."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "dew-ci-test-cache"}
    method = "PUT" if data is not None else "GET"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code != 404:
            print(f"the test cache answered {error.code} to {method} {url}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"the test cache is unreachable ({error}); {method} {url} skipped", file=sys.stderr)
        return None


def reused(python: str, name: str, split: str, common: str) -> dict | None:
    """The green row of test file `name` (in `split`'s group) recorded on this host's class
    (`device_class`), if it ran green there on exactly these inputs; else None. The record of the
    last run on these shared inputs decides: a key hashes them (`cache_key`), so no run on others,
    whose record the last run's may be, can be fresh here."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return None
    record = cache_request(cache_url(python, row(python, name, split), device_class(), common), token)
    recorded = None if record is None else json.loads(record)
    return recorded["row"] if recorded is not None and fresh(recorded, name, common) else None


def fresh(recorded: dict, name: str, common: str) -> bool:
    """Whether test file `name`'s record was made by a green run on exactly these inputs. A red run,
    or one of a file that started another Python process, whose code no event saw, leaves no key."""
    return recorded["key"] is not None and recorded["key"] == cache_key(common, name, recorded["depends"])


RECORD_TRIES = 4
"""How many times a record's write is tried, two, four and eight seconds apart."""


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


def parsed_row(row_name: str) -> tuple[str, str, str]:
    """`row`'s (python, name, split) back from its name."""
    python, rest = row_name.split(":", 1)
    name, _, split = rest.partition("#")
    return python, name, split


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


APART = re.compile(r"\bpytest\.mark\.torchax\b")
"""A file marked `torchax` imports torchax, which renames torch's privateuse1 backend to "jax" for
the whole process, so a torch.compile in a later file of its task fails (test_reference_helpers'
MXFP4, armada dew-20261011231242-dd32ac90): its tasks hold only files so marked."""


def packed(weights: dict[str, float], count: int) -> list[tuple[float, list[str]]]:
    """The names of `weights` packed heaviest first into the lightest of `count` bins: each bin's
    weight and its names, sorted, in bin order."""
    bins: list[tuple[float, int, list[str]]] = [(0.0, index, []) for index in range(count)]
    for name in sorted(weights, key=lambda name: (-weights[name], name)):
        held, index, names = heapq.heappop(bins)
        heapq.heappush(bins, (held + weights[name], index, [*names, name]))
    return [(held, sorted(names)) for held, _, names in sorted(bins, key=lambda bin_: bin_[1])]


def plan(target: float, timings: dict) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has
    timed runs alone, so however long it takes it holds up no other file, and its time is known from
    then on. Files marked `torchax` (`APART`) share tasks only with one another. Each entry carries
    the test cache's policy, DEW_TEST_CACHE here, to its task."""
    cache = "off" if os.environ.get("DEW_TEST_CACHE") == "off" else "on"
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
        apart = {name for name in light if APART.search(Path(name).read_text())}
        for family in (sorted(set(light) - apart), sorted(apart)):
            count = max(math.ceil(sum(light[name] for name in family) / target), 1)
            tasks += [(names, "", held) for held, names in packed({name: light[name] for name in family},
                                                                  count) if names]
        for index, (names, group, weight) in enumerate(tasks, start=1):
            entries.append({"name": f"{python}-{index}", "python": python, "tests": " ".join(names),
                            "split": group, "rows": [row(python, name, group) for name in names],
                            "weight": weight, "cache": cache})
    rest = [name for name in every if name not in NEWEST_FILES]
    for index in range(COLLECT_TASKS):
        names = rest[index::COLLECT_TASKS]
        entries.append({"name": f"{PYTHONS[-1]}-{COLLECT}-{index + 1}", "python": PYTHONS[-1],
                        "tests": " ".join(names), "split": COLLECT,
                        "rows": [row(PYTHONS[-1], name, COLLECT) for name in names], "weight": target,
                        "cache": cache})
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
    command = [f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q", "-rs",
               *tests]
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


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: bool = True,
         grace: float = 60.0, coverage: Path | None = None) -> int:
    """Run `tests` and write one verdict row a file, interrupting them past `deadline` seconds and
    killing them `grace` seconds after that.

    A row's seconds are its share of the task's wall time, in proportion to its tests' own times:
    the plan weighs a file by what it costs a container (its import, its collection, the
    container's pace), which pytest's case times leave out. Weighed by case times, tasks planned
    at 300 seconds ran a median of 392. With `coverage`, pytest runs under coverage.py, whose data
    file goes there (`coverage_items`)."""
    # Before any test writes under tests/, the inputs as the plan saw them.
    return tested(python, tests, split, out, deadline, cache, grace, shared(python), {}, coverage)


def tested(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: bool, grace: float,
           common: str, environ: Mapping[str, str], coverage: Path | None = None) -> int:
    """`task` on the inputs `common` hashes, its pytest under `environ` beside this process's."""
    began = time.monotonic()
    rows = [found for name in tests if cache and (found := reused(python, name, split, common))]
    tests = [name for name in tests if row(python, name, split) not in {found["name"] for found in rows}]
    if rows:
        print(f"{len(rows)} files ran green on these inputs on {device_class()}; their rows are reused")
    if not tests:
        out.write_text(json.dumps({"rows": rows}))
        return 0
    executed_report = out.with_name(f"{out.name}.executed.json")
    report = out.with_name(f"{out.name}.junit.xml")
    report.unlink(missing_ok=True)
    executed_report.unlink(missing_ok=True)
    grouping = ["-p", "split", f"--ci-group={split}"] if split else []
    # faulthandler prints every thread's stack when a test runs ten minutes. Unbuffered (-u), so the
    # log holds every test that finished when a task is cut short.
    # coverage.py follows the Python processes the tests start (pyproject.toml's patch), each writing
    # its data where COVERAGE_FILE says.
    measured = [] if coverage is None else ["-m", "coverage", "run"]
    environ = environ if coverage is None else {**environ, "COVERAGE_FILE": str(coverage / ".coverage")}
    command = [f".venv-{python}/bin/python", "-u", *measured, "-m", "pytest", "-q", "-m", "not network",
               "-rfE", "--tb=short", "--continue-on-collection-errors", "-o", "faulthandler_timeout=600",
               "-p", "executed", f"--junitxml={report}", *tests, *grouping]
    # pytest leads a process group of its own, so what its tests start ends with it. Its output is
    # streamed as it comes by a reader of its own, which nothing waits on past pytest's end: a test's
    # child holding the pipe open (a pool worker, a server) cannot keep the verdict from being written.
    lines: list[str] = []
    # split.py, the plugin that keeps a split file's group, imports from beside this file.
    path = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")]))
    done = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True,
                            env={**os.environ, **environ, "PYTHONPATH": path,
                                 "DEW_EXECUTED": str(executed_report)})

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
    ran = []
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
        # pytest exits 1 for failed tests and 5 for none collected; any other exit (a crash, a signal,
        # an internal error) leaves even a written report untrustworthy, so every row it holds is red.
        if done.returncode not in (0, 1, 5) or (not own and done.returncode == 1):
            red.append(f"pytest exited {done.returncode}" + ("" if own else f" with no report of {name}"))
        if expired:
            red.append(f"the task ran past its deadline of {deadline:.0f} s and was interrupted")
        seconds = own_seconds[name] * pace if pace else wall / len(tests)
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart.
        ran.append({"name": row(python, name, split), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([shown(output)] if red else [])),
                     "tests": {node_of(case): float(case.get("time") or 0) for case in own},
                     "timings": {row(python, name, ""): seconds}})
    out.write_text(json.dumps({"rows": rows + ran}))
    if coverage is None:
        # A traced run's rows, timed under tracing, are no rows to reuse.
        record(python, ran, executed_report, common)
    return 0


def record(python: str, rows: list[dict], executed_report: Path, common: str) -> None:
    """Keep each row a task ran with the files its tests executed, where they read the backend and
    its peak memory on a GPU: `gpu_tier` reads the reads, `gpu_lane` the peak. A green row keeps
    its key too, which `reused` reads it back by; a red row has none, so a later run never reuses
    an earlier green row of its file."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return
    # A run cut short, or whose collection failed, may have written no report, or none for a file.
    executed = json.loads(executed_report.read_text()) if executed_report.is_file() else {}
    held = device_class()
    for row_ in rows:
        name = parsed_row(row_["name"])[1]
        ran = executed.get(name, {"depends": None, "reads": [], "peak": None})
        if row_["exitCode"] == 0 and name not in executed:
            continue
        green = row_["exitCode"] == 0 and ran["depends"] is not None
        body = json.dumps({"key": cache_key(common, name, ran["depends"]) if green else None,
                           "depends": ran["depends"], "row": row_, "reads": ran["reads"],
                           "peak": ran["peak"]}).encode()
        # The record under these inputs, which `reused` reads, and the last run's, which the GPU tier and
        # lane read, only once the first is written, so it is never newer.
        if written(cache_url(python, row_["name"], held, common), token, body):
            written(cache_url(python, row_["name"], held), token, body)


def written(url: str, token: str, body: bytes) -> bool:
    """Whether `body` was PUT at `url`, tried `RECORD_TRIES` times: a record not written leaves the
    one before it, such as an older green row a red run should have replaced."""
    for attempt in range(RECORD_TRIES):
        if attempt:
            time.sleep(2 ** attempt)
        if cache_request(url, token, body) is not None:
            return True
    return False


def listed(prefix: str, token: str) -> list[str]:
    """The names of the Worker's records under `prefix`, without it, a page of a thousand at a time."""
    names, cursor = [], None
    while True:
        fields = {"list": prefix} | ({"cursor": cursor} if cursor else {})
        page = cache_request(f"{worker()}/?{urllib.parse.urlencode(fields)}", token)
        if page is None:
            raise SystemExit(f"the test cache did not list {prefix}")
        listing = json.loads(page)
        names += listing["keys"]
        cursor = listing["cursor"]
        if not cursor:
            return names


def gpu_tier(python: str) -> dict[str, list[str]]:
    """The test files that take another path on a GPU, each with its evidence: where its tests read
    the backend and its marks of `MARKS`, over every record of its rows,
    on every class, a split file's under each count of groups it was planned as. A file is out
    only where every record is of these inputs (`fresh`) and says neither, so a file whose inputs
    changed since it ran is in as "changed", and one without a record, or with one the Worker did not
    return, as "unrecorded": a new file, or one no green run of these tools has recorded."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        raise SystemExit("the GPU tier is chosen from the test cache's records, which need its URL and token")
    common = shared(python)
    prefix = f"{CACHE_PREFIX}/{python}/"

    def evidence_of(key: str) -> tuple[str, list[str]]:
        name = key.split("/", 1)[1].rsplit("/", 1)[0]  # <class>/<test file>/<group>
        record = cache_request(f"{worker()}/{prefix}{key}", token)
        if record is None:
            return name, ["unrecorded"]
        recorded = json.loads(record)
        return name, recorded["reads"] or ([] if fresh(recorded, name, common) else ["changed"])

    evidence: dict[str, set[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for name, found in pool.map(evidence_of, listed(prefix, token)):
            evidence.setdefault(name, set()).update(found)
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    unrecorded = {name: ["unrecorded"] for name in every if name not in evidence}
    return {name: sorted(evidence[name]) for name in every if evidence.get(name)} | unrecorded


MARKS = ("gpu", "kernel")
"""The marks of tests a GPU runs otherwise: those only a GPU runs, and a kernel's, which a CPU interprets."""


def marked(python: str) -> dict[str, int]:
    """Each test file of this checkout with tests marked one of `MARKS`, and how many, as pytest
    collects it on the CPU: what a batch's GPU run holds, whatever the records say."""
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    done = subprocess.run([f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q", "-p",
                           "no:cacheprovider", "-m", " or ".join(MARKS), *every],
                          capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    if done.returncode not in (0, 5):
        raise SystemExit(f"collecting the marked tests exited {done.returncode}\n"
                         f"{done.stdout[-4000:]}{done.stderr[-2000:]}")
    counts: dict[str, int] = {}
    for node in done.stdout.splitlines():
        if "::" in node:
            counts[node.split("::")[0]] = counts.get(node.split("::")[0], 0) + 1
    return counts


GPU_HEADROOM = (1.25, 2 ** 30)
"""A file's share of the GPU over its recorded peak, a factor and bytes beside it: the allocator's
fragmentation, which can leave a share its own size short of a block it needs."""


def gpu_lane(python: str, tests: list[str], out: Path, *, cache: bool, processes: int,
             deadline: float) -> int:
    """Run `tests` on this process's share of the one GPU it sees, each file once, as a task of its
    own (`tested`), and write their verdict rows to `out`.

    A file's process may hold the peak memory its last run on this GPU's class held
    (tools/armada/executed.py's `gpu_peak`, green or red) and `GPU_HEADROOM`; files so measured
    start heaviest first while their shares fit the share this process was given
    (`XLA_PYTHON_CLIENT_MEM_FRACTION`, as gpu-run sets it), at most `processes` at once. A file no
    run has measured, one that started another Python process (its children's memory is measured
    nowhere, and they size what they run by the device they see) and one whose last run left no
    report each run alone with the whole share. A file that runs out of a measured share is red: it took more
    than its last run, whose peak its next run is given."""
    token = os.environ["DEW_XLA_CACHE_TOKEN"]
    [total] = (int(megabytes) * 2 ** 20 for megabytes in gpu_query("memory.total"))
    budget = float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) * total
    held = device_class()

    def needs(name: str) -> float:
        record = cache_request(cache_url(python, row(python, name, ""), held), token)
        recorded = json.loads(record) if record else {"depends": None, "peak": None}
        peak = recorded["peak"]
        if recorded["depends"] is None or peak is None:
            return budget
        factor, slack = GPU_HEADROOM
        return min(peak * factor + slack, budget)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        shares = dict(zip(tests, pool.map(needs, tests), strict=True))
    # Before any test writes under tests/, the inputs every file's key hashes.
    common = shared(python)
    waiting = sorted(tests, key=lambda name: -shares[name])
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(tests)}
    running: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        while waiting or running:
            free = budget - sum(shares[name] for name in running.values())
            fits = next((name for name in waiting if shares[name] <= free), None)
            if fits is not None and len(running) < processes:
                waiting.remove(fits)
                environ = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(shares[fits] / total)}
                running[pool.submit(tested, python, [fits], "", verdicts[fits], deadline, cache, 60.0, common,
                                    environ)] = fits
                continue
            done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                del running[future]
    rows = [verdict for name in tests for verdict in json.loads(verdicts[name].read_text())["rows"]]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def durations(verdicts: list[dict]) -> dict[str, float]:
    """Each test of the first of green runs' `verdicts`, the newest, at its least time over them all,
    as their floor's Python ran it: what it takes where nothing slows it. A container can run three
    times as slowly as the rest of its run, and a run's compilations can miss the shared cache, and
    neither moves a test's time. A test the newest run did not have is gone; a reused row (the test
    cache) carries the times of the run that recorded it."""
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
    newest_rows = [row for row in verdicts[0]["rows"] if row["name"].startswith(f"{PYTHONS[0]}:")]
    newest = {node for row in newest_rows for node in row["tests"]}
    return {node: round(min(seconds), 3) for node, seconds in sorted(times.items()) if node in newest}


ARTIFACTS = "/armada/task/artifacts"
"""Where an armada task's files are kept for `armada map --artifacts`."""


def coverage_items(target: float, timings: dict) -> list[str]:
    """The plan's tasks for the oldest Python, as armada map items: each runs its files under coverage.py
    with the test cache off, its rows and data among its artifacts."""
    return [f"mkdir -p {ARTIFACTS} && python3 tools/armada/ci.py task --python={entry['python']} "
            f"--tests='{entry['tests']}' --split='{entry['split']}' --out={ARTIFACTS}/rows.json --cache=off "
            f"--coverage={ARTIFACTS}"
            for entry in plan(target, timings) if entry["python"] == PYTHONS[0] and entry["split"] != COLLECT]


def coverage_report(directory: Path) -> int:
    """Combine the coverage data `coverage_items`' tasks left under `directory` and write its report:
    coverage.py's own JSON, and a table a module a line, the least covered first. It runs where the
    artifacts were downloaded, under this interpreter, which must have coverage.py."""
    if importlib.util.find_spec("coverage") is None:
        print(f"coverage.py is not importable from {sys.executable}; install the test extra", file=sys.stderr)
        return 1
    python = sys.executable
    data = directory / ".coverage"
    parts = sorted(str(path) for path in directory.glob("*/.coverage.*"))
    if not parts:
        print(f"no coverage data under {directory}", file=sys.stderr)
        return 1
    subprocess.run([python, "-m", "coverage", "combine", "--keep", f"--data-file={data}", *parts], check=True)
    report = directory / "coverage.json"
    subprocess.run([python, "-m", "coverage", "json", f"--data-file={data}", "-o", str(report)], check=True)
    files = json.loads(report.read_text())["files"]
    rows = sorted((measured["summary"]["percent_covered"], name, measured["summary"])
                  for name, measured in files.items())
    lines = ["| module | statements | missed | branches | partial | covered |", "|---|---|---|---|---|---|"]
    lines += [f"| {name} | {summary['num_statements']} | {summary['missing_lines']} "
              f"| {summary['num_branches']} | {summary['num_partial_branches']} | {percent:.1f}% |"
              for percent, name, summary in rows]
    (directory / "coverage.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} modules from {len(parts)} data files: {directory / 'coverage.md'}")
    return 0


def pythons(*, check: bool) -> int:
    """Print the tested versions, one a line; with `check`, refuse the files that name them apart."""
    if not check:
        print(*PYTHONS, sep="\n")
        return 0
    problems = []
    pinned = (REPO / ".python-version").read_text().strip()
    if pinned != PYTHONS[0]:
        problems.append(f".python-version is {pinned}, and the oldest tested version is {PYTHONS[0]}")
    smoke = json.loads((REPO / ".armada.json").read_text())["environment"]["smoke"]
    smoked = tuple(re.findall(r"\.venv-(\d+\.\d+)/", smoke))
    if smoked != PYTHONS:
        problems.append(f".armada.json's smoke imports dew in {smoked}; the tested versions are {PYTHONS}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


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
    running.add_argument(
        "--cache", choices=("on", "off"), required=True,
        help="whether a file that ran green on these inputs is reused (the plan's DEW_TEST_CACHE)")
    running.add_argument("--deadline", type=float, default=1680.0,
                         help="seconds before armada's own task timeout (.armada.json) to interrupt at")
    running.add_argument("--coverage", type=Path, default=None,
                         help="run the files under coverage.py, its data file in this directory")
    measuring = operations.add_parser("coverage")
    stages = measuring.add_subparsers(dest="stage", required=True)
    itemizing = stages.add_parser("items")
    itemizing.add_argument("--target", type=float, required=True)
    itemizing.add_argument("--timings", type=Path, required=True)
    reporting = stages.add_parser("report")
    reporting.add_argument("directory", type=Path)
    choosing = operations.add_parser("gpu-tier")
    choosing.add_argument("--evidence", action="store_true", help="each file's backend reads and marks too")
    choosing.add_argument("--marked", action="store_true",
                          help="only the files with a test marked to run on a GPU, as collected here")
    choosing.add_argument("--shards", type=int, default=1, help="how many GPUs share the tier")
    lane = operations.add_parser("gpu-lane")
    lane.add_argument("--tests", required=True)
    lane.add_argument("--out", type=Path, required=True)
    lane.add_argument("--cache", choices=("on", "off"), required=True)
    lane.add_argument("--processes", type=int, required=True, help="the most test files running at once")
    lane.add_argument("--deadline", type=float, default=3600.0, help="seconds a file may run")
    versions = operations.add_parser("pythons")
    versions.add_argument("--check", action="store_true",
                          help="refuse a .python-version other than the oldest, or a smoke of other versions")
    recording = operations.add_parser("durations")
    recording.add_argument("verdicts", type=Path, nargs="+",
                           help="green runs' verdicts, the newest first: armada verdict <sha> --json")
    args = parser.parse_args()
    if args.operation == "pythons":
        return pythons(check=args.check)
    if args.operation == "durations":
        recorded = durations([json.loads(path.read_text()) for path in args.verdicts])
        DURATIONS.write_text(json.dumps(dict(sorted(recorded.items())), indent=0) + "\n")
        print(f"{len(recorded)} tests, {sum(recorded.values()):.0f} s, in {DURATIONS}")
        return 0
    if args.operation == "gpu-tier":
        tier = ({name: [f"{count} marked"] for name, count in marked(PYTHONS[0]).items()} if args.marked
                else gpu_tier(PYTHONS[0]))
        if args.evidence:
            for name, evidence in tier.items():
                print(name, *evidence, sep="\t")
            return 0
        # Each shard's files on a line, spread by their tests' recorded times.
        recorded = recorded_tests()
        for _, names in packed({name: sum(recorded.get(name, [1.0])) for name in tier}, args.shards):
            print(*names)
        return 0
    if args.operation == "gpu-lane":
        return gpu_lane(PYTHONS[0], args.tests.split(), args.out, cache=args.cache == "on",
                        processes=args.processes, deadline=args.deadline)
    if args.operation == "coverage":
        if args.stage == "report":
            return coverage_report(args.directory)
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps(coverage_items(args.target, timings)))
        return 0
    if args.operation == "plan":
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps({"include": plan(args.target, timings)}))
        return 0
    if args.split == COLLECT:
        return collect(args.python, args.tests.split(), args.out, args.deadline)
    return task(args.python, args.tests.split(), args.split, args.out, args.deadline, args.cache == "on",
                coverage=args.coverage)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT...
    python3 tools/armada/ci.py gpu-tier [--marked] [--evidence] [--shards=N]
    python3 tools/armada/ci.py gpu-lane --tests="tests/a.py tests/b.py" --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py coverage items --target=SECONDS --timings=FILE > ITEMS
    python3 tools/armada/ci.py coverage report DIR
    python3 tools/armada/ci.py gangs

The plan prints a task matrix for each Python CI proves. Whole test files are
packed heaviest first into the lightest of the tasks, each near `target`
seconds, so a task imports only its own modules. A file heavier than that runs
as groups of its tests (`groups`, tools/armada/split.py): runs of consecutive
tests in node-id order, each as light as that many runs let it be, a heavier
test alone. Each group also pays the file's setup (its module's fixtures,
imports and compiles), which is what armada measured the file at beyond its
tests' times in tests/test_durations.json, at the cut its rows were last timed
at and shared among that cut's groups (`measured_cut`), so the file takes the
fewest groups whose heaviest, setup included, holds to `target` (`split`). A file weighs armada's
median of its last green runs, or its tests' sum in tests/test_durations.json where that is more
(those runs may predate the file's current tests), else that sum scaled to armada, else the mean.
Each entry names its rows, one a file (or a file's group), so armada grades a task that leaves
one out as not green, and armada queues the heaviest entries first.

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

    armada map --commit=SHA --artifacts=DIR --items=ITEMS --timeout=6000 --secrets=DEW_XLA_CACHE_TOKEN \
        -- sh -c {item}

A task reads the shared XLA cache as CI's do (--commit gives the checkout, its environments and the
cache's URL), and has `COVERAGE_DEADLINE` and armada's timeout past it; a pass with a task cut short is
refused (`report`), its coverage partial. A target well under CI's keeps tasks short.
"""

import argparse
import ast
import concurrent.futures
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
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


def _beside(name: str):
    """tools/armada/`name`.py, the module beside this file, loaded by its path: this file runs as a
    script and is loaded by the tests, and puts nothing on sys.path."""
    spec = importlib.util.spec_from_file_location(f"armada_{name}", Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


child_groups = _beside("child_groups")
SETTLE = 30.0
"""Seconds a task's process group has, after SIGKILL, to be reaped and gone: inside armada's own timeout
with the default deadline and grace (1680 + 60 + 30 + the reader's 10 < .armada.json's 1800)."""

REPO = Path(__file__).resolve().parents[2]
_CLASSIFIERS = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["classifiers"]
_PREFIX = "Programming Language :: Python :: "
PYTHONS = tuple(sorted((classifier.removeprefix(_PREFIX) for classifier in _CLASSIFIERS
                        if re.fullmatch(re.escape(_PREFIX) + r"\d+\.\d+", classifier)),
                       key=lambda version: tuple(int(part) for part in version.split("."))))
"""The tested versions pyproject.toml's classifiers declare, oldest first: the floor runs every
test on every run, the newest every test on the daily run (DEW_TEST_CACHE off) and on a pushed
run the files whose behaviour changed with it (`NEWEST_FILES`). `pythons` prints them for
tools/armada/install.sh, and `pythons --check` holds .python-version and the environment's smoke
(.armada.json) to them."""
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
"""The files the newest Python runs whole on a pushed run: those its changes broke before, and
the ones around the same machinery. It imports every other test file there (`COLLECT`), and runs
every file on the daily run. The list is kept by hand, from those breaks, because no record can
derive it: the test cache records the checkout's files a run executed and read (executed.py),
while each break lived in the standard library's or a dependency's behaviour under the new
Python (PEP 649's lazy annotations read through tyro, forkserver as the default start method,
threads ended at interpreter exit), which no record sees; and the plan runs where the records'
token is not. Across 34 suite runs no file was red on 3.14 alone."""
COLLECT = "collect"
"""The split of a task that imports test files without running them."""
COLLECT_TASKS = 4
SKIPPED = {"tests/test_gen_api.py", "tests/test_multihost.py"}
"""CI runs the first in the lint job, under the griffe it pins, and the second in the multihost job,
across the containers of an armada gang."""
DURATIONS = Path("tests/test_durations.json")
SHARED = ("pyproject.toml", "constraints.txt", ".armada.json", "tools/armada/ci.py",
          "tools/armada/child_groups.py", "tools/armada/split.py", "tools/armada/executed.py",
          "tools/armada/followed.py", str(DURATIONS))
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
RECORDERS = ("ci.py", "executed.py", "followed.py")
"""The tools that write and read the records, beside each other."""
CACHE_PREFIX = "testcache/" + hashlib.sha256(
    b"".join(Path(__file__).with_name(name).read_bytes() for name in RECORDERS)).hexdigest()[:16]
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
    """The hash of file `name`'s bytes, of directory `name`'s entries' names but its hidden ones and
    compiled bytecode's (a run's own leftovers), or b"missing"."""
    path = Path(name)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).digest()
    if path.is_dir():
        entries = sorted(entry.name for entry in path.iterdir()
                         if not entry.name.startswith(".") and entry.name != "__pycache__")
        return hashlib.sha256("\n".join(entries).encode()).digest()
    return b"missing"


def cache_key(common: str, name: str, depends: list[str]) -> str:
    """The key of test file `name` whose run executed or read `depends` (executed.py): those files'
    contents, those directories' entries, and `common`."""
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
    sums over a split file's groups (`task`), or its tests' sum in
    tests/test_durations.json where that is more: each test's record is its
    least time (`durations`), so a sum past the median may mean the measured
    runs predate the file's current tests, and the sum is a floor on what it
    takes. base/kernels' route census, 1,576 tests recorded at 2,654 s in a
    file main ran in 6 s, was packed by that median with six other files into
    one task, which ran past its deadline (train 18). Else the recorded sum, scaled by
    how much slower armada ran the files measured both ways; else None, for a
    file nothing has timed."""
    recorded = {name: sum(tests) for name, tests in recorded_tests().items()}
    measured = {name: timings.get("files", {})[row(python, name, "")] for name in files
                if row(python, name, "") in timings.get("files", {})}
    ratios = sorted(measured[name] / recorded[name] for name in files
                    if name in measured and recorded.get(name, 0) > 1)
    slower = ratios[len(ratios) // 2] if ratios else 1.0
    return {name: max(measured[name], recorded.get(name, 0.0)) if name in measured
            else recorded[name] * slower if name in recorded else None for name in files}


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


def split(tests: list[float], seconds: float, target: float, cut: int = 1) -> list[float]:
    """The weights of the groups a file that took `seconds` as `cut` groups runs as. Of those
    seconds, what its recorded `tests`' times leave is what each group paid besides its tests (its
    module's fixtures, imports and compiles, which a test's least recorded time does not hold), so
    each group weighs its tests plus that setup. The file is cut (`groups`) into the fewest runs
    whose tests fit what the setup leaves of `target`, at least half of it, so a file whose setup
    is most of a group is not cut into slivers that each pay it again; a heavier test runs alone.
    Counting the setup once a group, rather than scaling every test by the file's total, keeps a
    file from being cut finer each run for the setup its last cut added. A file keeps the cut it
    ran as while that cut's heaviest group, setup included, stays within `KEPT` of `target`:
    a group's compiled programs are cached under its cut (tests/remote_cache.py), so a new cut
    compiles every group from nothing, and the setup that measures moves the next cut again
    (armada dew-20261020144644-dceff433: a new cut fetched no programs, and 14 of its 20 split
    files would have been cut again). A file with no recorded time runs as equal counts of its
    tests."""
    if sum(tests) <= 0:
        count = math.ceil(seconds / target)
        return [seconds / count] * count
    setup = max(seconds - sum(tests), 0.0) / cut
    if cut > 1 and max(sum(tests[run.start:run.stop]) for run in groups(tests, cut)) + setup <= KEPT * target:
        count = cut
    else:
        room = max(target - setup, target / 2)
        count = len(starts(milliseconds(tests), round(room * 1000)))
    return [sum(tests[run.start:run.stop]) + setup for run in groups(tests, count) if run]


KEPT = 1.5
"""How far past the target a split file's heaviest group may run before the file is cut again."""


def measured_cut(timings: dict, python: str, name: str) -> tuple[float, int] | None:
    """What `name` took under `python` at the cut its rows in `timings` name, and that cut's count:
    its groups' rows summed where every group of a count has its row (the most groups if several
    counts do), else its whole row, else None. A row's median is measured at the cut its name
    holds, unlike the file's median, which mixes the cuts of the runs it remembers."""
    whole, counts = None, {}
    for row_name, seconds in timings.get("rows", {}).items():
        python_, name_, split_ = parsed_row(row_name)
        if (python_, name_) != (python, name) or split_ == COLLECT:
            continue
        if not split_:
            whole = seconds
            continue
        index, count = (int(part) for part in split_.split("/"))
        counts.setdefault(count, {})[index] = seconds
    complete = [count for count, held in counts.items() if len(held) == count]
    if complete:
        return sum(counts[max(complete)].values()), max(complete)
    return None if whole is None else (whole, 1)


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
        # Every file on every Python daily; on a pushed run the newest takes `NEWEST_FILES` whole.
        whole = python == PYTHONS[0] or cache == "off"
        files = every if whole else [name for name in every if name in NEWEST_FILES]
        timed = weights(files, timings, python)
        tasks: list[tuple[list[str], str, float]] = [
            ([name], "", target) for name, seconds in timed.items() if seconds is None]
        for name, seconds in timed.items():
            if seconds is not None and seconds > target:
                at, count = measured_cut(timings, python, name) or (seconds, 1)
                cut = split(recorded.get(name, []), at, target, count)
                # A file one group holds runs whole, alone, under its own row.
                tasks += [([name], f"{group}/{len(cut)}" if len(cut) > 1 else "", weight)
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
    rest = [] if cache == "off" else [name for name in every if name not in NEWEST_FILES]
    for index in range(COLLECT_TASKS if rest else 0):
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


def red_as(kind: str, nodes: list[str]) -> dict:
    """What turned a red row red, for grading it again (`land.sh`, `suspect.sh`): `kind` "nodes", its
    `failed` tests alone, by node id; "collect", its file's collection; "deadline", its task's; or
    "whole", its file otherwise (a test that did not collect, a missing import, pytest's own exit)."""
    return {"kind": kind, "failed": nodes}


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
             "output": errors.get(name, ""), **(red_as(COLLECT, []) if name in errors else {})}
            for name in tests]
    out.write_text(json.dumps({"rows": rows}))
    return 0


CLOSED_HUB = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_ENDPOINT": "http://127.0.0.1:9"}
"""What a task's tests see of the Hugging Face Hub, which none of them may reach (`-m "not network"`):
the libraries offline, and the Hub's address a closed port for any that would ask it regardless. A
test that needs the Hub is marked network; one that reads a model or dataset otherwise brings it."""


def apportioned(own_seconds: Mapping[str, float], wall: float, rows: int) -> dict[str, float]:
    """Each file's share of a task's `wall` time, in proportion to its tests' own times (`task`), or an
    even split over the task's `rows` (a file named twice is two) where no test recorded a time."""
    total = sum(own_seconds.values())
    pace = wall / total if total > 0 else 0.0
    return {name: own * pace if pace else wall / rows for name, own in own_seconds.items()}


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: bool = True,
         grace: float = 60.0, coverage: Path | None = None) -> int:
    """Run `tests` and write one verdict row a file, interrupting them past `deadline` seconds, killing
    their process group `grace` seconds after that, and giving it `SETTLE` seconds to be gone.

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
    done = child_groups.start(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, **environ, **CLOSED_HUB, "PYTHONPATH": path,
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
        expired = True
    # Past the deadline ^C first, so pytest reports what ran; a test stuck in C ignores it. Then the group is
    # killed, what pytest's tests left running with it, and given `SETTLE` to be gone.
    if left := child_groups.stop(done, interrupt=grace if expired else None, settle=SETTLE):
        print(f"processes {left} of the task's group still ran {SETTLE:.0f} s after SIGKILL", flush=True)
    reader.join(10)
    output = "".join(lines)
    cases = list(ET.parse(report).getroot().iter("testcase")) if report.is_file() else []
    ran = []
    own_seconds = {name: sum(float(case.get("time") or 0) for case in cases if file_of(case) == name)
                   for name in tests}
    shares = apportioned(own_seconds, time.monotonic() - began, len(tests))
    for name in tests:
        own = [case for case in cases if file_of(case) == name]
        failed = [(case, kind, node) for case in own for kind in ("failure", "error")
                  if (node := case.find(kind)) is not None]
        red = [f"{kind.upper()} {case.get('classname')}::{case.get('name')}: {node.get('message')}"
               for case, kind, node in failed]
        skipped = [f"SKIPPED on a missing import {case.get('classname')}::{case.get('name')}: "
                   f"{node.get('message')}" for case in own for node in case.iter("skipped")
                   if "could not import" in f"{node.get('message')} {node.text}"]
        # pytest exits 1 for failed tests and 5 for none collected; any other exit (a crash, a signal,
        # an internal error) leaves even a written report untrustworthy, so every row it holds is red.
        exited = ([f"pytest exited {done.returncode}" + ("" if own else f" with no report of {name}")]
                  if done.returncode not in (0, 1, 5) or (not own and done.returncode == 1) else [])
        red += skipped + exited
        if expired:
            red.append(f"{INTERRUPTED} of {deadline:.0f} s and was interrupted")
        # A failed case with no class path is a file that did not collect, no test of it.
        nodes = sorted({node_of(case) for case, _, _ in failed if case.get("classname")})
        whole = skipped or exited or not nodes or any(not case.get("classname") for case, _, _ in failed)
        cause = ({} if not red else red_as("deadline", []) if expired else red_as("whole", []) if whole
                 else red_as("nodes", nodes))
        seconds = shares[name]
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart.
        ran.append({"name": row(python, name, split), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([shown(output)] if red else [])),
                     "tests": {node_of(case): float(case.get("time") or 0) for case in own},
                     "timings": {row(python, name, ""): seconds}, **cause})
    out.write_text(json.dumps({"rows": rows + ran}))
    if coverage is None:
        # A traced run's rows, timed under tracing, are no rows to reuse.
        record(python, ran, executed_report, common)
    return 0


def record(python: str, rows: list[dict], executed_report: Path, common: str) -> None:
    """Keep each row a task ran with the files its tests executed, where they read the backend and
    its peak memory on a GPU and its children's pools: `gpu_tier` reads the reads, `gpu_needs` the
    memory. A green row keeps
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
        ran = executed.get(name, {"depends": None, "reads": [], "peak": None, "shares": 0, "pools": 0})
        if row_["exitCode"] == 0 and name not in executed:
            continue
        green = row_["exitCode"] == 0 and ran["depends"] is not None
        body = json.dumps({"key": cache_key(common, name, ran["depends"]) if green else None,
                           "depends": ran["depends"], "row": row_, "reads": ran["reads"],
                           "peak": ran["peak"], "shares": ran["shares"], "pools": ran["pools"]}).encode()
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
    done = subprocess.run([f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q",
                           "-m", " or ".join(MARKS), *every],
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


def gpu_needs(recorded: dict | None, budget: float) -> tuple[float, float]:
    """The GPU share a test file's process is given, as `XLA_PYTHON_CLIENT_MEM_FRACTION` of it, and what
    the file holds with its children, by its last run's `recorded` peak (tools/armada/executed.py's
    `gpu_peak`, or a child's that took its fraction, the larger) and `GPU_HEADROOM`: a child that took
    the share's fraction holds one more share, and one that named its own fraction the pool it held.
    A file no run measured, one that started a process it did not follow, and one whose last run left
    no report take all of `budget`."""
    if recorded is None or recorded["depends"] is None or recorded["peak"] is None:
        return budget, budget
    factor, slack = GPU_HEADROOM
    share = min(recorded["peak"] * factor + slack, budget)
    return share, min(share * (1 + recorded.get("shares", 0)) + recorded.get("pools", 0), budget)


def gpu_lane(python: str, tests: list[str], out: Path, *, cache: bool, processes: int,
             deadline: float) -> int:
    """Run `tests` on this process's share of the one GPU it sees, each file once, as a task of its
    own (`tested`), and write their verdict rows to `out`.

    A file's process is given the share its last run on this GPU's class measured, green or red,
    and files start heaviest first while what they hold with their children (`gpu_needs`) fits
    the share this process was given (`XLA_PYTHON_CLIENT_MEM_FRACTION`, as gpu-run sets it), at
    most `processes` at once. A file that runs out of a measured share is red: it took more than
    its last run, whose peak its next run is given. A file given the lane's whole share (unmeasured,
    measured that large, or with a child the lane did not follow, as the fla reference's in another
    environment's Python) grows its pool as it uses it rather than taking the share at its first GPU
    op (`XLA_PYTHON_CLIENT_PREALLOCATE`), so such a child has what the file has not yet taken; a file
    that grows to the whole share still leaves its child none.

    First the pinned fixtures the tests read offline are fetched while the Hub is reachable
    (tools/fetched_fixtures.py): a store file already whole is kept, and a pin whose upload is due stops
    the lane before any file runs."""
    subprocess.run([f".venv-{python}/bin/python", "tools/fetched_fixtures.py", "fetch"], check=True)
    token = os.environ["DEW_XLA_CACHE_TOKEN"]
    [total] = (int(megabytes) * 2 ** 20 for megabytes in gpu_query("memory.total"))
    budget = float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) * total
    held = device_class()

    def needs(name: str) -> tuple[float, float]:
        record = cache_request(cache_url(python, row(python, name, ""), held), token)
        return gpu_needs(json.loads(record) if record else None, budget)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        needed = dict(zip(tests, pool.map(needs, tests), strict=True))
    shares = {name: share for name, (share, _) in needed.items()}
    holds = {name: held for name, (_, held) in needed.items()}
    # Before any test writes under tests/, the inputs every file's key hashes.
    common = shared(python)
    waiting = sorted(tests, key=lambda name: -holds[name])
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(tests)}
    running: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        while waiting or running:
            free = budget - sum(holds[name] for name in running.values())
            fits = next((name for name in waiting if holds[name] <= free), None)
            if fits is not None and len(running) < processes:
                waiting.remove(fits)
                environ = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(shares[fits] / total),
                           **({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} if shares[fits] >= budget else {})}
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
INTERRUPTED = "the task ran past its deadline"
"""How a row says its task was interrupted: tests after the point it reached did not run."""
COVERAGE_DEADLINE = 5400.0
"""A coverage task's deadline. Branch tracing slows a task several times over, and a task cut short leaves
its files' coverage partial, which `coverage_report` refuses."""


def coverage_items(target: float, timings: dict) -> list[str]:
    """The plan's tasks for the oldest Python, as armada map items: each runs its files under coverage.py
    with the test cache off, its rows and data among its artifacts, by `COVERAGE_DEADLINE`."""
    return [f"mkdir -p {ARTIFACTS} && python3 tools/armada/ci.py task --python={entry['python']} "
            f"--tests='{entry['tests']}' --split='{entry['split']}' --out={ARTIFACTS}/rows.json --cache=off "
            f"--deadline={COVERAGE_DEADLINE:.0f} --coverage={ARTIFACTS}"
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
    # A task that wrote no rows or was interrupted left the coverage of its files' later tests out.
    def ran_short(task: Path) -> bool:
        verdict = task / "rows.json"
        return not verdict.exists() or any(INTERRUPTED in found["output"]
                                           for found in json.loads(verdict.read_text())["rows"])

    # armada keeps each task's artifacts in a directory named by its index.
    short = [task.name for task in sorted(directory.iterdir())
             if task.is_dir() and task.name.isdecimal() and ran_short(task)]
    if short:
        print(f"{len(short)} tasks ran short, so the pass is partial; run them again with a longer deadline: "
              f"{', '.join(short)}", file=sys.stderr)
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


GANG = 4
"""Ranks of each multi-host gang: four containers, one host each."""


def gangs(source: Path = Path("tests/test_multihost.py")) -> list[dict]:
    """The multi-host gangs CI runs side by side, as armada map items: one a model of `source`'s
    MODELS, for its layouts across hosts, and one for every other test, so each case runs in
    exactly one of them and a model added there is a gang of its own. Read from the source, as
    the CI runner has no Dew to import it."""
    models = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.Assign) and [target.id for target in node.targets
                                                       if isinstance(target, ast.Name)] == ["MODELS"])
    own = [f"layouts and {model}" for model in models]
    return [{"gang": GANG, "select": select} for select in own] + [
        {"gang": GANG, "select": " and ".join(f"not ({select})" for select in own)}]


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
    operations.add_parser("gangs")
    versions = operations.add_parser("pythons")
    versions.add_argument("--check", action="store_true",
                          help="refuse a .python-version other than the oldest, or a smoke of other versions")
    recording = operations.add_parser("durations")
    recording.add_argument("verdicts", type=Path, nargs="+",
                           help="green runs' verdicts, the newest first: armada verdict <sha> --json")
    args = parser.parse_args()
    if args.operation == "pythons":
        return pythons(check=args.check)
    if args.operation == "gangs":
        print(json.dumps(gangs()))
        return 0
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

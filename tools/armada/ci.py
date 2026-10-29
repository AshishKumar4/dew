#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT
    python3 tools/armada/ci.py gpu-tier [--marked] [--evidence] [--shards=N]
    python3 tools/armada/ci.py gpu-lane --tests="tests/a.py tests/b.py" --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py gpu-lane --network [--tests=...] --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py coverage items --target=SECONDS --timings=FILE > ITEMS
    python3 tools/armada/ci.py coverage report DIR
    python3 tools/armada/ci.py gangs

The plan prints a task matrix for each Python CI proves. Whole test files are
packed heaviest first into the lightest of the tasks, each near `target`
seconds, so a task imports only its own modules. A file heavier than that runs
as groups of its tests (`session.groups`, tools/armada/split.py): runs of consecutive
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
every row it holds is red, so a hang is graded, with what pytest reports as
it is interrupted, rather than leaving the task without a verdict. Each row
also carries its tests' own times, which `durations` folds into tests/test_durations.json from main's CI
(`armada verdict <sha> --json`): a file whose record no longer cuts it as it
runs takes what its tests took (`folded`). The land queue folds main's CI so
at each train, as a head of its own, and no hand edits the record.

The test cache reuses a file's row when its inputs are unchanged since it ran green. A
green row records the checkout's Python files its tests executed
(tools/armada/executed.py) and its key, a hash of those files, the
directories under src/ that hold them, however deep (so a module added
beside one, or an __init__.py on the way to one, changes it), the file
itself and what every file shares (`shared`: the environment's locks, the
CI tools, the names of the non-Python files under tests/ but its fixtures and
under src/, of the Python files under tests/ but the test modules, and src/'s
entries). A fixture is the input of the runs that opened, listed or probed it.
A test module, or a module in a package a file never imported, is no input
of that file's: a train that loses a head reruns only the files the head's
modules reached (the land queue's recompose after a pull). A task
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
recorded and reused on the GPU's class as a CPU task's are on the CPU's. With `--network`, a GPU node that
reaches the Hub runs every file's network-marked tests, which every other task sets aside, as one lane of
their own, recorded on a class of their own, so each train reruns only the files whose inputs changed.

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
import functools
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
import types
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
session = _beside("session")
r2 = _beside("r2")
watched = _beside("watched")
SETTLE = 30.0
"""Seconds a task's process group has, after SIGKILL, to be reaped and gone: inside armada's own timeout
with the default deadline and grace (session.DEADLINE + session.GRACE + 30 + the reader's 10 < .armada.json's
1800)."""

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
SHARED = ("pyproject.toml", "constraints.txt", "tools/armada/session.py", "tools/armada/child_groups.py",
          "tools/armada/split.py", "tools/armada/executed.py", "tools/armada/followed.py",
          "tools/armada/watched.py")
"""The files every test file's key covers whole, beside every lock (a GPU node builds its environment
from its CUDA lock), the environment's own inputs, which .armada.json names (`environment_files`),
and .armada.json's parts a test process sees (`armada_seen`): what every test process runs or reads,
the session it runs in, the process group it runs as and the plugins loaded into it among them. The
plan's own code (this file) runs outside the tests, so a change to it reruns none of them; nor does
the durations record (`session.DURATIONS`), which only a split file's groups are cut from, so a
group's key covers its own file's entries alone (`cache_key`); nor does a fixture or a package's data,
which is the input of the files whose runs opened it, however they opened it (executed.py's `Opens`)."""
ARMADA_SEEN = ("environment", "environments", "env")
""".armada.json's parts a test process runs in, beside its task's command and secrets (`armada_seen`):
its pool, size, target, plan and the task's deadline and mode decide where and when tasks run, never
what a test sees (owner, 2026-10-27: a pool change reran every file)."""


def armada_seen(text: str | None) -> str:
    """The parts of .armada.json (`text`, or None where there is none) a test process sees: its
    environments, their variables (`ARMADA_SEEN`), and its task's command and secrets, as one text."""
    if text is None:
        return "missing"
    config = json.loads(text)
    task = config.get("task", {})
    return json.dumps({**{part: config.get(part) for part in ARMADA_SEEN},
                       "command": task.get("command"), "secrets": task.get("secrets")}, sort_keys=True)


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
RECORD_FORMAT = 2
"""The version of a record's form, what `record` writes and `reused` reads: a change to its fields or
their meaning takes the next, so a branch grading with it never breaks main's records. A change to
what a record holds (executed.py, followed.py) needs none: they are `SHARED`, so every key moves with
them."""
CACHE_PREFIX = f"testcache/format-{RECORD_FORMAT}"
"""Where records live: one namespace per `RECORD_FORMAT`."""


def shared(python: str) -> str:
    """The hash of what every test file's run shares: `python`, the `SHARED` files, every lock,
    .armada.json's parts a test process sees (`armada_seen`), and the names of the Python files under
    tests/ but its test modules and of src/'s own entries, whose arrival can change what an import finds
    (a conftest.py, a helper or a top-level package that shadows another), and of every data file under
    tests/ but its fixtures and under src/ (an existence probe before the recorder starts). A module within
    a package under src/ is its importers' input alone, with its package's other modules (`inputs`), a test
    module is no other file's, and a fixture is its openers', listers' and probers' (executed.py's `Opens`,
    followed.py's `PROBES`): so a train that loses a head runs again only what that head's files reached
    (owner, 2026-10-27: 8 fixtures changed reran all 485 tasks of train 222's second round)."""
    digest = hashlib.sha256(python.encode())
    for path in sorted(Path("tests").rglob("*.py")):
        if "__pycache__" not in path.parts and not path.name.startswith("test_"):
            digest.update(str(path).encode())
    # The names of every data file under tests/ but its fixtures, and of every package's data: an existence
    # probe a library makes at import, before the recorder starts, is no record's. A fixture is its openers',
    # listers' and probers' alone, as a run records every probe it makes (followed.py's `PROBES`; Main,
    # 2026-10-29: train 277 reused 2 of 1967 rows once a head added fixtures). A build's output and a fetched
    # payload, which a planner's checkout lacks, are not named: they are their build's and their manifest's
    # (`built`, FETCHED, `contents`), whose names are.
    for root in ("tests", "src"):
        for path in sorted(Path(root).rglob("*")):
            if (path.is_file() and path.suffix != ".py" and not built(path) and FIXTURES not in path.parents
                    and not path.with_name(path.name + FETCHED).is_file()):
                digest.update(b"fixture:" + str(path).encode())
    armada = Path(".armada.json")
    digest.update(armada_seen(armada.read_text() if armada.is_file() else None).encode())
    for name in sorted({*SHARED, *map(str, Path("locks").glob("*.txt")), *environment_files(), "src"}):
        digest.update(name.encode() + contents(name))
    return digest.hexdigest()


def lane_inputs(python: str) -> str:
    """The hash of what a GPU lane's file shares (`gpu_lane`): what every test file's run shares (`shared`),
    and this file, which sets the environment, share and deadline each file runs under on a GPU, where a
    CPU task's are the session's."""
    return hashlib.sha256((shared(python) + contents(__file__).hex()).encode()).hexdigest()


FIXTURES = Path("tests/fixtures")
"""The tests' fixtures: each its openers', listers' and probers' input alone (`shared`), and a native reader's
directories up to it (`native_inputs`)."""
FETCHED = ".fetched.json"
"""A fetched fixture's manifest's suffix, beside the payload it pins (tools/fetched_fixtures.py)."""


def built(path: Path) -> bool:
    """Whether `path` is a build's output, never an input: compiled bytecode, or an editable install's
    metadata (setuptools' <name>.egg-info), which a task's environment writes into its checkout and a
    planner's lacks."""
    return "__pycache__" in path.parts or any(part.endswith(".egg-info") for part in path.parts)


def contents(name: str) -> bytes:
    """The hash of file `name`'s bytes, of directory `name`'s entries' names but its hidden ones and a
    build's (`built`), or b"missing". A fetched fixture's payload is the digest its manifest pins, whether
    it is downloaded here or not, so a planner that never fetches it keys it as a task does. An execution
    limit a run read (followed.py's `LIMITS`: "limit:cpu", "limit:memory") is the one binding this process
    (host.py's `limit`): a row green on 4 vCPUs is not reused on 8, nor on a planner, which reads its own (row
    1998, Main 2026-10-28). A path a run probed (followed.py's `PROBED`: "probe:" and the path) is its kind
    where it is a directory, whose probe never reads its entries, and as read otherwise."""
    if name.startswith("limit:"):
        return hashlib.sha256(host().limit(name).encode()).digest()
    if name.startswith("probe:"):
        probed = name.removeprefix("probe:")
        return b"directory" if Path(probed).is_dir() else contents(probed)
    path = Path(name)
    manifest = path.parent / (path.name + FETCHED)
    if path.name and manifest.is_file():
        return bytes.fromhex(json.loads(manifest.read_text())["sha256"])
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).digest()
    if path.is_dir():
        entries = sorted(entry.name for entry in path.iterdir()
                         if not entry.name.startswith(".") and not built(entry))
        return hashlib.sha256("\n".join(entries).encode()).digest()
    return b"missing"


def cache_key(common: str, name: str, split: str, depends: list[str]) -> str:
    """The key of test file `name`'s row in `split`'s group whose run executed or read `depends`
    (executed.py): its `inputs`' contents, its directories' entries, `common`, and for a group of a split
    file, the group and the tests split.py cuts into it from the durations record (`members`), the record's
    only part a run's outcome depends on: not their seconds, which each train's fold of main's durations
    moves (Main, 2026-10-28: every train re-keyed its split files and reran them cold, row 2000)."""
    digest = hashlib.sha256(common.encode())
    for path in inputs(name, depends):
        digest.update(path.encode() + contents(path))
    if split and split != COLLECT:
        digest.update(split.encode() + json.dumps(members(name, split)).encode())
    return digest.hexdigest()


def inputs(name: str, depends: list[str]) -> list[str]:
    """What test file `name`'s key covers, given that its run executed or read `depends`: those, and each
    directory under src/ or tools/ that holds a module it executed, however deep, whose entries name the
    modules an import there finds: a module added beside one it imported (which shadows it, or which an
    import it tried now finds), or an __init__.py added to a namespace package on the way to one, which
    executes as the import passes, changes its key; one added to a package it never imported does not.
    tools/ itself is among them, as a test puts it, or a directory under it, on sys.path and imports a tool
    by its name, which a package added there shadows; src/'s own entries every key covers (`shared`)."""
    packages = {str(parent) for path in depends
                if path.startswith(("src/", "tools/")) and path.endswith(".py")
                for parent in Path(path).parents
                if parent.parts[:1] in (("src",), ("tools",)) and parent != Path("src")}
    return sorted({name, *depends, *packages})


@functools.cache
def host() -> types.ModuleType:
    """host.py beside this file, loaded by its path, as this file puts nothing on sys.path."""
    spec = importlib.util.spec_from_file_location("dew_host", Path(__file__).with_name("host.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_class() -> str:
    """This host's CPU microarchitecture, the one class the records, the tests' compilation cache and
    the shared XLA bundles key it by (`host`)."""
    return host().cpu_class()


def device_class(network: bool = False) -> str:
    """The class a task's results belong to: on a CUDA lane (`JAX_PLATFORMS`, as
    tests/lane_environment.py reads it) the models of the GPUs it sees as the driver names them,
    before the CPU's class, whose float32 the lane's CPU devices compute (`cpu_class`); else the
    CPU's class alone. A network lane's (`gpu_lane`) is a class of its own, `network` after it: its
    records hold the network-marked tests alone, run with the Hub open, apart from the same file's
    other tests and their peaks."""
    if "cuda" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        held = cpu_class()
    else:
        held = "-".join([*(re.sub(r"[^A-Za-z0-9.]+", "-", model) for model in sorted(set(gpu_query("name")))),
                         cpu_class()])
    return f"{held}-network" if network else held


def gpu_query(field: str) -> list[str]:
    """`nvidia-smi`'s `field` of each GPU this process sees (`CUDA_VISIBLE_DEVICES`, when set)."""
    given = os.environ.get("CUDA_VISIBLE_DEVICES")
    return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                           *(["-i", given] if given else [])], capture_output=True, text=True,
                          check=True).stdout.split("\n")[:-1]


def store(token: str) -> r2.Store:
    """The Worker that holds the records."""
    return r2.Store(os.environ["DEW_XLA_CACHE_URL"], token, "dew-ci-test-cache", timeout=10)


def record_key(python: str, row_name: str, held: str, inputs: str | None = None) -> str:
    """Where the record of `row_name` on class `held` (`device_class`) lives: the last run's, or with
    `inputs`, the last of the runs on those shared inputs (`shared`), apart from the last runs' list
    (`listed`). Any run of another commit replaces the last record, so a commit's retry reads its own
    under its inputs first (`reused`). The Worker's key alphabet has no colon or hash."""
    _, name, split = parsed_row(row_name)
    at = f"{CACHE_PREFIX}/{python}" if inputs is None else f"{CACHE_PREFIX}-inputs/{inputs[:16]}/{python}"
    return f"{at}/{held}/{name}/" + ("-of-".join(split.split("/")) if split else "whole")


def key_parts(key: str) -> tuple[str, str, str]:
    """A record's key under its Python's prefix (`record_key`'s `<class>/<test file>/<group>`) as its class,
    its test file and its group (`whole`, or a split's `<index>-of-<count>`)."""
    held, rest = key.split("/", 1)
    name, group = rest.rsplit("/", 1)
    return held, name, group


def recorded(key: str, token: str) -> dict | None:
    """The record at `key`, read (`record_of`), or None where there is none or the Worker did not give it."""
    found = record_of(key, token)
    return None if found is None else json.loads(found)


def record_of(key: str, token: str) -> bytes | None:
    """The record at `key`, or None where there is none and where the Worker did not give it, which it
    reports and which costs only a rerun."""
    try:
        return store(token).get(key)
    except (r2.R2Error, ValueError) as error:
        print(f"the test cache did not give {key}: {error}", file=sys.stderr)
        return None


READ_TOKEN = "DEW_XLA_CACHE_READ_TOKEN"
"""The test cache's read-only token, which the plan reads records with (`reusable`): it never writes them."""


def pinned_class() -> str | None:
    """The one CPU class .armada.json pins its tasks to (task.cpu), as `cpu_class` names it, which their rows
    are recorded under; None where it pins none or several, or the tasks run on a CUDA lane, whose class
    (`device_class`) the plan cannot know."""
    config = Path(".armada.json")
    if not config.is_file() or "cuda" in os.environ.get("JAX_PLATFORMS", "").split(","):
        return None
    cpu = json.loads(config.read_text()).get("task", {}).get("cpu")
    return f"{cpu['vendor']}-{cpu['family']}-{cpu['model']}" if isinstance(cpu, dict) else None


def reusable(python: str, names: list[str], common: str) -> set[str]:
    """The files of `names` whose whole row ran green on exactly these inputs (`common`, and each file's
    own, `fresh`) on the class the tasks run on (`pinned_class`), read at plan time with the cache's read
    token (`READ_TOKEN`): a task of their own would only look each up and exit. None where the plan has no
    token or no single class (Main, 2026-10-28: a lane run with most files unchanged booted a container a
    task)."""
    token, held = os.environ.get(READ_TOKEN), pinned_class()
    if not token or held is None or "DEW_XLA_CACHE_URL" not in os.environ:
        return set()

    def hit(name: str) -> bool:
        found = recorded(record_key(python, row(python, name, ""), held, common), token)
        return found is not None and fresh(found, name, common)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        return {name for name, found in zip(names, pool.map(hit, names), strict=True) if found}


def reused(python: str, name: str, split: str, common: str, network: bool = False) -> dict | None:
    """The green row of test file `name` (in `split`'s group) recorded on this host's class
    (`device_class`), if it ran green there on exactly these inputs; else None. The record of the last
    run on these shared inputs decides: a key hashes them (`cache_key`), so no run on others, whose
    record the last run's may be, can be fresh here."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return None
    found = recorded(record_key(python, row(python, name, split), device_class(network), common), token)
    if found is None or not fresh(found, name, common):
        return None
    return found["row"]


def fresh(recorded: dict, name: str, common: str) -> bool:
    """Whether test file `name`'s record was made by a green run on exactly these inputs. A red run,
    or one of a file that started another Python process, whose code no event saw, leaves no key."""
    return recorded["key"] is not None and recorded["key"] == cache_key(
        common, name, parsed_row(recorded["row"]["name"])[2], recorded["depends"])


RECORD_TRIES = 4
"""How many times a record's write is tried, two, four and eight seconds apart."""


def recorded_tests() -> dict[str, list[float]]:
    """Each file's tests' times in tests/test_durations.json, in node-id order."""
    tests: dict[str, list[float]] = {}
    for node, seconds in sorted(durations_record().items()):
        tests.setdefault(node.split("::")[0], []).append(seconds)
    return tests


def durations_record() -> dict[str, float]:
    """The durations record (`session.DURATIONS`): each test's node id and its seconds."""
    return json.loads(session.DURATIONS.read_text()) if session.DURATIONS.is_file() else {}


def members(name: str, split: str) -> list[str]:
    """The node ids of test file `name`'s recorded tests that split.py cuts into group `split` ("G/N"):
    the file's entries in the durations record in node-id order, cut by their seconds (`session.groups`)."""
    recorded = {node: seconds for node, seconds in durations_record().items() if node.split("::")[0] == name}
    ordered = sorted(recorded)
    index, count = (int(part) for part in split.split("/"))
    run = session.groups([recorded[node] for node in ordered], count)[index - 1]
    return ordered[run.start:run.stop]


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


def split(tests: list[float], seconds: float, target: float, cut: int = 1) -> list[float]:
    """The weights of the groups a file that took `seconds` as `cut` groups runs as. Of those
    seconds, what its recorded `tests`' times leave is what each group paid besides its tests (its
    module's fixtures, imports and compiles, which a test's least recorded time does not hold), so
    each group weighs its tests plus that setup. The file is cut (`session.groups`) into the fewest runs
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
    heaviest = max(sum(tests[run.start:run.stop]) for run in session.groups(tests, cut)) if cut > 1 else None
    if heaviest is not None and heaviest + setup <= KEPT * target:
        count = cut
    else:
        room = max(target - setup, target / 2)
        count = len(session.starts(session.milliseconds(tests), round(room * 1000)))
    return [sum(tests[run.start:run.stop]) + setup for run in session.groups(tests, count) if run]


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


PLACED = ("src/", "tests/", "tools/", "examples/", "recipes/", "docs/", "tutorials/", "site/", ".github/")
"""Where a changed file is placed by the records (executed.py): a Python file a test ran, a page, a
notebook or a workflow it read (followed.py's `HIDDEN_INPUTS`), a fixture its openers read. A change
elsewhere is placed by none, and runs the whole suite (`policy`): a workflow change reran every file while
no record placed it (Main, 2026-10-28: 5 of 10 trains ran with the cache off, 0.75% of rows reused)."""
PLACED_FILES = ("README.md", "CONTRIBUTING.md", "AGENTS.md", "LICENSE", "CHANGELOG.md")
"""The checkout's own files at its root that tests read as a page is read, and the changelog, which
no test reads (test_changelog_merge writes its own) and most heads add an entry to: placed by none,
it turned the cache off for every train holding such a head, which then ran its whole suite on every
round (train 149: 30 task-hours, twice)."""


def policy(base: str) -> str:
    """The test cache's policy (DEW_TEST_CACHE) for this checkout's change from `base`: 'off', every file
    run, for a change to what every test process runs or reads (`SHARED`, a lock, the environment's
    files) or to a file the records do not place (`PLACED`); else 'on', each file run where its inputs
    changed. A tool is such an input of each file whose run executed or read it, which its key hashes, so
    a changed tool runs those files again and no other (train 115 reran its whole suite for five tools
    no pulled head had touched, under a policy that reran every file that read any tool)."""
    changed = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], check=True,
                             capture_output=True, text=True).stdout.split()
    if ".armada.json" in changed:
        before = subprocess.run(["git", "show", f"{base}:.armada.json"], capture_output=True, text=True,
                                check=False)
        armada = Path(".armada.json")
        if armada_seen(before.stdout if before.returncode == 0 else None) == armada_seen(
                armada.read_text() if armada.is_file() else None):
            changed.remove(".armada.json")  # where and when tasks run, never what a test sees
    whole = {*SHARED, *map(str, Path("locks").glob("*.txt")), *environment_files(), ".armada.json",
             ".python-version"}
    if any(name in whole or not (name.startswith(PLACED) or name in PLACED_FILES) for name in changed):
        return "off"
    return "on"


def borrowed(timings: dict, python: str) -> dict:
    """`timings` with each file's and row's time `python` has none of taken from another Python's
    (the most of theirs): 3.14 runs every file only on the daily run, and a file weighed by its
    tests' least recorded times instead is dispatched late by armada's longest-first order (train
    143: 3.14 tasks of 500-800 s started 550-760 s into the run)."""
    def filled(times: dict) -> dict:
        others: dict[str, float] = {}
        for name, seconds in times.items():
            other, rest = name.split(":", 1)
            if other != python and f"{python}:{rest}" not in times:
                others[f"{python}:{rest}"] = max(seconds, others.get(f"{python}:{rest}", 0.0))
        return {**times, **others}
    return {**timings, **{key: filled(timings.get(key, {})) for key in ("files", "rows")}}


COLD_SECONDS = 360.0
"""What a task adds to its estimate when it runs cold, the test cache's policy off (a lock, JAX's build or
the recorder changed): the compiles and imports no warm run paid. Train 254's cold suite ran about 470 s a
task where its plan estimated about 120 (1,389 tasks in 2,700 s on 242 containers, 2026-10-28)."""


def plan(target: float, timings: dict) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has
    timed runs alone, so however long it takes it holds up no other file, and its time is known from
    then on. Files marked `torchax` (`APART`) share tasks only with one another. Each entry carries
    the test cache's policy, DEW_TEST_CACHE here (`policy`), to its task. Under the policy "on", the files
    whose rows the cache holds for exactly these inputs (`reusable`) share one task, which looks each up
    again in its container, so the verdict names every file and containers scale with the change. Under the
    policy "off" every task's weight carries COLD_SECONDS, so armada sizes the pool by the work, not the
    longest task, as warm estimates did."""
    cache = os.environ.get("DEW_TEST_CACHE") or "on"
    # The whole suite's run alone (DEW_WHOLE_SUITE, every fifth publish) runs whole_suite cells (`selected`).
    cells = "1" if os.environ.get("DEW_WHOLE_SUITE") else "0"
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    recorded = recorded_tests()
    entries = []
    for python in PYTHONS:
        # Every file on every Python daily; on a pushed run the newest takes `NEWEST_FILES` whole.
        whole = python == PYTHONS[0] or cache == "off"
        files = every if whole else [name for name in every if name in NEWEST_FILES]
        timings = borrowed(timings, python)
        timed = weights(files, timings, python)
        kept = sorted(reusable(python, [name for name, seconds in timed.items() if seconds is None
                                        or seconds <= target], shared(python))) if cache == "on" else []
        timed = {name: seconds for name, seconds in timed.items() if name not in kept}
        # A lookup a file, in seconds: the task's weight is its container's start and those lookups.
        tasks: list[tuple[list[str], str, float]] = [(kept, "", float(len(kept)))] if kept else []
        tasks += [([name], "", target) for name, seconds in timed.items() if seconds is None]
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
                            "weight": weight + (COLD_SECONDS if cache == "off" else 0.0), "cache": cache,
                            "whole": cells})
    rest = [] if cache == "off" else [name for name in every if name not in NEWEST_FILES]
    for index in range(COLLECT_TASKS if rest else 0):
        names = rest[index::COLLECT_TASKS]
        entries.append({"name": f"{PYTHONS[-1]}-{COLLECT}-{index + 1}", "python": PYTHONS[-1],
                        "tests": " ".join(names), "split": COLLECT,
                        "rows": [row(PYTHONS[-1], name, COLLECT) for name in names], "weight": target,
                        "cache": cache, "whole": cells})
    return entries


MARKED = ("whole_suite", "release_env")
"""The marks a file's row names its tests by, as pytest resolved them (executed.py): its whole_suite
cells, which `slow` reports as the whole suite's alone, and its release checks, which `release_plan`
plans."""
RELEASE = "release:"
"""A release check's row, beside its file's own: `row`'s group, its requirement after it."""


def release_plan(verdict: dict) -> list[dict]:
    """The release checks of a whole suite's `verdict` (`armada verdict <sha> --json`), a task each
    requirement: the files whose rows name checks under it (`release_env`, as pytest resolved each test's
    mark before any selection: executed.py), which the run's own tasks set aside, as the suite's environment
    lacks it (`selected`). Its run's second job runs them (.github/workflows/ci.yml, `release`)."""
    checks: dict[str, set[str]] = {}
    for found in verdict["rows"]:
        for requirement in found.get("release_env", {}):
            checks.setdefault(requirement, set()).add(parsed_row(found["name"])[1])
    return [{"release": requirement, "python": PYTHONS[0], "tests": " ".join(sorted(names))}
            for requirement, names in sorted(checks.items())]


def release(python: str, tests: list[str], requirement: str, out: Path) -> int:
    """`requirement`'s release checks in `tests` (`release_plan`), as a whole suite's second job runs each:
    its rows in `out`, each red one named, exit 1 on any."""
    task(python, tests, "", out, session.DEADLINE, "off", release=requirement)
    red = [found["name"] for found in json.loads(out.read_text())["rows"] if found["exitCode"]]
    for name in red:
        print(f"RED {name}")
    return 1 if red else 0


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


def targeted(python: str, arguments: list[str], report: Path) -> int:
    """pytest's exit status on `arguments` (files, node ids and options) run as a CI task runs its files
    (`session.pytest_run`): a lane's targeted run (tmp/run_tests.py) and the gangs select as CI does, and an
    option of their own (a `-m` after CI's) chooses instead."""
    return subprocess.run([*session.pytest_run(python, report), *arguments],
                          env=session.pytest_environment({}),
                          check=False).returncode


def apportioned(own_seconds: Mapping[str, float], wall: float, rows: int) -> dict[str, float]:
    """Each file's share of a task's `wall` time, in proportion to its tests' own times (`task`), or an
    even split over the task's `rows` (a file named twice is two) where no test recorded a time."""
    total = sum(own_seconds.values())
    pace = wall / total if total > 0 else 0.0
    return {name: own * pace if pace else wall / rows for name, own in own_seconds.items()}


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str = "on",
         grace: float = session.GRACE, coverage: Path | None = None, whole: bool = False,
         release: str = "") -> int:
    """Run `tests` and write one verdict row a file, interrupting them past `deadline` seconds, killing
    their process group `grace` seconds after that, and giving it `SETTLE` seconds to be gone.

    A row's seconds are its share of the task's wall time, in proportion to its tests' own times:
    the plan weighs a file by what it costs a container (its import, its collection, the
    container's pace), which pytest's case times leave out. Weighed by case times, tasks planned
    at 300 seconds ran a median of 392. With `coverage`, pytest runs under coverage.py, whose data
    file goes there (`coverage_items`)."""
    # Before any test writes under tests/, the inputs as the plan saw them.
    return tested(python, tests, split, out, deadline, cache, grace, shared(python), {}, coverage,
                  whole=whole, release=release)


def pytest_command(run: list[str], tests: list[str], split: str, release: str) -> list[str]:
    """CI's pytest `run` over `tests`, with the plugins that record what ran (executed.py), keep one group
    of a split file (`split`, split.py) and a release check's tests (`release`, release.py). A release
    check's pytest runs with its requirement over the suite's environment, whose torch and jax it keeps:
    uv's overlay."""
    grouping = (["-p", "split", f"--ci-group={split}"] if split else []) + (
        ["-p", "release", f"--ci-release={release}"] if release else [])
    command = [*run, "-p", "executed", *tests, *grouping]
    return ["uv", "run", "--no-project", "--python", command[0], "--with", release, "--", "python",
            *command[1:]] if release else command


def tested(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str, grace: float,
           common: str, environ: Mapping[str, str], coverage: Path | None = None,
           network: bool = False, whole: bool = False, release: str = "") -> int:
    """`task` on the inputs `common` hashes, its pytest under `environ` beside this process's, a green row
    reused as the cache's policy (`policy`) allows. With `network`, the network-marked tests alone, with the
    Hub open, recorded on the network lane's class (`device_class`); `release`, a release check's."""
    began = time.monotonic()
    group = split or (f"{RELEASE}{release}" if release else "")
    rows = [found for name in tests
            if cache != "off" and (found := reused(python, name, group, common, network))]
    tests = [name for name in tests if row(python, name, group) not in {found["name"] for found in rows}]
    if rows:
        print(f"{len(rows)} files ran green on these inputs on {device_class(network)}; "
              "their rows are reused")
    if not tests:
        out.write_text(json.dumps({"rows": rows}))
        return 0
    executed_report = out.with_name(f"{out.name}.executed.json")
    collection = out.with_name(f"{out.name}.collected.json")
    report = out.with_name(f"{out.name}.junit.xml")
    for stale in (report, executed_report, collection):
        stale.unlink(missing_ok=True)
    # coverage.py follows the Python processes the tests start (pyproject.toml's patch), each writing
    # its data where COVERAGE_FILE says.
    measured = [] if coverage is None else ["-m", "coverage", "run"]
    environ = environ if coverage is None else {**environ, "COVERAGE_FILE": str(coverage / ".coverage")}
    run = selected(session.pytest_run(python, report, *measured), network, whole, bool(release))
    command = pytest_command(run, tests, split, release)
    # pytest leads a process group of its own, so what its tests start ends with it. Its output is
    # streamed as it comes by a reader of its own, which nothing waits on past pytest's end: a test's
    # child holding the pipe open (a pool worker, a server) cannot keep the verdict from being written.
    lines: list[str] = []
    environment = session.pytest_environment({**environ, "DEW_EXECUTED": str(executed_report),
                                              "DEW_COLLECTED": str(collection)})
    if network:
        environment = {key: value for key, value in environment.items() if key not in session.CLOSED_HUB}
    # Every data file the task's processes open, held from before pytest starts until its group stopped
    # (`watched`): each is the input of every file the task runs.
    opens = watched.Opens(os.getcwd())
    done = child_groups.start(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env=environment)

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
    # A process of the group still running may yet open a file: no record of this task keeps a key.
    closed = opens.closed()
    opened = None if left else closed
    reader.join(10)
    output = "".join(lines)
    cases = list(ET.parse(report).getroot().iter("testcase")) if report.is_file() else []
    # How collection found each file: a file with no row is green only where every test it holds was set aside
    # by the selection, as pytest reported it (executed.py); a run that wrote no report greens nothing.
    counts = json.loads(collection.read_text()) if collection.is_file() else {}
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
        # an internal error) leaves even a written report untrustworthy, so every row it holds is red. A file
        # whose every test the selection set aside has no report however pytest exits, another file's failure
        # among the task's included.
        counted = counts.get(name, {})
        deselected = bool(counted) and counted["collected"] == counted["deselected"]
        exited = ([f"pytest exited {done.returncode}" + ("" if own else f" with no report of {name}")]
                  if done.returncode not in (0, 1, 5) or (not own and not deselected and done.returncode == 1)
                  else [])
        unaccounted = ([] if exited or (report.is_file() and (own or deselected))
                       else [f"no evidence {name} ran: " + ("pytest wrote no report" if not report.is_file()
                                                            else "pytest reported none of its tests, and its "
                                                                 "collection shows them not all deselected")])
        red += skipped + exited + unaccounted
        if expired:
            red.append(f"{INTERRUPTED} of {deadline:.0f} s and was interrupted")
        # A failed case with no class path is a file that did not collect, no test of it.
        nodes = sorted({node_of(case) for case, _, _ in failed if case.get("classname")})
        whole = skipped or exited or unaccounted or not nodes or any(not case.get("classname")
                                                                      for case, _, _ in failed)
        cause = ({} if not red else red_as("deadline", []) if expired else red_as("whole", []) if whole
                 else red_as("nodes", nodes))
        seconds = shares[name]
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart.
        ran.append({"name": row(python, name, group), "exitCode": 1 if red else 0, "seconds": seconds,
                     "output": "\n".join(red + ([output[-4000:]] if red else [])),
                     # Its first failed test's exception as pytest gave it ("RuntimeError: ..."), which
                     # `exhausted` reads.
                     **({"first_failure": failed[0][2].get("message") or ""} if failed else {}),
                     "tests": {node_of(case): float(case.get("time") or 0) for case in own},
                     # Its tests pytest marks whole_suite (executed.py), which `slow` never flags.
                     **{key: cells for key, cells in counts.get(name, {}).items() if key in MARKED},
                     "timings": {row(python, name, ""): seconds}, **cause})
    out.write_text(json.dumps({"rows": rows + ran}))
    if coverage is None:
        # A traced run's rows, timed under tracing, are no rows to reuse.
        record(python, ran, executed_report, common, network, opened)
    return 0


def native_inputs(opened: set[str], recorded: set[str]) -> set[str]:
    """The fixture directories, each up to tests/fixtures, of what a task's processes opened natively
    (`watched`: an open no Python process recorded, as a tokenizer's Rust code makes): a native reader's
    probe beside what it opened is no record's, as followed.py's `PROBES` are Python's. The fetched store
    is none: its payloads are content-addressed, and a planner, which never downloads them, keys them as a
    task does (`contents`)."""
    store = FIXTURES / "fetched"
    return {str(parent) for name in opened - recorded if FIXTURES in (path := Path(name)).parents
            and store != path and store not in path.parents
            for parent in path.parents if parent == FIXTURES or FIXTURES in parent.parents}


def record(python: str, rows: list[dict], executed_report: Path, common: str, network: bool = False,
           opened: set[str] | None = None) -> None:
    """Keep each row a task ran with the files its tests executed and read, every data file the task's
    processes opened (`opened`, the task's watch: None where it was blind, and no row keeps a key), where
    they read the backend and its peak memory on a GPU and its children's pools: `gpu_tier` reads the
    reads, `gpu_needs` the memory. A green row keeps its key too, which `reused` reads it back by; a red
    row has none, so a later run never reuses an earlier green row of its file."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return
    # A run cut short, or whose collection failed, may have written no report, or none for a file.
    executed = json.loads(executed_report.read_text()) if executed_report.is_file() else {}
    held = device_class(network)
    if opened is not None:
        recorded = {read for ran in executed.values() for read in ran["depends"] or ()}
        opened = opened | native_inputs(opened, recorded)
    for row_ in rows:
        name = parsed_row(row_["name"])[1]
        ran = executed.get(name, {"depends": None, "reads": [], "peak": None, "shares": 0, "pools": 0})
        if ran["depends"] is not None:
            ran = {**ran, "depends": None if opened is None else sorted({*ran["depends"], *opened})}
        if row_["exitCode"] == 0 and name not in executed:
            continue
        green = row_["exitCode"] == 0 and ran["depends"] is not None
        key = cache_key(common, name, parsed_row(row_["name"])[2], ran["depends"]) if green else None
        body = json.dumps({"key": key,
                           "depends": ran["depends"], "row": row_, "reads": ran["reads"],
                           "peak": ran["peak"], "shares": ran["shares"], "pools": ran["pools"]}).encode()
        # The record under these inputs, which `reused` reads, and the last run's, which the GPU tier and
        # lane read, only once the first is written, so it is never newer.
        if written(record_key(python, row_["name"], held, common), token, body):
            written(record_key(python, row_["name"], held), token, body)


def written(key: str, token: str, body: bytes) -> bool:
    """Whether `body` was PUT at `key`, tried `RECORD_TRIES` times while the Worker is unavailable: a
    record not written leaves the one before it, such as an older green row a red run should have
    replaced."""
    for attempt in range(RECORD_TRIES):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            store(token).put(key, body)
            return True
        except r2.Unavailable as error:
            print(f"the test cache did not take {key}: {error}", file=sys.stderr)
        except (r2.R2Error, ValueError) as error:
            print(f"the test cache refused {key}: {error}", file=sys.stderr)
            return False
    return False


def listed(prefix: str, token: str) -> list[str]:
    """The names of the Worker's records under `prefix`, without it, a page of a thousand at a time; an
    `r2.R2Error` where the Worker did not list them, its caller's to answer."""
    return list(store(token).keys(prefix))


def gpu_tier(python: str) -> dict[str, list[str]]:
    """The test files that take another path on a GPU, each with its evidence: where its tests read
    the backend and its marks of `session.MARKS`, over every record of its rows,
    on every class, a split file's under each count of groups it was planned as. A file is out
    only where every record is of these inputs (`fresh`), a CPU task's (`shared`) or a GPU lane's
    (`lane_inputs`), and says neither, so a file whose inputs changed since it ran is in as "changed",
    and one without a record, or with one the Worker did not return, as "unrecorded": a new file, or one
    no green run of these tools has recorded."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        raise SystemExit("the GPU tier is chosen from the test cache's records, which need its URL and token")
    inputs = (shared(python), lane_inputs(python))
    prefix = f"{CACHE_PREFIX}/{python}/"

    def evidence_of(key: str) -> tuple[str, list[str]]:
        _, name, _ = key_parts(key)
        found = recorded(f"{prefix}{key}", token)
        if found is None:
            return name, ["unrecorded"]
        current = any(fresh(found, name, common) for common in inputs)
        return name, found["reads"] or ([] if current else ["changed"])

    try:
        keys = listed(prefix, token)
    # The tier is chosen from the records: none listed, none is chosen.
    except r2.R2Error as error:
        raise SystemExit(f"the test cache did not list {prefix}: {error}") from error
    evidence: dict[str, set[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for name, found in pool.map(evidence_of, keys):
            evidence.setdefault(name, set()).update(found)
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    unrecorded = {name: ["unrecorded"] for name in every if name not in evidence}
    return {name: sorted(evidence[name]) for name in every if evidence.get(name)} | unrecorded


def marked(python: str) -> dict[str, int]:
    """Each test file of this checkout with tests marked one of `session.MARKS`, and how many, as pytest
    collects it on the CPU: what a batch's GPU run holds, whatever the records say."""
    return collected(python, " or ".join(session.MARKS))


def network_files(python: str) -> dict[str, int]:
    """Each test file of this checkout with tests marked network, and how many: what the network lane runs
    (`gpu_lane`)."""
    return collected(python, "network")


def collected(python: str, marks: str) -> dict[str, int]:
    """Each test file of this checkout with tests pytest's `-m marks` selects, and how many, as pytest
    collects it on the CPU."""
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    done = subprocess.run([f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q", "-m", marks,
                           *every], capture_output=True, text=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    if done.returncode not in (0, 5):
        raise SystemExit(f"collecting the tests marked {marks} exited {done.returncode}\n"
                         f"{done.stdout[-4000:]}{done.stderr[-2000:]}")
    counts: dict[str, int] = {}
    for node in done.stdout.splitlines():
        if "::" in node:
            counts[node.split("::")[0]] = counts.get(node.split("::")[0], 0) + 1
    return counts


def selected(command: list[str], network: bool, whole: bool = False, release: bool = False) -> list[str]:
    """CI's pytest `command` (`session.pytest_run`), or with `network` the same selecting the network-marked
    tests alone, every one the rest of CI sets aside. It sets aside the tests marked whole_suite too, but on
    the whole suite's own run (`whole`: DEW_WHOLE_SUITE, every fifth publish's ci/full-whole-*), whatever
    the cache's policy: a policy-off train runs every file, never a whole-suite cell (Main, 2026-10-28:
    train 250's policy-off suite ran G03's GRPO cells, 826-1,430 s each, to its deadline). The release checks
    run in their own tasks alone (`release`), under a requirement the suite's environment lacks
    (`release_plan`)."""
    at = command.index(session.SELECTION[1])
    checks = " and release_env" if release else " and not release_env"
    cells = checks + ("" if whole else " and not whole_suite")
    return [*command[:at], "network" if network else session.SELECTION[1] + cells, *command[at + 1:]]


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


EXHAUSTED = re.compile(r"(?:[\w.]+\.)?(?:XlaRuntimeError|JaxRuntimeError|RuntimeError|OutOfMemoryError): "
                       r"(?:RESOURCE_EXHAUSTED: (?:Out of memory|Failed to allocate)\b"
                       r"|CUDA_ERROR_OUT_OF_MEMORY\b|CUDA (?:error: )?out of memory)")
"""An exception that is the GPU's device memory running out, as pytest gives it: XLA's runtime error whose
status is RESOURCE_EXHAUSTED for an allocation (out of memory, or a failed allocation), or the CUDA driver's
out-of-memory error, raised as such. A block past the GPU's own resources (XLA's RESOURCE_EXHAUSTED for a
shared memory size limit, as `dew.nn.kernels.tuning` reads it) is none, as no larger share helps it; nor is an
assertion, whatever its message says."""


def exhausted(failure: str | None) -> bool:
    """Whether a red row's first failed test raised the GPU's memory running out (`EXHAUSTED`, its exception's
    start), which a smaller share than the file needs causes, rather than failed on its own."""
    return failure is not None and EXHAUSTED.match(failure) is not None


def borrowed_records(names: list[str], python: str, held: str, token: str) -> dict[str, tuple[str, dict]]:
    """For each of `names`, test files no run on class `held` measured, the class and last record of
    the whole file on another class whose run measured its GPU peak, the one that holds the most
    (`gpu_needs`): what a file has taken on a GPU, which a first lane on a class takes as its share.
    One no class measured is absent, as every one is where the Worker does not list its records. Its run on
    `held` records its own, which every later lane reads."""
    if not names:
        return {}
    prefix = f"{CACHE_PREFIX}/{python}/"
    wanted = set(names)
    try:
        every = listed(prefix, token)
    # The Worker did not list its records: borrowing only packs a lane, so each such file runs alone with the
    # lane's whole share, as one measured on no class does.
    except r2.R2Error as error:
        print(f"the test cache did not list {prefix}: {error}; files unmeasured here run alone",
              file=sys.stderr)
        return {}
    keys = [key for key in every
            if (parts := key_parts(key))[0] != held and parts[2] == "whole" and parts[1] in wanted]

    def read(key: str) -> tuple[str, str, dict | None]:
        kind, name, _ = key_parts(key)
        return name, kind, recorded(f"{prefix}{key}", token)

    def holds(found: dict) -> float:
        return gpu_needs(found, float("inf"))[1]

    borrowed: dict[str, tuple[str, dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for name, kind, found in pool.map(read, keys):
            if found is None or found.get("depends") is None or found.get("peak") is None:
                continue
            if name not in borrowed or holds(found) > holds(borrowed[name][1]):
                borrowed[name] = (kind, found)
    return borrowed


def gpu_lane(python: str, tests: list[str], out: Path, *, cache: str, processes: int,
             deadline: float, network: bool = False) -> int:
    """Run `tests` on this process's share of the one GPU it sees, each file once, as a task of its
    own (`tested`), and write their verdict rows to `out`. With `network`, the network lane: each file's
    network-marked tests alone, which every other task deselects (`session.SELECTION`), run with the Hub
    open on a node that reaches it, their records on a class of their own (`device_class`), so the test
    cache reruns only those whose inputs changed (Main, 2026-10-25: no CI run reached the published
    checkpoints' parity tests).

    A file's process is given the share its last run on this GPU's class measured, green or red,
    and files start heaviest first while what they hold with their children (`gpu_needs`) fits
    the share this process was given (`XLA_PYTHON_CLIENT_MEM_FRACTION`, as gpu-run sets it), at
    most `processes` at once. A file that runs out of a measured share is red: it took more than
    its last run, whose peak its next run is given. A file no run on this class measured borrows its
    share from the largest any other GPU class's last run measured (`borrowed`), so a first lane on
    a class packs its files as a measured one does; one red under a borrowed share whose first failure is
    its memory running out (`exhausted`) runs once more, alone, with the lane's whole share, as that red is
    the borrowed share's; one whose first failure is any other is red, run once. A file
    given the lane's whole share (measured on no class, measured that large, or with a child the lane
    did not follow, as the fla reference's in another environment's Python) grows its pool as it uses
    it rather than taking the share at its first GPU op (`XLA_PYTHON_CLIENT_PREALLOCATE`), so such a
    child has what the file has not yet taken; a file that grows to the whole share still leaves its
    child none. Each file's process takes its share of the CPUs this process may use, as of the files
    that run at once (`OMP_NUM_THREADS`): a torch reference arm's small ops on all of a node's 30 CPUs
    ran 13 times slower than on 4 (test_gemma3n's gradient reference on an A100 node, 1175 s against
    armada's 326 s), and the more so as files share them.

    First the pinned fixtures the tests read offline are fetched while the Hub is reachable
    (tools/fetched_fixtures.py): a store file already whole is kept, and a pin whose upload is due stops
    the lane before any file runs."""
    subprocess.run([f".venv-{python}/bin/python", "tools/fetched_fixtures.py", "fetch"], check=True)
    token = os.environ["DEW_XLA_CACHE_TOKEN"]
    [total] = (int(megabytes) * 2 ** 20 for megabytes in gpu_query("memory.total"))
    budget = float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) * total
    held = device_class(network)
    threads = str(max(1, len(os.sched_getaffinity(0)) // processes))

    def needs(name: str) -> dict | None:
        return recorded(record_key(python, row(python, name, ""), held), token)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        records = dict(zip(tests, pool.map(needs, tests), strict=True))
    borrowed = borrowed_records([name for name, found in records.items() if found is None], python, held,
                                token)
    needed = {name: gpu_needs(records[name] or (borrowed[name][1] if name in borrowed else None), budget)
              for name in tests}
    shares = {name: share for name, (share, _) in needed.items()}
    holds = {name: held for name, (_, held) in needed.items()}
    # Before any test writes under tests/, the inputs every file's key hashes.
    common = lane_inputs(python)
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(tests)}

    def run(names: list[str]) -> None:
        waiting = sorted(names, key=lambda name: -holds[name])
        running: dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ThreadPoolExecutor(processes) as pool:
            while waiting or running:
                free = budget - sum(holds[name] for name in running.values())
                fits = next((name for name in waiting if holds[name] <= free), None)
                if fits is not None and len(running) < processes:
                    waiting.remove(fits)
                    # Its pool taken at its first op, or grown where it holds the whole share, as this lane
                    # says whatever its own environment does: a GPU node's session grows every job's pool.
                    environ = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(shares[fits] / total),
                               "XLA_PYTHON_CLIENT_PREALLOCATE": "false" if shares[fits] >= budget else "true",
                               "OMP_NUM_THREADS": threads}
                    running[pool.submit(tested, python, [fits], "", verdicts[fits], deadline, cache,
                                        session.GRACE, common, environ, None, network)] = fits
                    continue
                done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    future.result()
                    del running[future]

    run(tests)
    # Red under a share another class measured, its first failure an allocation's (`exhausted`): that red is
    # the borrowed share's, not the file's, and the file runs again alone with the lane's whole share, its row
    # saying so with the first failure, as packed work run again alone is interference. Any other first
    # failure (an assertion, an accuracy miss, an illegal address) is the file's: its red stands, run once.
    again: dict[str, str] = {}
    for name in borrowed:
        failures = [verdict.get("first_failure")
                    for verdict in json.loads(verdicts[name].read_text())["rows"] if verdict["exitCode"] != 0]
        # Below the whole budget only: a file that already had it would run again as it ran.
        if failures and exhausted(failures[0]) and shares[name] < budget:
            again[name] = (f"red under a share of {shares[name] / 2 ** 30:.1f} GiB borrowed from "
                           f"{borrowed[name][0]}'s measurement, out of memory ({failures[0]}); run again "
                           "alone with the lane's whole share")
    for name in again:
        shares[name] = holds[name] = budget
    run(list(again))
    for name, interference in again.items():
        rows = json.loads(verdicts[name].read_text())["rows"]
        verdicts[name].write_text(json.dumps({"rows": [{**verdict, "interference": interference}
                                                       for verdict in rows]}))
    rows = [verdict for name in tests for verdict in json.loads(verdicts[name].read_text())["rows"]]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def folded(record: dict[str, float], verdict: dict, target: float) -> dict[str, float]:
    """The durations `record` with main's CI `verdict` (`armada verdict <sha> --json`) folded in, one test
    file at a time, each whole, as its floor Python's rows ran it. A file whose rows are all green and
    complete (its whole row, or every group of its cut) takes the times they observed where its record is
    stale (`stale`); a file with a red or missing row keeps its entries, as does a file the verdict did not
    run, and a file the checkout no longer has loses them. A reused row (the test cache) carries the times
    of the run that recorded it. Each train folds main's CI so (the land queue), and nothing else writes the
    record."""
    rows: dict[str, list[dict]] = {}
    for row_ in verdict["rows"]:
        python, name, split_ = parsed_row(row_["name"])
        if python == PYTHONS[0] and split_ != COLLECT and Path(name).is_file():
            rows.setdefault(name, []).append(row_)
    kept = {node: seconds for node, seconds in record.items() if Path(node.split("::")[0]).is_file()}
    for name, held in rows.items():
        splits = {parsed_row(row_["name"])[2] for row_ in held}
        counts = {int(split_.split("/")[1]) for split_ in splits if split_}
        complete = splits == {""} or (len(counts) == 1 and splits == {f"{group}/{count}" for count in counts
                                                                      for group in range(1, count + 1)})
        if not complete or any(row_["exitCode"] != 0 or "tests" not in row_ for row_ in held):
            continue
        observed = {node: seconds for row_ in held for node, seconds in row_["tests"].items()}
        own = {node: seconds for node, seconds in kept.items() if node.split("::")[0] == name}
        if stale(own, observed, sum(row_["seconds"] for row_ in held), max(counts, default=1), target):
            kept = {node: seconds for node, seconds in kept.items() if node.split("::")[0] != name}
            kept.update({node: round(seconds, 3) for node, seconds in observed.items()})
    return dict(sorted(kept.items()))


def stale(own: dict[str, float], observed: dict[str, float], seconds: float, cut: int, target: float) -> bool:
    """Whether a file's record `own` no longer cuts it as it runs: its tests differ from those its rows
    `observed` (one added, removed or renamed); a file within `target` (it took `seconds`), or one with no
    tests to cut, is recorded past it, so the plan would split what runs whole; or, for a file past it (as
    `cut` groups), the groups the
    plan cuts from the record (`split`) run a group past `KEPT` of the target as its tests ran, setup
    included, and those it would cut from what they took run their heaviest shorter. A cut that fits is
    kept whatever its tests' times drift, as the plan keeps it, and so is one whose heaviest group no cut
    shortens (a test heavier than the budget, which runs alone either way): a fold rewrites a file only
    where its plan changes for the better."""
    if own.keys() != observed.keys():
        return True
    if seconds <= target or not own:
        return sum(own.values()) > target
    nodes = sorted(own)
    recorded, ran = [own[node] for node in nodes], [observed[node] for node in nodes]
    setup = max(seconds - sum(ran), 0.0) / cut

    def heaviest(weights: list[float]) -> float:
        count = len(split(weights, seconds, target, cut))
        return max(sum(ran[run.start:run.stop]) + setup for run in session.groups(weights, count) if run)

    return heaviest(recorded) > KEPT * target and heaviest(ran) < heaviest(recorded)


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
        return not verdict.exists() or any(found["exitCode"] and found["kind"] == "deadline"
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


def slow(verdict: dict, target: float, base: dict[str, float] | None = None) -> list[dict]:
    """Each test of `verdict`'s rows (`armada verdict <sha> --json`) whose own seconds pass `target`, the
    group target, slowest first: no cut brings its group within the target. A whole_suite cell is one too,
    reported as running on the whole suite alone (`whole_suite`, as pytest marked it in its row:
    executed.py), which train CI never runs (bible §7: every test past the target is reported). With `base`
    (main's durations record), only the tests it lacks: a new slow test (owner, 2026-10-28), a cell newly
    marked whole_suite too."""
    found = []
    for row_ in verdict["rows"]:
        cells = set(row_.get("whole_suite", []))
        for test, seconds in (row_.get("tests") or {}).items():
            if seconds <= target or (base is not None and test in base):
                continue
            found.append({"test": test, "file": test.partition("::")[0], "seconds": round(seconds, 1),
                          "target": target, **({"whole_suite": True} if test in cells else {})})
    return sorted(found, key=lambda item: (-item["seconds"], item["test"]))


def flagged(verdict: Path, new: Path | None) -> int:
    """`slow` on the verdict at `verdict` and this checkout's group target, each test a line of JSON; with
    `new` (a durations record), 1 where any test that record lacks passes the target."""
    target = json.loads(Path(".armada.json").read_text())["target"]
    base = json.loads(new.read_text()) if new is not None else None
    found = slow(json.loads(verdict.read_text()), target, base)
    for item in found:
        print(json.dumps(item))
    return 1 if found and base is not None else 0


def task_arguments(running: argparse.ArgumentParser) -> None:
    """The `task` operation's arguments, which armada's task command fills from a plan entry."""
    running.add_argument("--python", choices=PYTHONS, required=True)
    running.add_argument("--tests", required=True)
    running.add_argument("--split", default="")
    running.add_argument("--out", type=Path, required=True)
    running.add_argument(
        "--cache", required=True,
        help="the plan's DEW_TEST_CACHE (`policy`): on, or off")
    running.add_argument("--whole", choices=("0", "1"), default="0",
                         help="the plan's DEW_WHOLE_SUITE: 1 on the whole suite's run, which runs its cells")
    running.add_argument("--deadline", type=float, default=session.DEADLINE,
                         help="seconds before armada's own task timeout (.armada.json) to interrupt at")
    running.add_argument("--coverage", type=Path, default=None,
                         help="run the files under coverage.py, its data file in this directory")


def release_arguments(planning: argparse.ArgumentParser, checking: argparse.ArgumentParser) -> None:
    """The `release-plan` and `release` operations' arguments: a whole suite's second job (`release_plan`)."""
    planning.add_argument("verdict", type=Path, help="armada verdict <sha> --json")
    checking.add_argument("--python", choices=PYTHONS, required=True)
    checking.add_argument("--tests", required=True)
    checking.add_argument("--requirement", required=True)
    checking.add_argument("--out", type=Path, required=True)


def releasing(args: argparse.Namespace) -> int:
    """`release-plan`'s tasks printed, or `release`'s checks run (`release_plan`, `release`)."""
    if args.operation == "release-plan":
        print(json.dumps(release_plan(json.loads(args.verdict.read_text()))))
        return 0
    return release(args.python, args.tests.split(), args.requirement, args.out)


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/armada/ci.py")
    operations = parser.add_subparsers(dest="operation", required=True)
    deciding = operations.add_parser("policy", help="the test cache's policy for this checkout's change")
    deciding.add_argument("base", help="the commit the change is read from (its merge-base with HEAD)")
    planning = operations.add_parser("plan")
    planning.add_argument("--target", type=float, required=True)
    planning.add_argument("--timings", type=Path, required=True)
    task_arguments(operations.add_parser("task"))
    selecting = operations.add_parser(
        "pytest", help="pytest on the arguments as a CI task runs it, CI's selection first; its exit status")
    selecting.add_argument("--python", choices=PYTHONS, required=True)
    selecting.add_argument("--report", type=Path, required=True, help="where pytest writes its JUnit report")
    selecting.add_argument("arguments", nargs=argparse.REMAINDER,
                           help="after --, pytest's own: files, node ids and options")
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
    lane.add_argument("--tests", help="the test files; with --network, every file holding a "
                                       "network-marked test by default (`network_files`)")
    lane.add_argument("--network", action="store_true",
                      help="the network-marked tests alone, with the Hub open, on the network lane's class")
    lane.add_argument("--out", type=Path, required=True)
    lane.add_argument("--cache", required=True, help="the test cache's policy (`policy`)")
    lane.add_argument("--processes", type=int, required=True, help="the most test files running at once")
    lane.add_argument("--deadline", type=float, default=3600.0, help="seconds a file may run")
    operations.add_parser("gangs")
    versions = operations.add_parser("pythons")
    versions.add_argument("--check", action="store_true",
                          help="refuse a .python-version other than the oldest, or a smoke of other versions")
    flagging = operations.add_parser(
        "slow", help="each test of a verdict past the group target, a JSON line each; --new: exit 1 on any")
    flagging.add_argument("verdict", type=Path, help="armada verdict <sha> --json")
    flagging.add_argument("--new", type=Path, default=None,
                          help="the durations record the change is measured against: only the tests it lacks")
    release_arguments(
        operations.add_parser("release-plan", help="a whole suite verdict's release checks, a task each"),
        operations.add_parser("release", help="one requirement's release checks; exit 1 on a red row"))
    recording = operations.add_parser("durations")
    recording.add_argument("verdict", type=Path, help="main's CI verdict: armada verdict <sha> --json")
    args = parser.parse_args()
    if args.operation == "pythons":
        return pythons(check=args.check)
    if args.operation == "gangs":
        print(json.dumps(gangs()))
        return 0
    if args.operation == "slow":
        return flagged(args.verdict, args.new)
    if args.operation in ("release-plan", "release"):
        return releasing(args)
    if args.operation == "durations":
        target = json.loads(Path(".armada.json").read_text())["target"]
        before = durations_record()
        recorded = folded(before, json.loads(args.verdict.read_text()), target)
        if recorded != before:
            session.DURATIONS.write_text(json.dumps(recorded, indent=0) + "\n")
        changed = sorted({node.split("::")[0] for node in before.keys() ^ recorded.keys()}
                         | {node.split("::")[0] for node in before.keys() & recorded.keys()
                            if before[node] != recorded[node]})
        print(f"{len(changed)} files refolded in {session.DURATIONS}: {' '.join(changed) or 'none'}")
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
    if args.operation == "pytest":
        # argparse keeps the `--` that sets pytest's arguments apart from these.
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        return targeted(args.python, arguments, args.report)
    if args.operation == "gpu-lane":
        if args.tests is None and not args.network:
            parser.error("gpu-lane names its --tests; only the network lane (--network) takes every file "
                         "holding a network-marked test by default")
        tests = args.tests.split() if args.tests is not None else list(network_files(PYTHONS[0]))
        return gpu_lane(PYTHONS[0], tests, args.out, cache=args.cache, processes=args.processes,
                        deadline=args.deadline, network=args.network)
    if args.operation == "coverage":
        if args.stage == "report":
            return coverage_report(args.directory)
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps(coverage_items(args.target, timings)))
        return 0
    if args.operation == "policy":
        print(policy(args.base))
        return 0
    if args.operation == "plan":
        print(json.dumps({"include": plan(args.target, json.loads(args.timings.read_text())
                                          if args.timings.is_file() else {})}))
        return 0
    if args.split == COLLECT:
        return collect(args.python, args.tests.split(), args.out, args.deadline)
    return task(args.python, args.tests.split(), args.split, args.out, args.deadline, args.cache,
                coverage=args.coverage, whole=args.whole == "1")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT
    python3 tools/armada/ci.py gpu-tier [--evidence] [--shards=N]
    python3 tools/armada/ci.py gpu-lane --tests="tests/b.py#1/2" --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py gpu-lane --network [--tests=...] --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py gpu-lane --prepare --tests="tests/b.py#1/2" --out=FILE --processes=N
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
every row it holds is red (`kind` "deadline"), so a hang is graded, with what
pytest reports as it is interrupted, rather than leaving the task without a
verdict. Its rows' seconds, which armada times the next plan by, are what the
task knows of each file (`deadline_shares`): the file its tests were running
at the deadline, the one whose test started last, carries its own tests'
seconds and the task's wall time no file's tests account for, at least what
the file needs, so the next plan cuts it finer; a file that finished its tests'
own seconds; and a file that never started has none: its row says
`"unobserved": true`, with `seconds` 0, and armada weighs no plan by it (its
`timings` still name its file, at 0, so a file it shares with other rows is not
timed by those alone). A check-1 site too long for one task, marked
check1_shards(seeds, per_task) on its test
(`sites`), runs apart from its file's tasks as armada's map and judge
(`site_entries`): map tasks of per_task of its seeds each, or fewer where their
measured seconds put a task past the group target (`cut_draws`), which draw them into
their artifacts and judge nothing, and a judge that waits on them, pools their
draws from its inputs and reports the site's row, `<file>#<test>`, as a run
that drew every seed itself would; a map task that does not pass exits 1, and
armada reads its judge red, naming it. Other work too long for one task, marked
shards(units, per_task), runs so too, its map tasks taking per_task of its
units each. A check-1 site that also declares a weighed check (reference_error's
`conditioning`), check1_shards(seeds, per_task, conditioning,
conditioning_per_task), draws its weighed units in map tasks of their own,
conditioning_per_task each, past its unweighted ones, every one a later task
its judge asks for. Each row
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
marked to run on a GPU (`gpu-tier --evidence` reports it); `gpu-tier` names the
files marked to run on a GPU, a day's and a batch's alike: the behavioural flows
(tests/test_gpu_*.py; no unit test runs on a GPU, ledger 2452). A GPU
node runs them with `gpu-lane`, each as a task of a CUDA lane, whose rows are
recorded and reused on the GPU's class as a CPU task's are on the CPU's. With `--prepare`, a lane's files
run first on the node's CPU, no GPU drawn, for what their GPU run reads and must not compute while it holds
the GPU (`gpu_prepared`: the tests marked `prepares`, their float64 truths and their programs compiled for
the node's GPU), nothing of it recorded. With `--network`, a GPU node that
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
import dataclasses
import functools
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Container, Mapping, Sequence
from pathlib import Path


@functools.cache
def _loaded(path: Path):
    """The module at `path`, loaded once by its path: this file runs as a script and is loaded by the tests,
    and puts nothing on sys.path. Its neighbours here, and the gate's standard-library contract the plan
    shares (tests/check1_wire.py, and src/dew/parity_stages.py's units, `PARITY_STAGES`)."""
    spec = importlib.util.spec_from_file_location(f"armada_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
@functools.cache
def check1_wire():
    """Check 1's map and judge contract, its one owner (tests/check1_wire.py): what runs a site's test as a
    map task (DRAWS, RECORDS), as their judge (POOL), a judge's ask for its later map tasks (LATER) and its
    stages log (STAGES). Loaded where a plan shards a site, so a checkout of the harness alone (a page
    generator's scratch copy) imports this file."""
    return _loaded(HERE.parents[1] / "tests" / "check1_wire.py")
PARITY_STAGES = HERE.parents[1] / "src" / "dew" / "parity_stages.py"
child_groups = _loaded(HERE / "child_groups.py")
session = _loaded(HERE / "session.py")
coverage_report = _loaded(HERE / "coverage_report.py")
r2 = _loaded(HERE / "r2.py")
watched = _loaded(HERE / "watched.py")
followed = _loaded(HERE / "followed.py")
progress = _loaded(HERE / "progress.py")
digests = _loaded(HERE / "digests.py")
grading = _loaded(HERE / "grading.py")
host = _loaded(HERE / "host.py")
stalls = _loaded(HERE / "stalls.py")
costs = _loaded(HERE / "costs.py")
command_line = _loaded(HERE / "arguments.py")
rows = _loaded(HERE / "rows.py")
row, parsed_row, row_place, key_parts = rows.row, rows.parsed_row, rows.row_place, rows.key_parts
VARIED, varied, cut_of = rows.VARIED, rows.varied, rows.cut_of
# The names the land queue's tools read from this module at every commit they run (land.sh's `ci.STALE`,
# run_tests.py's `node_of`), older commits' and newer alike.
STALE, node_of = grading.STALE, grading.node_of
code_of, unit, units = digests.code_of, digests.unit, digests.units
SETTLE = 30.0
"""Seconds a task's process group has, after SIGKILL, to be reaped and gone: inside armada's own timeout
with the default deadline and grace (session.DEADLINE + session.GRACE + 30 + the reader's 10 < .armada.json's
1800)."""

REPO = Path(__file__).resolve().parents[2]
HASHED_BY = "3.13"
"""The Python this file runs under wherever it runs, the plan's and every task's alike (`pinned`), so a key
the plan computes is the one a task recorded: a module's units are what this Python's parser and compiler
make of its text (`digested`), which each release changes (2026-10-30: plans under the PC's 3.14 and a GitHub
runner's 3.12 found none of the 3.13 tasks' records fresh, and every file took a task of its own). It is the
Python the tasks' image runs this file under, so a task runs it as it is."""
_CLASSIFIERS = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["classifiers"]
_PREFIX = "Programming Language :: Python :: "
PYTHONS = tuple(sorted((classifier.removeprefix(_PREFIX) for classifier in _CLASSIFIERS
                        if re.fullmatch(re.escape(_PREFIX) + r"\d+\.\d+", classifier)),
                       key=lambda version: tuple(int(part) for part in version.split("."))))
"""The tested versions pyproject.toml's classifiers declare, oldest first: the floor runs every
test on every run, the newest every test on the whole suite's run (DEW_WHOLE_SUITE) and on a pushed
run, whatever its cache's policy, the files whose behaviour changed with it (`NEWEST_FILES`).
`pythons` prints them for tools/armada/install.sh, and `pythons --check` holds .python-version
and the environment's smoke (.armada.json) to them."""
NEWEST_FILES = (
    # Annotations, which 3.14 evaluates lazily (PEP 649): records and their config (ec3c5fb54).
    "tests/test_config.py", "tests/test_encoders.py", "tests/test_registry_records.py",
    "tests/test_records.py", "tests/test_record_keys.py", "tests/test_linen_records.py",
    "tests/test_run_records.py",
    # The command line, through tyro's reading of those annotations, and the API report's reading of every
    # public signature (3082: a union printed `Optional[X]` on 3.12 and `X | None` on 3.14).
    "tests/test_config_cli.py", "tests/test_tpu_cli.py", "tests/test_api_report.py",
    # Serialization.
    "tests/test_pickle_sources.py",
    # Data loading: providers without TensorFlow (5cec9cc9d, ee7fa9ee0, 20294b686) and a
    # fetcher pool under 3.14's forkserver start (1b8a67804).
    "tests/test_data.py", "tests/test_providers.py", "tests/test_hf_data.py", "tests/test_data_mixtures.py",
    "tests/test_online_loader.py",
    # Threads at interpreter exit, which 3.14 ends differently (e8fececd5, 72caf7ed6), and the
    # compilation cache's codec (931d6f2e1).
    "tests/test_iterator_lifecycle.py", "tests/test_native_rollout_server.py", "tests/test_distribution.py",
    "tests/test_instrumentation.py",
    # Host memory a test bounds or records: a pool's export, once bounded by tracemalloc's peak, which moved
    # on 3.14 past half its model in CI's process alone and never saw XLA's buffers, now by resident pages
    # (ledger 3794); and a mixture's compile, its tracemalloc peak recorded.
    "tests/test_saved_sources.py", "tests/test_mixture_scale.py",
)
"""The files the newest Python runs whole on a pushed run, its cache's policy on or off: those its
changes broke before, and the ones around the same machinery. It imports every other test file there
(`COLLECT`), and runs every file on the whole suite's run (DEW_WHOLE_SUITE): a policy-off train ran every file
on both Pythons, doubling its suite (train 542's 3.14 rows were 43 container-hours of 87), which the
daily, the land queue's gate, runs anyway. The list is kept by hand, from those breaks, because no
record can derive it: the test cache records the checkout's files a run executed and read (executed.py),
while each break lived in the standard library's or a dependency's behaviour under the new
Python (PEP 649's lazy annotations read through tyro, forkserver as the default start method,
threads ended at interpreter exit, a process's allocations under tracemalloc), which no record
sees; and the plan runs where the records' token is not. A file red on the newest Python alone is
found by the daily run, which the land queue reads, and joins the list."""
COLLECT = "collect"
"""The split of a task that imports test files without running them."""
COLLECT_SECONDS = 6.0
"""A file's collection where armada recorded none: past the slowest rate it has run at (train 459: 5.1 s a
file on a Genoa host, 1.0 on a Turin one)."""
SKIPPED = {"tests/test_gen_api.py", "tests/test_multihost.py"}
"""CI runs the first in the lint job, under the griffe it pins, and the second in the multihost job,
across the containers of an armada gang."""


SHARED = ("pyproject.toml", "constraints.txt", "tools/armada/session.py", "tools/armada/child_groups.py",
          "src/dew/_cleanup.py", "tools/armada/split.py", "tools/armada/executed.py",
          "tools/armada/followed.py", "tools/armada/lifetime.py", "tools/armada/watched.py",
          "tools/armada/progress.py")
"""The files every test file's key covers whole, a Python module by its code (`code_of`), beside every
lock (a GPU node builds its environment from its CUDA lock), the environment's own inputs, which
.armada.json names (`environment_files`), and .armada.json's parts a test process sees (`armada_seen`):
what every test process runs or reads,
the session it runs in, the process group it runs as and the plugins loaded into it among them. The
plan's own code (this file) runs outside the tests, so a change to it reruns none of them; nor does
the durations record (`session.DURATIONS`), which only a split file's groups are cut from, so a
group's key covers its own file's entries alone (`cache_key`); nor does a fixture or a package's data,
which is the input of the files whose runs opened it, however they opened it (executed.py's `Opens`)."""
WHOLE_GPU = "0.75"
"""The share of a GPU a lane given the whole of it hands its files (`gpu_lane`): what JAX's client
preallocates where no XLA_PYTHON_CLIENT_MEM_FRACTION is set."""
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
    .armada.json's parts a test process sees (`armada_seen`), and the names of tests/'s conftest.py and
    __init__.py files and of src/'s own entries, whose arrival can change what an import finds (a conftest.py
    or a top-level package that shadows another), and of every data file under tests/ but its fixtures and
    under src/ (an existence probe before the recorder starts). A module within a package under src/ is its
    importers' input alone, with its package's other modules (`inputs`), a helper module under tests/ the
    input of each run that imported a module of its name (followed.py's `shadowing`), a test module no other
    file's, and a fixture is its openers', listers' and probers' (executed.py's `Opens`,
    followed.py's `PROBES`): so a train that loses a head runs again only what that head's files reached
    (owner, 2026-10-27: 8 fixtures changed reran all 485 tasks of train 222's second round)."""
    digest = hashlib.sha256(python.encode())
    # A conftest.py, which pytest loads by its path for every test beside and below it, and a package's
    # __init__.py; a helper module is the input of each run that imported a module of its name, which it
    # would shadow (followed.py's `shadowing`, digests.py's `looked_up`), so a helper of a new name reruns
    # no file (owner, 2026-10-29: train 315's one new helper under tests/ reran all 2490 tasks).
    for path in sorted(Path("tests").rglob("*.py")):
        if "__pycache__" not in path.parts and path.name in ("conftest.py", "__init__.py"):
            digest.update(str(path).encode())
    # The names of every data file under tests/ but its fixtures, and of every package's data: an existence
    # probe a library makes at import, before the recorder starts, is no record's. A fixture is its openers',
    # listers' and probers' alone, as a run records every probe it makes (followed.py's `PROBES`;
    # 2026-10-29: train 277 reused 2 of 1967 rows once a head added fixtures). A build's output and a fetched
    # payload, which a planner's checkout lacks, are not named: they are their build's and their manifest's
    # (`digests.built`, fetched_fixtures' `manifest`, `digests.contents`), whose names are.
    for root in ("tests", "src"):
        for path in sorted(Path(root).rglob("*")):
            if (path.is_file() and path.suffix != ".py" and not digests.built(path)
                    and digests.FIXTURES not in path.parents
                    and not digests.fetched.manifest(path).is_file()):
                digest.update(b"fixture:" + str(path).encode())
    armada = Path(".armada.json")
    digest.update(armada_seen(armada.read_text() if armada.is_file() else None).encode())
    for name in sorted({*SHARED, *map(str, Path("locks").glob("*.txt")), *environment_files(), "src"}):
        digest.update(name.encode() + (code_of(name) if name.endswith(".py") else digests.contents(name)))
    return digest.hexdigest()


def lane_inputs(python: str) -> str:
    """The hash of what a GPU lane's file shares (`gpu_lane`): what every test file's run shares (`shared`),
    and this file, which sets the environment, share and deadline each file runs under on a GPU, where a
    CPU task's are the session's."""
    return hashlib.sha256((shared(python) + code_of(__file__).hex()).encode()).hexdigest()


@dataclasses.dataclass
class Reading:
    """One reading of the checkout's inputs (`cache_key`): each input's part by its path, and each module's
    units by its path, read once, by whichever of the threads looking rows up asks first. A reading lasts
    while the checkout stands still: a plan's lookups, gpu_tier's, a task's before its tests run (2026-10-31:
    train 368's plan read each of a row's ~3,000 inputs again for each of 1,682 rows, 9.5 minutes before its
    first task)."""

    parts: dict[str, bytes] = dataclasses.field(default_factory=dict)
    modules: dict[str, dict[str, bytes] | bytes] = dataclasses.field(default_factory=dict)
    durations: dict[str, dict[str, float]] | None = None
    cuts: dict[tuple[str, int], list[list[str]]] = dataclasses.field(default_factory=dict)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def cut(self, name: str, count: int) -> list[list[str]]:
        """File `name`'s tests cut into `count` groups (`members`), the record parted by file once a reading
        and each cut once (train 662's plan: 3,620 group keys each parsed the record, a minute of two)."""
        # Under the lock: the plan's threads share a reading (train 679: a half-parted record cut empty).
        with self.lock:
            if self.durations is None:
                self.durations = parted_record()
            if (name, count) not in self.cuts:
                own = own_tests(name, self.durations.get(name, {}))
                self.cuts[(name, count)] = session.cut(list(own), list(own.values()), count)
            return self.cuts[(name, count)]

    def part(self, path: str) -> bytes:
        """Input `path`'s part of a key: a unit's digest (`unit`), or its contents' (`digests.contents`)."""
        found = self.parts.get(path)
        if found is None:
            with self.lock:
                found = self.parts.get(path)
                if found is None:
                    found = self.parts[path] = self.read(path)
        return found

    def read(self, path: str) -> bytes:
        module, _, executed = path.partition(followed.CODE)
        if not executed or executed == followed.WHOLE or not Path(module).is_file():
            return digests.contents(path)
        if module not in self.modules:
            self.modules[module] = units(module)
        found = self.modules[module]
        return found.get(executed, b"missing") if isinstance(found, dict) else found


def cache_key(common: str, name: str, split: str, depends: list[str],
              reading: Reading | None = None) -> str:
    """The key of test file `name`'s row in `split`'s group whose run executed or read `depends`
    (executed.py): its `inputs`' contents, its directories' entries, `common`, and for a group of a split
    file, the group and the tests split.py cuts into it from the durations record (`members`), the record's
    only part a run's outcome depends on: not their seconds, which each train's fold of main's durations
    moves (the maintainer, 2026-10-28: every train re-keyed its split files and reran them cold, row 2000).
    A check-1 site's judge is keyed by its test, its inputs its map tasks' with its own (`site_inputs`): the
    site's key; a map task's, by its seeds besides (`sites`). The inputs as `reading` finds them: the keys of
    one reading read each input once, however many rows name it."""
    digest = hashlib.sha256(common.encode())
    reading = Reading() if reading is None else reading
    for path in inputs(name, depends):
        digest.update(path.encode() + reading.part(path))
    if session.SITE.fullmatch(split):
        digest.update(split.encode())
    elif split and split != COLLECT:
        # A process variant's (`varied`) by its name beside its cut's tests; a whole file's variant, its name.
        cut = cut_of(split)
        digest.update(split.encode() + (json.dumps(members(name, cut, reading)).encode() if cut else b""))
    return digest.hexdigest()


def inputs(name: str, depends: list[str]) -> list[str]:
    """What test file `name`'s key covers, given that its run executed or read `depends`: those, its own code
    whole (followed.py's `WHOLE`), and the import lookups that found each module it executed under src/ or
    tools/ (`digests.lookups`): a module or package of its name added beside one it imported (which
    shadows it), or an __init__.py added to a namespace package on the way to one, which executes as the
    import passes, changes its key; a module of another name added beside it does not, unless an import its
    run tried and failed looks for that name (followed.py's `missed`). tools/ is searched so, as a test puts
    it, or a directory under it, on sys.path and imports a tool by its name; src/'s own entries every key
    covers (`shared`). The
    harness's own modules (tools/armada/, which runs every test process and records it, `SHARED`) are no
    import of a test's, so their lookups are none of its inputs (the maintainer, 2026-10-29: train 296's one
    added tool reran 1,794 of 1,795; train 668: one module added to nn/backbones re-keyed 589 rows)."""
    own = f"{name}{followed.CODE}{followed.WHOLE}"
    return sorted({own, *(path for path in depends if not digests.built_input(path)),
                   *(looked for path in depends
                     for looked in digests.lookups(path.partition(followed.CODE)[0]))})


def cpu_class() -> str:
    """This host's CPU microarchitecture, the one class the records, the tests' compilation cache and
    the shared XLA bundles key it by (`digests.host`)."""
    return digests.host().cpu_class()


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


SEND = r2.KeptSend()
"""How every request to the Worker is sent: on a connection its thread keeps (`r2.KeptSend`)."""


def store(token: str) -> r2.Store:
    """The Worker that holds the records."""
    return r2.Store(os.environ["DEW_XLA_CACHE_URL"], token, "dew-ci-test-cache", timeout=10, send=SEND)


def record_key(python: str, row_name: str, held: str, inputs: str | None = None) -> str:
    """Where the record of `row_name` on class `held` (`device_class`) lives: the last run's, or with
    `inputs`, the last of the runs on those shared inputs (`shared`), apart from the last runs' list
    (`listed`). Any run of another commit replaces the last record, so a commit's retry reads its own
    under its inputs first (`reused`). The Worker's key alphabet has no colon or hash."""
    at = f"{CACHE_PREFIX}/{python}" if inputs is None else f"{CACHE_PREFIX}-inputs/{inputs[:16]}/{python}"
    return f"{at}/{row_place(row_name, held)}"


def green_key(python: str, row_name: str, held: str, key: str) -> str:
    """Where row `row_name`'s green run on class `held` whose inputs hash to `key` (`cache_key`) is kept,
    apart from the last run's record (`record_key`): every green run of a row on any inputs stays found by
    its own key, so a run of another commit, which replaces the last record, loses none of them
    (`green_record`; owner, 2026-10-29: a lane's one-line change over a green verdict reran half its suite,
    its records replaced). The row's own place holds it, as the files one process ran share what it executed
    before them and so their keys (`executed.py`): kept by the key alone, one file's run stood for each of
    them (2026-10-30: a kept task reported one file's row for six files)."""
    return f"{CACHE_PREFIX}-green/{python}/{row_place(row_name, held)}/{key}"


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


def pinned_classes() -> list[str]:
    """The CPU classes .armada.json pins its tasks to (task.cpu, one or a list), as `cpu_class` names them,
    which their rows are recorded under, a task's on the class it lands on; none where it pins none, or the
    tasks run on a CUDA lane, whose class (`device_class`) the plan cannot know."""
    config = Path(".armada.json")
    if not config.is_file() or "cuda" in os.environ.get("JAX_PLATFORMS", "").split(","):
        return []
    cpu = json.loads(config.read_text()).get("task", {}).get("cpu")
    return [f"{each['vendor']}-{each['family']}-{each['model']}" for each in
            ([cpu] if isinstance(cpu, dict) else cpu or [])]


def reusable(python: str, rows: list[tuple[str, str]], common: str) -> dict[tuple[str, str], str]:
    """The rows of `rows` (each a file and its group, "" for the whole file) that ran green on exactly these
    inputs (`common`, and each file's own, `fresh`) on any class the tasks run on (`pinned_classes`), which a
    task of theirs could land on, read at plan time with the cache's read token (`READ_TOKEN`): a task of
    their own would only look each up and exit. A heavy file's groups are rows like any other, each looked up
    by its own (2026-10-30: the plan looked up whole files alone, and a docs-only change's 1,034 green groups
    of heavy files each booted a container to read its record). Each with where its green run is kept
    (`green_key`'s class and key, `KEPT_AT`'s token), which the plan hands the task that holds it, so no run
    of another commit replacing a row's last record between plan and task can hide it (2026-10-30: a kept
    task found 23 of its 259 files and ran the rest). None where the plan has no token or no class (the
    maintainer, 2026-10-28: a lane run with most files unchanged booted a container a task; 2026-10-29: the
    suite pinned to 25/17 and 26/17)."""
    token, held = os.environ.get(READ_TOKEN), pinned_classes()
    if not token or not held or "DEW_XLA_CACHE_URL" not in os.environ:
        return {}
    # The plan's one reading of the checkout, which every row's key reads its inputs from.
    reading = Reading()

    def hit(each_row: tuple[str, str]) -> str | None:
        name, group = each_row
        for each in held:
            found = green_record(python, name, group, common, token, [each], reading)
            # Kept only where its green run is kept by its key, as the task reads it.
            if found is not None and recorded(green_key(python, row(python, name, group), each, found["key"]),
                                              token) is not None:
                return f"{each}:{found['key']}"
        return None

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        return {each: found for each, found in zip(rows, pool.map(hit, rows), strict=True) if found}


def eager_sites(python: str, sites: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
    """The check-1 sites of `sites` (each its file and test) whose judge's last run on a class the tasks
    run on (`pinned_classes`) needed its later map tasks (`LATER_JUDGED`), which the plan then draws in the
    first wave (`site_entries`): read with the plan's read token from each class's last-run record
    (`record_key`, whatever inputs it ran on and whatever its key), apart from reuse, which it never grants.
    A site no class's record says so of, a record or field missing, or a Worker that did not answer, keeps
    its later tasks later."""
    token, held = os.environ.get(READ_TOKEN), pinned_classes()
    if not sites or not token or not held or "DEW_XLA_CACHE_URL" not in os.environ:
        return set()

    def needed(site: tuple[str, str]) -> bool:
        return any(((recorded(record_key(python, row(python, *site), each), token) or {}).get("row") or {})
                   .get(LATER_JUDGED) is True for each in held)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        return {site for site, later in zip(sites, pool.map(needed, sites), strict=True) if later}


KEPT_AT = "@"
"""What joins a kept row's file to where its green run is kept in a plan entry's tests (`reusable`):
`tests/test_a.py@<class>:<key>`, or a group's, `tests/test_a.py#2/3@<class>:<key>`, which the task reads
(`kept_of`)."""

GROUP_AT = "#"
"""What joins a kept row's file to its group in its token (`KEPT_AT`), as a row's name does (`row`)."""


def kept_token(name: str, group: str, at: str) -> str:
    """The token of file `name`'s row in `group` ("" for the whole file), kept where `at` says (`KEPT_AT`)."""
    return f"{name}{GROUP_AT + group if group else ''}{KEPT_AT}{at}"


KEPT_BYTES = 64 * 1024
"""The most bytes of tokens and row names a kept task holds: its entry travels whole as one environment
string (armada's ARMADA_ITEM) and its tokens as one argument (--tests), each of which Linux starts no process
with past 131,071 bytes (MAX_ARG_STRLEN); half of that, for the entry's other fields (2026-10-31: main
de315f9a05's kept rows were one task, whose 149,515-byte argument started none)."""


def kept_tasks(python: str, found: Mapping[tuple[str, str], str],
               target: float) -> list[list[tuple[str, str]]]:
    """The rows `found` kept (`reusable`), in tasks of at most the target's lookups, a lookup a row in
    seconds, so no kept task outlasts the rest of the run, and at most `KEPT_BYTES` of their tokens and
    names."""
    tasks: list[list[tuple[str, str]]] = []
    held: list[tuple[str, str]] = []
    size = 0
    for name, group in sorted(found):
        # Its token, its row's name, and the JSON quotes and separators around them.
        cost = len(f"{kept_token(name, group, found[(name, group)])} {row(python, name, group)}".encode()) + 6
        if held and (len(held) >= max(int(target), 1) or size + cost > KEPT_BYTES):
            tasks.append(held)
            held, size = [], 0
        held.append((name, group))
        size += cost
    return tasks + ([held] if held else [])


def kept_of(tests: list[str]) -> tuple[list[str], dict[tuple[str, str], tuple[str, str]]]:
    """A task's `tests`, each a file's name, or a kept row's token, its file's name with its group where it
    has one and where its green run is kept (`KEPT_AT`): the files the task runs whole, those kept among
    them, and each kept row's class and key by its file and group."""
    names, kept = [], {}
    for test in tests:
        at_row, _, at = test.partition(KEPT_AT)
        name, _, group = at_row.partition(GROUP_AT)
        if not group:
            names.append(name)
        if at:
            held, _, key = at.partition(":")
            kept[(name, group)] = (held, key)
    return names, kept


def green_record(python: str, name: str, split: str, common: str, token: str,
                 classes: list[str], reading: Reading | None = None) -> dict | None:
    """The record of test file `name`'s row (in `split`'s group) that ran green on exactly these inputs on one
    of `classes`, the first found; else None. The plan (`reusable`) and a task (`reused`) read it alike, from
    one reading of the inputs (`Reading`)."""
    for each in classes:
        found = recorded(record_key(python, row(python, name, split), each, common), token)
        if found is None or found["depends"] is None:
            continue
        # The last run's inputs hashed once: its own key where fresh (`fresh`), else the green run of exactly
        # these, kept by that key (`green_key`) whatever ran since.
        key = cache_key(common, name, split, found["depends"], reading)
        if found["key"] is not None and found["key"] == key:
            return found
        kept = recorded(green_key(python, row(python, name, split), each, key), token)
        if kept is not None and kept["key"] == key:
            return kept
    return None


def reused_classes(network: bool = False) -> list[str]:
    """The classes whose green rows a task on this host reuses: every class the tasks are pinned to, where
    this host's is one of them, as the plan counted a row green on any of them reusable (`reusable`), so the
    task holding the files it kept reuses them wherever it lands; else this host's class alone."""
    own, held = device_class(network), pinned_classes()
    return held if own in held else [own]


def reused(python: str, name: str, split: str, common: str, network: bool = False,
           kept: tuple[str, str] | None = None, reading: Reading | None = None) -> dict | None:
    """The green row of test file `name` (in `split`'s group) recorded on a class this host reuses
    (`reused_classes`), if it ran green there on exactly these inputs; else None. A file its plan kept
    (`kept`: its green run's class and key, `reusable`) reads that run, fresh here; any other, the record of
    the last run on these shared inputs: a key hashes them (`cache_key`), so no run on others, whose record
    the last run's may be, can be fresh here, all from one reading of the inputs (`Reading`)."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return None
    if kept is not None:
        found = recorded(green_key(python, row(python, name, split), *kept), token)
        if found is not None and found["key"] == kept[1] and fresh(found, name, common, reading):
            return found["row"]
    found = green_record(python, name, split, common, token, reused_classes(network), reading)
    return None if found is None else found["row"]


def fresh(recorded: dict, name: str, common: str, reading: Reading | None = None) -> bool:
    """Whether test file `name`'s record was made by a green run on exactly these inputs, as `reading` finds
    them (`cache_key`). A red run, or one of a file that started another Python process, whose code no event
    saw, leaves no key."""
    return recorded["key"] is not None and recorded["key"] == cache_key(
        common, name, parsed_row(recorded["row"]["name"])[2], recorded["depends"], reading)


RECORD_TRIES = 4
"""How many times a record's write is tried, two, four and eight seconds apart."""


def recorded_tests() -> dict[str, list[float]]:
    """Each file's tests' times recorded (`own_tests`), one shared setup's one weight (`session.units`)."""
    found = {name: own_tests(name, tests) for name, tests in parted_record().items()}
    return {name: session.unit_seconds(list(own), list(own.values())) for name, own in found.items() if own}


def parted_record() -> dict[str, dict[str, float]]:
    """The durations record by test file, each file's tests in node-id order."""
    parted: dict[str, dict[str, float]] = {}
    record = durations_record()
    for node in sorted(record):
        parted.setdefault(node.split("::")[0], {})[node] = record[node]
    return parted


def own_tests(name: str, tests: dict[str, float]) -> dict[str, float]:
    """Of file `name`'s recorded `tests`, those but its check-1 sites', which run apart (`sites`)."""
    return session.own(tests, sites(name))


def durations_record() -> dict[str, float]:
    """The durations record (`session.DURATIONS`): each test's node id and its seconds."""
    return json.loads(session.DURATIONS.read_text()) if session.DURATIONS.is_file() else {}


def members(name: str, split: str, reading: Reading | None = None) -> list[str]:
    """The node ids split.py cuts into group `split` ("G/N") of test file `name`'s recorded tests but its
    sites' (`own_tests`), by their seconds, a shared setup's together, once a reading (`Reading.cut`)."""
    index, count = (int(part) for part in split.split("/"))
    return (Reading() if reading is None else reading).cut(name, count)[index - 1]


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
        # A map task's seconds are kept under its task's name, no row's; a site's judge is no group of the cut
        # (`sites`).
        if ":" not in row_name:
            continue
        python_, name_, split_ = parsed_row(row_name)
        # Nor is a GPU lane's process variant (`varied`): no armada plan cuts by one.
        if ((python_, name_) != (python, name) or split_ == COLLECT or session.SITE.fullmatch(split_)
                or VARIED in split_):
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


PLACED = ("src/", "tests/", "tools/", "examples/", "recipes/", "docs/", "tutorials/", "site/", ".github/",
          "kernels/", "stubs/", "licenses/", "images/", ".thresh/")
"""Where a changed file is placed by the records (executed.py): a Python file a test ran, a page, a
notebook or a workflow it read (followed.py's `HIDDEN_INPUTS`), a fixture its openers read. A native
kernel package's sources (kernels/), which the environment installs as a built wheel, never as these
files, a type stub, a license, an image and thresh's annotations are placed alike: each is the input of
the files whose runs read it, of none where none does. A change elsewhere is placed by none, and runs
the whole suite (`policy`): a workflow change reran every file while no record placed it (the
maintainer, 2026-10-28: 5 of 10 trains ran with the cache off, 0.75% of rows reused), and a change to
a kernel package's wrapper ran train 537's whole suite on both Pythons."""
PLACED_FILES = ("README.md", "CONTRIBUTING.md", "AGENTS.md", "LICENSE", "CHANGELOG.md", "MANIFEST.in")
"""The checkout's own files at its root that tests read as a page is read, and the changelog, which
no test reads (test_changelog_merge writes its own) and most heads add an entry to: placed by none,
it turned the cache off for every train holding such a head, which then ran its whole suite on every
round (train 149: 30 task-hours, twice)."""


def policy(base: str) -> str:
    """The test cache's policy (DEW_TEST_CACHE) for this checkout's change from `base`: 'off', every file run,
    for a change to what every test process runs or reads (`SHARED`, a Python module of it by its code, as
    `shared` keys it; a lock, the environment's files) or to a file the records do not place (`PLACED`); else
    'on', each file run where its inputs changed. A tool is such an input of each file whose run executed or
    read it, which its key hashes, so a changed tool runs those files again and no other (train 115 reran its
    whole suite for five tools no pulled head had touched, under a policy that reran every file that read any
    tool)."""
    changed = subprocess.run(["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"], check=True,
                             capture_output=True, text=True).stdout.split()
    if ".armada.json" in changed:
        before = at_commit(base, ".armada.json")
        armada = Path(".armada.json")
        if armada_seen(None if before is None else before.decode()) == armada_seen(
                armada.read_text() if armada.is_file() else None):
            changed.remove(".armada.json")  # where and when tasks run, never what a test sees
    whole = {*SHARED, *map(str, Path("locks").glob("*.txt")), *environment_files(), ".armada.json",
             ".python-version"}
    # A SHARED module whose code `base` holds as it is (a docstring's or a comment's edit) keys every row as
    # it did (`code_of`, `shared`), so it leaves the cache on (Main, 2026-11-07: 12 of the queue's last 20
    # train runs ran cold on harness edits).
    changed = [name for name in changed
               if not (name in SHARED and name.endswith(".py") and same_code(base, name))]
    if any(name in whole or not (name.startswith(PLACED) or name in PLACED_FILES) for name in changed):
        return "off"
    return "on"


def same_code(base: str, name: str) -> bool:
    """Whether Python file `name` is the same code at commit `base` as in this checkout (`code_of`): a file
    added or removed since is not."""
    before = at_commit(base, name)
    return before is not None and Path(name).is_file() and digests.code_digest(before, name) == code_of(name)


def at_commit(commit: str, name: str) -> bytes | None:
    """File `name`'s bytes at `commit`, or None where it holds none."""
    found = subprocess.run(["git", "show", f"{commit}:{name}"], capture_output=True, check=False)
    return found.stdout if found.returncode == 0 else None


def borrowed(timings: dict, python: str) -> dict:
    """`timings` with each file's and row's time `python` has none of taken from another Python's
    (the most of theirs): 3.14 runs every file only on the daily run, and a file weighed by its
    tests' least recorded times instead is dispatched late by armada's longest-first order (train
    143: 3.14 tasks of 500-800 s started 550-760 s into the run)."""
    def filled(times: dict) -> dict:
        others: dict[str, float] = {}
        # A map task's seconds are its own Python's, under its task's name (`site_task`).
        for name, seconds in times.items():
            if ":" not in name:
                continue
            other, rest = name.split(":", 1)
            if other != python and f"{python}:{rest}" not in times:
                others[f"{python}:{rest}"] = max(seconds, others.get(f"{python}:{rest}", 0.0))
        return {**times, **others}
    return {**timings, **{key: filled(timings.get(key, {})) for key in ("files", "rows")}}


COLD_SECONDS = 360.0
"""What a task adds to its estimate when it runs cold, the test cache's policy off (a lock, JAX's build or
the recorder changed): the compiles and imports no warm run paid. Train 254's cold suite ran about 470 s a
task where its plan estimated about 120 (1,389 tasks in 2,700 s on 242 containers, 2026-10-28)."""

SITE_MARK = re.compile(r"\bmark\.(?:" + "|".join(session.SHARDS) + r")\b")
"""A site's mark as a test file's source spells it (`sites`), which a file that holds none never parses."""
LATER_JUDGED = "check1_later"
"""A check-1 site judge's row's word that its judgement needed its later map tasks (`later_judged`), which the
next plan reads to draw them in the first wave (`eager_sites`)."""
INPUTS = "/armada/inputs"
"""Where a judge's task finds its map tasks' artifacts (armada's `{inputs}`), each task's under
`<job>/<index>`, where armada names none (`armada_dir`)."""
MAX_INPUTS = 1000
"""The most tasks' artifacts one armada job takes as its inputs (armada's MAX_INPUT_RUNS,
src/protocol.ts): armada cuts the judges of one environment and size into as many judges' jobs as keep
each within it (its `judgeChunks`), so a site's own map tasks, which one judge reads in one job, stay
within it (`refused_sites`)."""


@dataclasses.dataclass(frozen=True)
class Mark:
    """A site's mark (`session.SHARDS`): the units its map tasks together take, 0 to `units`, the most of them
    a map task takes, the line of its test's source that holds the mark, and the units a whole run of its test
    takes at least, over which its recorded seconds weigh one (`site_entries`): a check-1 site's declared
    seeds, of (1 + REDRAW) times as many its map tasks draw; a `shards` test's units, every one. A check-1
    site's weighed check (`conditioning`) draws `weighed` units more, `units` to `units + weighed`, at most
    `weighed_per_task` a map task: its draws cost their own, apart from the unweighted check's."""

    units: int
    per_task: int
    line: int
    timed: int
    weighed: int = 0
    weighed_per_task: int = 0


def armada_dir(variable: str, default: str) -> Path:
    """The directory armada names to a task by `variable` (ARMADA_ARTIFACTS, ARMADA_INPUTS), which every task
    it starts holds, else where its container keeps it (`default`)."""
    return Path(os.environ.get(variable) or default)


def sites(name: str) -> dict[str, Mark]:
    """Test file `name`'s tests marked one of `session.SHARDS`, each by its name in the file (a class's
    method's `Class.test_name`) with its mark (`Mark`). Read from the source, as `gangs` reads its models: the
    plan runs under a Python with no pytest (`pinned`) and imports no test, so it reads the mark on the test's
    own decorator, `@pytest.mark.check1_shards(seeds=S, per_task=P)` or `@pytest.mark.shards(units=N,
    per_task=P)`, of literal counts above zero, one a test, or a parameter's of the test's one literal
    parametrize list, `pytest.param(..., id="<id>", marks=pytest.mark.check1_shards(...))`, its site
    `test[id]`. A mark written otherwise (on a class or a module, on a parameter with no literal id or of a
    test parametrized twice, of counts computed, two on one test) would set its site aside from its file's
    tasks (`selected`) and plan none of its own, so nothing would run it: its file is refused, by the lines
    that hold it."""
    path = Path(name)
    text = path.read_text() if path.is_file() else ""
    return marked_sites(name, text) if SITE_MARK.search(text) else {}


@functools.cache
def marked_sites(name: str, text: str) -> dict[str, Mark]:
    """`sites` of test file `name`, whose source is `text`, read once a text: each site's mark as its source
    declares it (`session.declared_marks`), its units staged as check 1 draws them (`PARITY_STAGES`)."""
    staged = _loaded(PARITY_STAGES).staged_units
    return {site: Mark(staged(declared) if kind == "check1_shards" else declared, counts["per_task"],
                       line, declared, staged(counts.get("conditioning", 0)),
                       counts.get("conditioning_per_task", 0))
            for site, (kind, counts, line) in session.declared_marks(name, text).items()
            for declared in [counts[session.SHARDS[kind]]]}


def draws_of(mark: Mark, per_task: int | None = None, weighed_per_task: int | None = None
             ) -> list[tuple[int, int]]:
    """The units each map task of a site marked `mark` takes, at most its per_task each (`per_task` where
    given, as `cut_draws` cuts finer): together 0 to its units, every seed both of check 1's stages draw, or
    every unit of a `shards` test's work; a check-1 site's first stage (0 to its declared seeds, `timed`) and
    its second each in tasks of their own, the second's `later` (`site_entries`); and its weighed check's
    units past them (`Mark.weighed`), at most `weighed_per_task` a task, every one later."""
    each = per_task or mark.per_task
    stages = [(0, mark.timed), (mark.timed, mark.units)] if mark.units > mark.timed else [(0, mark.units)]
    spans = [(start, min(start + each, stop)) for first, stop in stages for start in range(first, stop, each)]
    if not mark.weighed:
        return spans
    end, weighed = mark.units + mark.weighed, weighed_per_task or mark.weighed_per_task
    return spans + [(start, min(start + weighed, end)) for start in range(mark.units, end, weighed)]


def cut_draws(mark: Mark, rates: Mapping[bool, costs.Cost | None], target: float) -> list[tuple[int, int]]:
    """The spans of a site's map tasks (`draws_of`): its declared per_task each, or fewer where its
    measured cost (`site_rate`, the unweighted check's and the weighed check's apart, `rates` by whether
    weighed) puts a task of its declared cut past `target`: as many as `target` holds past its setup
    (`Cost.fitting`), at least one, so no map task is the run's longest for its mark's cut (row 4376: a
    64-seed shard ran 966 s where every other group took 256 s or less). Its tasks' starts are among what
    the cost measures, so a cut converges from above on the one that fills the target. A kind whose finer
    cut would pass what its judges' job takes (`MAX_INPUTS`, the two kinds' tasks together) keeps the finest
    of its own within it."""
    def fitting(units: int, declared: int, cost: costs.Cost | None) -> int:
        return declared if cost is None else max(1, min(units, cost.fitting(declared, target)))

    per_task = fitting(mark.units, mark.per_task, rates[False])
    weighed = fitting(mark.weighed, mark.weighed_per_task, rates[True]) if mark.weighed else None
    while per_task < mark.per_task and len(
            draws_of(mark, per_task, mark.weighed_per_task or None)) > MAX_INPUTS:
        per_task += 1
    while weighed is not None and weighed < mark.weighed_per_task and len(
            draws_of(mark, per_task, weighed)) > MAX_INPUTS:
        weighed += 1
    return draws_of(mark, per_task, weighed)


def refused_sites(marked: Mapping[str, Mapping[str, Mark]]) -> None:
    """Refuse the marks of `marked` (each test file's sites, `sites`) whose own map tasks pass what one
    judges' job takes (`MAX_INPUTS`): its judge reads them all in one job, which armada would refuse after
    every map task ran, and the run would go ungraded; marks within it pass it together, armada cutting
    their judges into jobs. A mark is refused on the run that adds it, naming its line."""
    over = {(name, test): count for name, held in marked.items() for test, mark in held.items()
            if (count := len(draws_of(mark))) > MAX_INPUTS}
    if over:
        each = "; ".join(f"{name}, line {marked[name][test].line}: {test}, {count} map tasks"
                         for (name, test), count in sorted(over.items()))
        raise SystemExit(f"a site marked check1_shards or shards plans more map tasks than the {MAX_INPUTS} "
                         "tasks' artifacts its judges' job takes; give a map task more units (per_task): "
                         f"{each}")


def site_task(python: str, name: str, test: str, span: tuple[int, int] | None = None) -> str:
    """The task name of test file `name`'s site `test` under `python`: its judge's, or with `span` its map
    task's that draws those seeds. armada keeps a task's name where it is at most 80 of [A-Za-z0-9._-],
    which a judge waits on its map tasks by (`after`) and armada keeps a map task's seconds under, so it stays
    one from plan to plan: its file's stem and a digest of the file and test."""
    digest = hashlib.sha256(f"{name}::{test}".encode()).hexdigest()[:12]
    drawn = "" if span is None else f"-draws-{span[0]}-{span[1]}"
    return f"{python}-{Path(name).stem[:40]}-{digest}{drawn}"


def rate(python: str, name: str, test: str, spans: Sequence[tuple[int, int]], measured: Mapping[str, float],
         timings: Mapping[str, float], *, weighed: bool) -> costs.Cost | None:
    """What a map task of test file `name`'s site `test` takes under `python`: its setup and a unit's draws as
    its last judge timed them from its map tasks' records (`timings`, armada's file timings, `site_costs`),
    whatever their sizes; else over those of its map tasks' `spans` that armada measured (under their task
    names, `site_task`, `costs.fitted`); None where it measured none. Tasks of one size, cut by their seconds
    over units, charged each draw the setup, so a site was cut finer and measured at one size again, run after
    run (2026-11-11: 565 map tasks to 1,344), which the records' setup ends. Once a site runs as map tasks,
    the durations record holds no whole run of it (its test's seconds are one task's, its judge's pooling
    among them), so its map tasks are what tell its cost."""
    found = costs.recorded(site_task(python, name, test), timings, weighed=weighed)
    return found or costs.fitted([(stop - start, seconds) for start, stop in spans
                                  if (seconds := measured.get(site_task(python, name, test, (start, stop))))])


def site_rate(python: str, name: str, test: str, mark: Mark, timings: Mapping[str, Mapping[str, float]], *,
              weighed: bool) -> costs.Cost | None:
    """What a map task of test file `name`'s site `test` takes under `python` (`rate`), by its judge's records
    of its map tasks (`timings`' files), else over its map tasks of any cut armada measured, by their names
    (`site_task`), each span read from its name's end: its weighed check's units apart from the rest
    (`weighed`), as they cost their own."""
    measured = timings.get("rows", {})
    prefix = f"{site_task(python, name, test)}-draws-"
    spans = [(int(start), int(stop)) for task in measured if task.startswith(prefix)
             for start, _, stop in [task[len(prefix):].partition("-")]]
    return rate(python, name, test, [span for span in spans if (span[0] >= mark.units) == weighed], measured,
                timings.get("files", {}), weighed=weighed)


def site_costs(python: str, name: str, test: str) -> dict[str, float]:
    """A judge's timings of its site's map tasks (`costs.timed`), from the records each wrote among its
    artifacts beside its draws (map_judge's `record_units`: `<site>.<start>-<stop>.seconds.json`), which its
    inputs hold (ARMADA_INPUTS); nothing where they hold none of the site's."""
    held = sites(name)
    if test not in held:
        return {}
    records = []
    for path in armada_dir("ARMADA_INPUTS", INPUTS).rglob("*.seconds.json"):
        record = json.loads(path.read_text())
        python_, _, rest = record["site"].partition("::")
        node = rest.rpartition("::")[0]
        if (python_, node.split("::")[0]) == (python, name) and session.site_name(node, held) == test:
            records.append(costs.Drawn(task_start=record["task_start"], task_stop=record["task_stop"],
                                       before=record["before"], draws=record["draws"]))
    return costs.timed(site_task(python, name, test), records, held[test].units)


def site_entries(python: str, marked: Mapping[str, Mapping[str, Mark]],
                 kept: Mapping[tuple[str, str], str], timings: dict, target: float,
                 eager: Container[tuple[str, str]] = ()) -> list[dict]:
    """The plan's entries for each site of `marked` (`sites`, by its file) under `python`: a map task each
    `per_task` of its units, or fewer where their measured rate puts a task past `target` (`cut_draws`),
    which runs its test over them alone, records their draws among its
    artifacts and judges nothing; and its judge, which runs once they all passed (armada's `after`), pools
    their draws from its inputs and reports the site's row as a run that drew every unit itself would. A site
    whose judge's row ran green on exactly these inputs (`kept`, `reusable`) has none: a kept task reads its
    row, and no draw runs. A map task weighs what armada measured it at, under its task's name
    (`site_task`), else its setup and units as the site's map tasks of its kind that armada measured cost
    (`rate`: its first stage runs every time, so a later task its judge has not asked for since weighs what
    its seeds take), else the target: armada keeps the measures of the tasks a run named, so a site the last
    run reused has none, and the durations record times its judge's pooling, no measure of its draws
    (`folded`); a judge, its row's pooling as armada measured it, else the target. A judge
    names no row: armada reads one whose map tasks did not all pass red under its task's name, naming them,
    which a named row would leave not graded. A check-1 site's tasks past its first stage are later, run
    only where its judge asks for them, but a site of `eager` (its judge's last run needed them,
    `eager_sites`), whose later tasks, its second stage's and its weighed check's alike, run in the first wave
    beside its first stage's: its judge reads each stage's seeds alone (`_staged` slices them), so its verdict
    is the one it reaches asking, and it asks for nothing its pool holds."""
    measured = timings.get("rows", {})
    entries = []
    for name, held in sorted(marked.items()):
        for test in sorted(held):
            if (name, test) in kept:
                continue
            mark = held[test]
            rates = {weighed: site_rate(python, name, test, mark, timings, weighed=weighed)
                     for weighed in (False, True)}
            spans = cut_draws(mark, rates, target)
            maps = []
            for start, stop in spans:
                group = f"{test}.draws-{start}-{stop}"
                task_name = site_task(python, name, test, (start, stop))
                # A weighed check's units cost their own, as those of them armada measured did.
                cost = rates[start >= mark.units]
                weight = measured.get(task_name) or (
                    cost.seconds(stop - start) if cost is not None else target)
                # A check-1 site's second-stage seeds, drawn only where its judge's first stage asks for
                # them (armada's `later`), or at once where its last judgement needed them (`eager`).
                maps.append({"name": task_name, "python": python, "tests": name, "split": group,
                             "rows": [row(python, name, group)], "weight": weight,
                             **({"later": True} if start >= mark.timed and (name, test) not in eager
                                else {})})
            entries += [*maps, {"name": site_task(python, name, test), "python": python, "tests": name,
                                "split": test, "after": [each["name"] for each in maps],
                                "weight": measured.get(row(python, name, test)) or target}]
    return entries


def plan(target: float, timings: dict, census: bool = False) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has
    timed runs alone, so however long it takes it holds up no other file, and its time is known from
    then on. Files marked `torchax` (`APART`) share tasks only with one another. Each entry carries
    the test cache's policy, DEW_TEST_CACHE here (`policy`), to its task. Under the policy "on", the files
    whose rows the cache holds for exactly these inputs (`reusable`) share tasks of the target's lookups each,
    which look each up again in their containers, so the verdict names every file and containers scale with
    the change. Under the policy "off" every task's weight carries COLD_SECONDS, so armada sizes the pool by
    the work, not the longest task, as warm estimates did. A site marked one of `session.SHARDS` runs as
    armada's map and judge, apart from its file's tasks (`site_entries`). The `census` (check 1's CENSUS) is
    the whole suite with the cache off, whatever this environment says, each entry carrying it to its task."""
    cache = "off" if census else os.environ.get("DEW_TEST_CACHE") or "on"
    # The whole suite's run alone (DEW_WHOLE_SUITE: the daily run's and the whole dispatch's, and the census)
    # runs whole_suite cells (`selected`) and every file on the newest Python.
    cells = "1" if census or os.environ.get("DEW_WHOLE_SUITE") else "0"
    # What every entry carries to its task (.armada.json's command): the cache's policy, the whole suite's
    # cells and the census.
    stamped = {"cache": cache, "whole": cells, "census": "1" if census else "0"}
    every = sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)
    # Every file's check-1 sites its marks shard, refused where one site's map tasks outnumber what its
    # judges' job takes (`refused_sites`).
    shards = {name: held for name in every if (held := sites(name))}
    refused_sites(shards)
    # Every file's shared setups, read so a mark the plan cannot read refuses its file now
    # (`session.shared_setups`), not where a cut first reads it.
    for name in every:
        session.shared_setups(name)
    recorded = recorded_tests()
    entries = []
    for python in PYTHONS:
        # Every file on every Python daily; on a pushed run, its cache's policy on or off, the newest takes
        # `NEWEST_FILES` whole.
        whole = python == PYTHONS[0] or cells == "1"
        files = every if whole else [name for name in every if name in NEWEST_FILES]
        timings = borrowed(timings, python)
        timed = weights(files, timings, python)
        # Each file's rows: a file heavier than the target runs as the groups it is cut into, each a row of
        # its own; a file one group holds runs whole, alone, under its own row.
        groups: dict[str, list[tuple[str, float]]] = {}
        for name, seconds in timed.items():
            if seconds is not None and seconds > target:
                at, count = measured_cut(timings, python, name) or (seconds, 1)
                cut = split(recorded.get(name, []), at, target, count)
                groups[name] = [(f"{group}/{len(cut)}" if len(cut) > 1 else "", weight)
                                for group, weight in enumerate(cut, start=1)]
        # Each check-1 site its file's marks shard is a row of its own, its judge's.
        marked = {name: shards[name] for name in files if name in shards}
        rows_of = [(name, "") for name in timed if name not in groups] + [
            (name, group) for name, cut in groups.items() for group, _ in cut] + [
            (name, test) for name, held in marked.items() for test in held]
        found = reusable(python, rows_of, shared(python)) if cache == "on" else {}
        # A lookup a row, in seconds: a kept task's weight is its container's start and those lookups.
        tasks: list[tuple[list[tuple[str, str]], float]] = [
            (held, float(len(held))) for held in kept_tasks(python, found, target)]
        tasks += [([(name, "")], target) for name, seconds in timed.items()
                  if seconds is None and (name, "") not in found]
        tasks += [([(name, group)], weight) for name, cut in groups.items() for group, weight in cut
                  if (name, group) not in found]
        light = {name: seconds for name, seconds in timed.items()
                 if seconds is not None and name not in groups and (name, "") not in found}
        apart = {name for name in light if APART.search(Path(name).read_text())}
        for family in (sorted(set(light) - apart), sorted(apart)):
            count = max(math.ceil(sum(light[name] for name in family) / target), 1)
            tasks += [([(name, "") for name in names], held)
                      for held, names in packed({name: light[name] for name in family}, count) if names]
        for index, (held_rows, weight) in enumerate(tasks, start=1):
            # A kept row says where its green run is kept, and its group where it has one (`kept_of`); a task
            # of one file's group runs that group (`split`).
            alone = held_rows[0][1] if len(held_rows) == 1 and held_rows[0] not in found else ""
            tokens = [kept_token(name, group, found[(name, group)]) if (name, group) in found else name
                      for name, group in held_rows]
            entries.append({"name": f"{python}-{index}", "python": python, "tests": " ".join(tokens),
                            "split": alone, "rows": [row(python, name, group) for name, group in held_rows],
                            "weight": weight + (COLD_SECONDS if cache == "off" else 0.0), **stamped})
        # The sites to run whose judge's last run needed its later map tasks draw them at once
        # (`eager_sites`).
        eager = eager_sites(python, [(name, test) for name, held in marked.items() for test in held
                                     if (name, test) not in found])
        entries += [{**entry, "weight": entry["weight"] + (COLD_SECONDS if cache == "off" else 0.0),
                     **stamped}
                    for entry in site_entries(python, marked, found, timings, target, eager)]
    # What the newest Python runs no test of, it imports (`COLLECT`): on a pushed run, every file but its own.
    rest = [] if cells == "1" else [name for name in every if name not in NEWEST_FILES]
    collecting = collect_entries(rest, timings, target)
    return keyed(entries + [{**entry, **stamped} for entry in collecting])


def keyed(entries: list[dict]) -> list[dict]:
    """`entries`, each with the `key` armada keeps its recorded use under (its `keyOf`): the rows it runs (a
    site's judge by its name), whether it was kept (it looks its rows up and runs none) and how it runs them
    (the cache's policy, the whole suite's cells, the census's compiles, each its own use), never an entry's
    name or weight, which each plan cuts anew. A task that recurs then packs on what it used, not on its
    class's estimate (2026-11-07: no record of a task ever matched the next plan's, and each container's
    CPU was about a quarter used). Hashed: a kept task's entry, which travels whole as one environment
    string, holds up to `KEPT_BYTES` of row names already."""
    def key(entry: dict) -> str:
        identity = {"rows": entry.get("rows") or [entry["name"]], "kept": KEPT_AT in entry["tests"],
                    **{field: entry.get(field) for field in ("cache", "whole", "census")}}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return [{**entry, "key": key(entry)} for entry in entries]


def collect_entries(names: list[str], timings: dict, target: float) -> list[dict]:
    """The newest Python's tasks that import `names` without running them (`collect`): each at most
    `target` seconds of files at the slowest per-file rate armada recorded for a collection (else
    `COLLECT_SECONDS`), and weighed at that. A collect task's rows split its seconds evenly, and the same
    files collect five times as long on one host class as on another, so the slowest rate keeps every task
    within the target on any host; weighed at the target, train 459's four tasks of 140 files started last
    and ran 485-910 s on Genoa hosts, and the run ended with them."""
    rates = [seconds for name, seconds in timings.get("rows", {}).items()
             if ":" in name and parsed_row(name)[0] == PYTHONS[-1] and parsed_row(name)[2] == COLLECT]
    rate = max(rates, default=COLLECT_SECONDS)
    # The files a task holds within the target, one where a file alone takes longer; and no task holds none:
    # armada cannot grade a row its entry names that its task never reports.
    each = max(math.floor(target / rate), 1)
    count = math.ceil(len(names) / each)
    return [{"name": f"{PYTHONS[-1]}-{COLLECT}-{index + 1}", "python": PYTHONS[-1], "tests": " ".join(held),
             "split": COLLECT, "rows": [row(PYTHONS[-1], name, COLLECT) for name in held],
             "weight": len(held) * rate}
            for index, held in enumerate(names[at::count] for at in range(count))]


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


def stale_row(python: str, name: str, group: str, why: str) -> dict:
    """Kept row `name` (in `group`'s group) said stale, not run (`grading.STALE`), with `why`: a row armada
    reads as not passed, so its run is graded, and the queue runs it again."""
    return {"name": row(python, name, group), "exitCode": 1, "seconds": 0.0,
            "output": f"stale, not run: the plan kept it on a record this task did not find, {why}",
            "tests": {}, "ynn_switched": [], "cells": {}, "timings": {row(python, name, ""): 0.0},
            **grading.red_as(grading.STALE, [])}


def stale_cost(name: str, group: str, reading: Reading) -> float:
    """What a kept row of test file `name` (in `group`'s group) costs a task that runs it, its record not
    found: its tests' recorded seconds, its check-1 sites' aside, which run apart (`own_tests`; a group's,
    `members`), each its least time, and what a run cold adds to them (`COLD_SECONDS`), as the plan weighs a
    cold row; a file the record does not name, that alone."""
    reading.cut(name, 1)
    own = own_tests(name, (reading.durations or {}).get(name, {}))
    nodes = members(name, cut, reading) if (cut := cut_of(group)) else list(own)
    return sum(own.get(node, 0.0) for node in nodes) + COLD_SECONDS


def afforded(python: str, tests: list[str], group: str, kept: Container[str],
             left: float) -> tuple[list[str], list[dict]]:
    """Of `tests` a task still has to run, those it runs and the rows of those it does not: a row its plan
    kept (`kept`) that this task did not find (its record replaced since) runs here only where what it costs
    run cold fits what is `left` of the task's deadline, in turn (`stale_cost`); any other is said stale, not
    run (`grading.STALE`), which the queue runs again as a row of its own, neither a red of its tests nor
    flaky (Main, 2026-11-07; row 4468: 163 of a kept task's 165 rows ran out its deadline one after
    another)."""
    fitting, stale_rows, reading = [], [], Reading()
    for name in tests:
        cost = stale_cost(name, group, reading)
        if name in kept and cost > left:
            stale_rows.append(stale_row(python, name, group, f"its {cost:.0f} s past the {left:.0f} s left"))
            continue
        fitting.append(name)
        left -= cost
    return fitting, stale_rows


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
    # A collection runs no test, so it moved no site onto YNNPACK's fusions: counted, not unknown.
    rows = [{"name": row(python, name, COLLECT), "exitCode": 1 if name in errors else 0, "seconds": seconds,
             "output": errors.get(name, ""), "ynn_switched": [], "cells": {},
             **(grading.red_as(COLLECT, []) if name in errors else {})}
            for name in tests]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def targeted(python: str, arguments: list[str], report: Path, tests: Sequence[str] = (),
             split: str = "", whole: bool = False) -> int:
    """pytest's exit status on `arguments` (files, node ids and options) and `tests` in `split`'s group, run
    as a CI task runs them (`session.pytest_run`, `selection`, `site_environment`): a lane's targeted run
    (tmp/run_tests.py) and the gangs select as CI does, and an option of their own (a `-m` after CI's)
    chooses instead; with `whole`, as main's whole run does, so a bisect of its red whole_suite cell runs it
    (row 4230). A file's group sets aside its sites as CI's does (`selected`), which run in their own
    tasks alone (2026-11-02: a targeted group ran a site's every seed itself, 2233 s against CI's 8)."""
    site = session.SITE.fullmatch(split)
    run = session.selected(session.pytest_run(python, report), network=False, whole=whole,
                           site=site is not None)
    return subprocess.run([*run, *arguments, *session.selection(list(tests), split)],
                          env=session.pytest_environment(site_environment(site)), check=False).returncode


def flow_cells(cases) -> dict[str, dict]:
    """A GPU flow's counts by its cases (each case's JUnit property `cells`, tests/conftest.py from
    tests/gpu_flow.py: passed, failed, skipped and the device), as every row a task runs carries them, an
    empty object where its file runs no flow: the train's native gate reads a flow none of whose cells ran as
    unqualified, and a reused row keeps its counts (ledger 2643)."""
    return {grading.node_of(case): json.loads(item.get("value") or "null") for case in cases
            for item in case.iter("property") if item.get("name") == "cells"}


def ynn_switched(cases) -> list[dict]:
    """The accuracy gate's sites a file's cases moved onto YNNPACK's fusions (each case's JUnit property
    `ynn_switched`, tests/conftest.py), green cases included: each its test node, label and XLA:CPU's
    refusal, as every row a task runs carries them for the train report to count, an empty list where
    nothing moved, so a row without the field is one recorded before it, uncounted."""
    return [{"node": grading.node_of(case), **site} for case in cases
            for item in case.iter("property") if item.get("name") == "ynn_switched"
            for site in json.loads(item.get("value") or "[]")]


def apportioned(own_seconds: Mapping[str, float], wall: float, rows: int) -> dict[str, float]:
    """Each file's share of a task's `wall` time, in proportion to its tests' own times (`task`), or an
    even split over the task's `rows` (a file named twice is two) where no test recorded a time."""
    total = sum(own_seconds.values())
    pace = wall / total if total > 0 else 0.0
    return {name: own * pace if pace else wall / rows for name, own in own_seconds.items()}


def deadline_shares(own_seconds: Mapping[str, float], reported: Container[str], wall: float,
                    running: str | None) -> dict[str, float]:
    """The seconds of the rows of a task that ran out of time, of the files that have any, by file: one that
    `reported` tests, its tests' own seconds, and the file `running` at the deadline, the one whose test
    started last (`followed.visited`), its own and all of the task's `wall` time no file's tests account for
    (its imports and its collection, and the test the interrupt stopped, which pytest's report gives no
    time), so its row holds at least what the file needs. A file that reported none and was not running never
    started, and has none: absent. `apportioned` split the wall time, the interrupted test's included, over
    the files that finished, in proportion to their tests' times, or evenly where none finished."""
    unaccounted = max(wall - sum(own_seconds.values()), 0.0)
    return {name: own + (unaccounted if name == running else 0.0) for name, own in own_seconds.items()
            if name == running or name in reported}


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str = "on",
         grace: float = session.GRACE, coverage: Path | None = None, whole: bool = False,
         release: str = "", census: bool = False) -> int:
    """Run `tests` and write one verdict row a file, interrupting them past `deadline` seconds, killing
    their process group `grace` seconds after that, and giving it `SETTLE` seconds to be gone.

    A row's seconds are its share of the task's wall time, in proportion to its tests' own times:
    the plan weighs a file by what it costs a container (its import, its collection, the
    container's pace), which pytest's case times leave out. Weighed by case times, tasks planned
    at 300 seconds ran a median of 392. A task that ran out of time times its rows otherwise
    (`deadline_shares`): the file its tests were running in at least what it needs, a file that never
    started none. With `coverage`, pytest runs under coverage.py, whose data file goes there
    (`coverage_items`)."""
    # Before any test writes under tests/, the inputs as the plan saw them; the rows its plan kept read
    # where it found their green runs (`kept_of`), a heavy file's kept groups each its own, and run where
    # their green run is not found there.
    began = time.monotonic()
    common = shared(python)
    names, kept = kept_of(tests)
    whole_files = {name: at for (name, group), at in kept.items() if not group}
    # The census's tests run under check 1's CENSUS (tests/check1_wire.py, where the gate reads it), which
    # also keeps their rows from the records (`tested`).
    environ = {check1_wire().CENSUS: "1"} if census else {}
    code = tested(python, names, split, out, deadline, cache, grace, common, environ, coverage, whole=whole,
                  release=release, kept=whole_files)
    # A map task's red row fails its task: armada runs no judge on draws a map task did not pass, and reads
    # the judge red, naming it (`site_entries`).
    site = session.SITE.fullmatch(split)
    if site and site["start"] is not None and any(found["exitCode"]
                                                  for found in json.loads(out.read_text())["rows"]):
        return told(out, 1)
    groups = [(name, group, at) for (name, group), at in kept.items() if group]
    if not groups:
        return told(out, code)
    rows = json.loads(out.read_text())["rows"]
    for index, (name, group, at) in enumerate(groups, start=1):
        left = deadline - (time.monotonic() - began)
        if left <= 0:
            # The task's deadline spent on the rows before it: this group is neither read nor run, stale as a
            # kept row the task could not run is (`tested`).
            rows.append(stale_row(python, name, group,
                                  f"the task's {deadline:.0f} s spent before it was read"))
            continue
        part = out.with_name(f"{out.name}.{index}")
        code = max(code, tested(python, [name], group, part, left, cache, grace, common, environ, coverage,
                                whole=whole, release=release, kept={name: at}))
        rows += json.loads(part.read_text())["rows"]
    out.write_text(json.dumps({"rows": rows}))
    return told(out, code)


def told(out: Path, code: int) -> int:
    """`code`, the task's verdict `out` said last: each red row, its name and its output's first lines (why
    it is red, its signal and the file it reached: `tested`) and last (what pytest said last). armada keeps a
    task's log tail beside its exit, and a map task's verdict reaches no report but its judge's red row, which
    names that tail (Main, 2026-11-09: a map task "exited 1 after 437 s", its row's "pytest exited -9" kept
    in its output alone)."""
    for found in json.loads(out.read_text())["rows"]:
        if found["exitCode"]:
            lines = (found.get("output") or "").strip().splitlines()
            shown = (lines if len(lines) <= 2 * TOLD_LINES
                     else [*lines[:TOLD_LINES], "...", *lines[-TOLD_LINES:]])
            print("\n".join([f"RED {found['name']}", *(f"  | {line}" for line in shown)]), flush=True)
    return code


TOLD_LINES = 5
"""The first and the last lines of a red row's output a task says (`told`)."""


def site_environment(site: re.Match | None) -> dict[str, str]:
    """What a check-1 site's own task runs its test under (`site`, its group's match of `session.SITE`): a map
    task draws its seeds into its artifacts, and its judge pools every map task's from its inputs, its ask for
    its later map tasks (LATER) none before it runs; nothing for any other group."""
    if site is None:
        return {}
    if site["start"] is None:
        # A judge's ask for its later map tasks (`ask_later`), and its stages (`later_judged`), are its own
        # run's alone.
        later_file().unlink(missing_ok=True)
        stages_file().unlink(missing_ok=True)
        # Check 1 appends to its log where the directory is (`_logged`).
        stages_file().parent.mkdir(parents=True, exist_ok=True)
        wire = check1_wire()
        return {wire.POOL: str(armada_dir("ARMADA_INPUTS", INPUTS)), wire.LATER: str(later_file()),
                wire.STAGES: str(stages_file())}
    wire = check1_wire()
    return {wire.DRAWS: f"{site['start']}:{site['stop']}",
            wire.RECORDS: str(armada_dir("ARMADA_ARTIFACTS", ARTIFACTS))}


def asked(site: re.Match | None) -> dict[str, bool]:
    """A row's ask for more, where its task is a check-1 site's judge that asked for its later map tasks
    (`ask_later`): armada runs them and it judges again (its `more`); nothing else."""
    return {"more": True} if site and site["start"] is None and later_file().is_file() else {}


def later_file() -> Path:
    """Where a judge writes that its judgement needs its later map tasks (LATER, `ask_later`)."""
    return armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / "check1-later.json"


def later_judged(site: re.Match | None) -> dict[str, bool]:
    """A check-1 site judge's row's `LATER_JUDGED`: true where its judgement needed its later map tasks, as
    it asked for them (`asked`) or as a judgement it completed reached check 1's second stage (its stages log,
    STAGES; a weighed check runs only past an undecided second stage), false where every judgement it
    completed decided at its first, absent where it completed none and asked nothing; nothing for any other
    row. The row's record (`record`) keeps it, which the next plan reads (`eager_sites`)."""
    if not site or site["start"] is not None:
        return {}
    logged = [json.loads(line) for line in stages_file().read_text().splitlines() if line.strip()] \
        if stages_file().is_file() else []
    if later_file().is_file() or any(entry.get("stage") == 2 for entry in logged):
        return {LATER_JUDGED: True}
    return {LATER_JUDGED: False} if logged else {}


def stages_file() -> Path:
    """Where a judge's check 1 logs each judgement it completes (STAGES)."""
    return armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / "check1-stages.jsonl"


def reused_rows(python: str, tests: list[str], group: str, common: str, network: bool,
                kept: Mapping[str, tuple[str, str]] | None = None) -> list[dict]:
    """The green rows of `tests` in `group` recorded on these inputs on this host's class (`reused`), those
    its plan kept read where it found them (`kept`), said. They are looked up before any test runs, from one
    reading of the checkout (`Reading`)."""
    kept = kept or {}
    reading = Reading()
    rows = [found for name in tests
            if (found := reused(python, name, group, common, network, kept.get(name), reading))]
    if rows:
        print(f"{len(rows)} files ran green on these inputs on {', '.join(reused_classes(network))}; "
              "their rows are reused")
    return rows


def task_files(out: Path) -> tuple[Path, Path, Path, Path, Path]:
    """The files beside a task's verdict `out` that its pytest writes, none left from an earlier run: the
    recorder's log of the process's whole life (lifetime.py), its children's beside it, read once the group
    has exited (`record`); how collection found each file (executed.py); pytest's report; the files the
    durations record does not wholly name (split.py); and its process id, whose stacks are read at its
    deadline (executed.py, `stalls.dumped`)."""
    log, collection, report, unrecorded, pid = (out.with_name(f"{out.name}.{suffix}") for suffix in (
        "lifetime.log", "collected.json", "junit.xml", "unrecorded.json", "pid"))
    for stale in (report, log, collection, unrecorded, pid):
        stale.unlink(missing_ok=True)
    shutil.rmtree(log.with_name(f"{log.name}.children"), ignore_errors=True)
    return log, collection, report, unrecorded, pid


def ran_pytest(command: list[str], environment: Mapping[str, str], deadline: float, grace: float,
               pid: Path, spy: Path) -> tuple[int, str, bool, str, set[str] | None, set[str]]:
    """A task's pytest `command` run to its `deadline`: its exit code, output, whether it passed the deadline,
    its threads' stacks then (`stalls.dumped`), the data files it opened (`watched`), none where its group
    outlived the kill, and the checkout's files it wrote (`watched.changed`). pytest leads its own process
    group; its output streams through a reader nothing waits on past its end, so a child holding the pipe
    open cannot keep the verdict from being written."""
    lines: list[str] = []
    # Every data file the task's processes open, held from before pytest starts until its group stopped
    # (`watched`): each is the input of every file the task runs; and every file of the checkout as it stood
    # then, so a file the run wrote is told from one it read alone (`watched.changed`, `record`).
    before = watched.stamps(os.getcwd(), followed.HIDDEN_INPUTS)
    opens = watched.Opens(os.getcwd(), frozenset({str(session.DURATIONS)}))
    done = child_groups.start(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env=dict(environment))

    def read() -> None:
        for line in done.stdout or ():
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    expired, dump, spent = False, "", 0.0
    try:
        done.wait(deadline)
    except subprocess.TimeoutExpired:
        # pytest as it stands at its deadline, before the interrupt below: its threads' activity, every
        # thread's stack and the host it stood on (`stalls.stood`), read within its grace, so a stalled
        # task's whole stop stays inside armada's own timeout (`SETTLE`'s sum).
        reading = time.monotonic()
        expired, dump = True, stalls.stood(pid, spy, host.state)
        spent = time.monotonic() - reading
    # Past the deadline ^C first, so pytest reports what ran; a test stuck in C ignores it. Then the group is
    # killed, what pytest's tests left running with it, and given `SETTLE` to be gone.
    if left := child_groups.stop(done, grace=max(grace - spent, 0.0) if expired else 0.0,
                                 interrupt=signal.SIGINT, settle=SETTLE):
        print(f"processes {left} of the task's group still ran {SETTLE:.0f} s after SIGKILL", flush=True)
    # A process of the group still running may yet open a file: no record of this task keeps a key.
    closed = opens.closed()
    reader.join(10)
    wrote = watched.changed(before, watched.stamps(os.getcwd(), followed.HIDDEN_INPUTS))
    return done.returncode, "".join(lines), expired, dump, None if left else closed, wrote


def stall_evidence(python: str, tests: list[str], group: str, dump: str, began: float) -> str:
    """A run past its deadline kept whole among the task's artifacts (`stalls.kept`), its path there: its red
    rows' evidence (armada's `artifacts`), named by the rows it grades, so a task's several runs (a GPU
    lane's) each keep their own. `began` is the run's monotonic start."""
    root = armada_dir("ARMADA_ARTIFACTS", ARTIFACTS)
    graded = hashlib.sha256("\n".join(row(python, name, group) for name in tests).encode()).hexdigest()[:16]
    kept = stalls.kept(root, f"{python}-{graded}", dump, time.time() - (time.monotonic() - began))
    return str(kept.relative_to(root))


def tested(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str, grace: float,
           common: str, environ: Mapping[str, str], coverage: Path | None = None,
           network: bool = False, whole: bool = False, release: str = "",
           kept: Mapping[str, tuple[str, str]] | None = None, recorded: bool = True,
           variant: str = "") -> int:
    """`task` on the inputs `common` hashes, its pytest under `environ` beside this process's, a green row
    reused as the cache's policy (`policy`) allows. With `network`, the network-marked tests alone, with the
    Hub open, recorded on the network lane's class (`device_class`); `release`, a release check's. A check-1
    site's map task (`session.SITE`) is never reused: its draws, among its artifacts, are what its judge
    reads. A run that is no test of its files (a GPU lane's prepare, `recorded` false) records no row. A
    process of the files under other flags than the lane's (`variant`, a GPU lane's `VARIANTS`) runs the
    same cut of them, its rows and their records named apart (`varied`)."""
    began = time.monotonic()
    group = varied(split or (f"{RELEASE}{release}" if release else ""), variant)
    site = session.SITE.fullmatch(group)
    rows = ([] if cache == "off" or release or (site is not None and site["start"] is not None)
            else reused_rows(python, tests, group, common, network, kept))
    tests = [name for name in tests if row(python, name, group) not in {found["name"] for found in rows}]
    if kept:
        tests, stale_rows = afforded(python, tests, group, kept, deadline - (time.monotonic() - began))
        rows += stale_rows
    if not tests:
        out.write_text(json.dumps({"rows": rows}))
        return 0
    log, collection, report, unrecorded, pid = task_files(out)
    # coverage.py follows the Python processes the tests start (pyproject.toml's patch), each writing
    # its data where COVERAGE_FILE says.
    measured = [] if coverage is None else ["coverage", "run"]
    environ = environ if coverage is None else {**environ, "COVERAGE_FILE": str(coverage / ".coverage")}
    run = session.selected(session.pytest_run(python, report, *measured), network, whole, bool(release),
                   site is not None)
    # A resuming task keeps its progress and sets aside the tests its recalled attempt passed (`resumed`,
    # SET_ASIDE); a site's map task (its artifacts its judge's draws), a release and a trace resume none, nor
    # does a run that records none (a prepare's: a test it prepared has not passed, and a test an earlier
    # attempt passed still prepares into this one's store).
    resumes = recorded and site is None and not release and coverage is None
    done_before = resumed(tests) if resumes else {}
    aside = out.with_name(f"{out.name}.set-aside.json")
    aside.write_text(json.dumps(sorted(done_before)))
    command = [*session.pytest_command(run, tests, split, release), *(["-p", "progress"] if resumes else [])]
    environment = session.pytest_environment({**environ, **site_environment(site),
                                              "DEW_EXECUTED_LOG": str(log), "DEW_COLLECTED": str(collection),
                                              "DEW_UNRECORDED": str(unrecorded), "DEW_PYTEST_PID": str(pid),
                                              "DEW_STACKS_READER": str(os.getpid()),
                                              **({progress.SET_ASIDE: str(aside)} if resumes else {})})
    if network:
        environment = {key: value for key, value in environment.items() if key not in session.CLOSED_HUB}
    # The task environment's py-spy (the test extra), which reads a stalled pytest's stacks (`stalls.dumped`).
    returncode, output, expired, dump, opened, wrote = ran_pytest(command, environment, deadline, grace, pid,
                                                                  Path(f".venv-{python}/bin/py-spy"))
    stall = stall_evidence(python, tests, group, dump, began) if expired else None
    cases = list(ET.parse(report).getroot().iter("testcase")) if report.is_file() else []
    # How collection found each file: a file with no row is green only where every test it holds was set aside
    # by the selection, as pytest reported it (executed.py); a run that wrote no report greens nothing.
    counts = json.loads(collection.read_text()) if collection.is_file() else {}
    ran = []
    shares = task_seconds(tests, cases, log, time.monotonic() - began, expired)
    # Where the task ran out of time, the file whose test started last is the one it was spent in (executed.py
    # logs each visit): its row is the deadline's red; a file pytest reported is graded by its own cases, the
    # exit the interrupt's, not its; and one with none is not graded but run again (`grading.STALE`), never
    # a red of its tests (Main, 2026-11-09: train 745 filed six files packed beside a hung one as reds
    # "at 0.0 s").
    # A task stalled before any test started (its collection) has no such file: each of its rows is the
    # deadline's.
    stood = followed.visited(followed.log_events(log)) if expired or not report.is_file() else []
    running = stood[-1] if stood else None
    for name in tests:
        own = [case for case in cases if grading.file_of(case) == name]
        red, failed, cause = grading.graded(name, own, cases, counts.get(name, {}), report.is_file(),
                                            returncode, deadline if expired else None, running, pid.is_file(),
                                            running)
        # A file that never started in a task that ran out of time has no seconds (`deadline_shares`): its
        # row is `unobserved`, and armada weighs no plan by it.
        unobserved = name not in shares
        seconds = shares.get(name, 0.0)
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart; a check-1 site's judge apart from its file, its pooling no part of the file's tasks.
        ran.append({"name": row(python, name, group), "exitCode": 1 if red else 0, "seconds": seconds,
                     # A stall's stacks whole, then the end of what pytest wrote; a red's stall's whole record
                     # among its evidence (`stalls.kept`).
                     "output": "\n".join(red + ([dump[:12000]] if dump else [])
                                         + ([output[-12000:]] if red else [])),
                     **({"artifacts": [stall]} if stall is not None and red else {}),
                     # Its first failed test's exception as pytest gave it ("RuntimeError: ...").
                     **({"first_failure": failed[0][2].get("message") or ""} if failed else {}),
                     # Where the task ran out of time, a file that never started: no seconds to time it by.
                     **({"unobserved": True} if unobserved else {}),
                     # A pytest a signal ended, its file's red whole: the signal, for whoever it routes to.
                     **({"signal": -returncode} if red and returncode < 0 and not expired else {}),
                     "tests": {**{node: seconds for node, seconds in done_before.items()
                                  if node.split("::", 1)[0] == name},
                               **{grading.node_of(case): float(case.get("time") or 0) for case in own}},
                     # Its whole_suite cells, every one collected, run or set aside (executed.py).
                     **{key: cells for key, cells in counts.get(name, {}).items() if key in MARKED},
                     "ynn_switched": ynn_switched(own), "cells": flow_cells(own),
                     # A judge's, beside its pooling, what its map tasks' records say they paid
                     # (`site_costs`).
                     "timings": {row(python, name, group if site else ""): seconds,
                                 **(site_costs(python, name, site["test"]) if site and site["start"] is None
                                    else {})}, **cause, **asked(site),
                     **later_judged(site)})
    out.write_text(json.dumps({"rows": rows + ran}))
    if (recorded and coverage is None and not release and not (environ and check1_wire().CENSUS in environ)
            and not done_before):
        # A traced run's rows, timed under tracing, are no rows to reuse, nor a release check's, which runs
        # under its requirement's overlay and on the whole suite's run alone, nor the census's, nor a resumed
        # one's, whose earlier attempt's opens no record holds, nor a prepare's, which ran no test to its end.
        record(python, ran, log, common, network, opened,
               set(json.loads(unrecorded.read_text())) if unrecorded.is_file() else set(), wrote)
    return 0


def task_seconds(tests: list[str], cases: list[ET.Element], log: Path, wall: float,
                 expired: bool) -> dict[str, float]:
    """Each of `tests`' seconds of its task's `wall` time, by its JUnit `cases`: in proportion to its tests'
    own times (`apportioned`); where the task ran out of time (`expired`), the file running at its deadline
    (its executed `log`'s last visit) at least what it needs, and one that never started none
    (`deadline_shares`)."""
    own_seconds = {name: sum(float(case.get("time") or 0) for case in cases if grading.file_of(case) == name)
                   for name in tests}
    if not expired:
        return apportioned(own_seconds, wall, len(tests))
    # The file whose test started last is the one running at the deadline (executed.py logs each visit).
    stood = followed.visited(followed.log_events(log))
    return deadline_shares(own_seconds, {grading.file_of(case) for case in cases}, wall,
                           stood[-1] if stood else None)


def resumed(tests: list[str]) -> dict[str, float]:
    """The tests of `tests` an earlier attempt of this task passed before the fleet recalled it (armada keeps
    a recalled attempt's artifacts and hands them to the next as ARMADA_RESUME, #4544), each with its seconds,
    as tools/armada/progress.py kept them; copied into this attempt's progress, so a second recall keeps them
    too. None outside a resumed task."""
    kept = Path(os.environ.get("ARMADA_RESUME", "/nonexistent")) / progress.PROGRESS
    if not kept.is_file():
        return {}
    passed = {}
    for line in kept.read_text().splitlines():
        found = json.loads(line)
        if found["node"].split("::", 1)[0] in tests:
            passed[found["node"]] = float(found["seconds"])
    artifacts = os.environ.get("ARMADA_ARTIFACTS")
    if artifacts and passed:
        Path(artifacts).mkdir(parents=True, exist_ok=True)
        with (Path(artifacts) / progress.PROGRESS).open("a") as carried:
            carried.writelines(json.dumps({"node": node, "seconds": seconds}) + "\n"
                               for node, seconds in passed.items())
    if passed:
        print(f"resumed: {len(passed)} tests an earlier attempt passed set aside", flush=True)
    return passed


def native_inputs(opened: set[str], recorded: set[str]) -> set[str]:
    """The fixture directories, each up to tests/fixtures, of what a task's processes opened natively
    (`watched`: an open no Python process recorded, as a tokenizer's Rust code makes): a native reader's
    probe beside what it opened is no record's, as followed.py's `PROBES` are Python's. The fetched store
    is none: its payloads are content-addressed, and a planner, which never downloads them, keys them as a
    task does (`digests.contents`)."""
    store = digests.FIXTURES / "fetched"
    return {str(parent) for name in opened - recorded if digests.FIXTURES in (path := Path(name)).parents
            and store != path and store not in path.parents
            for parent in path.parents if parent == digests.FIXTURES or digests.FIXTURES in parent.parents}


def wrote_inputs(depends: list[str], changed: Container[str]) -> bool:
    """Whether a run's inputs `depends` (followed.py's records) hold a file `changed` (`watched.changed`:
    written, removed or made) names: a file read, probed or run as a module, or the entries an import's lookup
    finds of its name (`followed.LOOKUP`, `digests.looked_up`). A directory's listing is weighed by none: a
    file made in it is new to every checkout a plan reads, so no plan finds the key it leaves."""
    for item in depends:
        if item.startswith(followed.LOOKUP):
            looked = item[len(followed.LOOKUP):]
            if any(path == looked or path.startswith((f"{looked}.", f"{looked}/__init__"))
                   for path in changed):
                return True
        elif item.removeprefix(followed.PROBED).partition(followed.CODE)[0] in changed:
            return True
    return False


def site_inputs(python: str, row_name: str, depends: list[str] | None) -> list[str] | None:
    """What row `row_name` of a check-1 site's task ran on (`session.SITE`), given what its own run executed
    and read (`depends`, None where that is not known): a map task's own, kept among its artifacts for its
    judge, which reads every one of its map tasks' among its inputs and keys on them with its own
    (`cache_key`), so the site's key covers what its draws ran, which the judge runs none of (an arm run as
    given, a branch of a later seed): its first stage's map tasks', and its later ones' where they ran; None
    where one of those is not known or not found. Any other row's, as it is."""
    _, name, group = parsed_row(row_name)
    site = session.SITE.fullmatch(group)
    if site is None:
        return depends
    test = site["test"]
    if site["start"] is not None:
        span = (int(site["start"]), int(site["stop"]))
        named = f"{site_task(python, name, test, span)}.inputs.json"
        kept = armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / named
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(depends))
        return depends
    # Its map tasks' inputs by the spans they drew, whatever cut the plan made of its units (`cut_draws`).
    prefix, suffix = f"{site_task(python, name, test)}-draws-", ".inputs.json"
    found = {tuple(int(end) for end in path.name[len(prefix):-len(suffix)].split("-")):
             json.loads(path.read_text())
             for path in armada_dir("ARMADA_INPUTS", INPUTS).rglob(f"{prefix}*{suffix}")}
    mark = sites(name).get(test)
    # A judge's later map tasks ran only where its first stage asked for them (`site_entries`), all of them:
    # one that judged without them judged over its first stage's alone. Each stage it judged over is its
    # units whole, so a map task missing from it is not known.
    first = sorted(span for span in found if mark and span[0] < mark.timed)
    later = sorted(span for span in found if mark and span[0] >= mark.timed)
    if (depends is None or mark is None or not drawn_whole(first, 0, mark.timed)
            or (later and not drawn_whole(later, mark.timed, mark.units + mark.weighed))
            or any(found[span] is None for span in (*first, *later))):
        return None
    return sorted({*depends, *(path for span in (*first, *later) for path in found[span])})


def drawn_whole(spans: list[tuple[int, int]], start: int, stop: int) -> bool:
    """Whether sorted `spans` draw units `start` to `stop` each once, end to end."""
    at = start
    for begin, end in spans:
        if begin != at:
            return False
        at = end
    return at == stop and bool(spans)


def record(python: str, rows: list[dict], log: Path, common: str, network: bool = False,
           opened: set[str] | None = None, unkeyed: Container[str] = (),
           changed: Container[str] = ()) -> None:
    """Keep each row a task ran with the files its tests executed and read, every data file the task's
    processes opened (`opened`, the task's watch: None where it was blind, and no row keeps a key), where
    they read the backend and its peak memory on a GPU and its children's pools: `gpu_tier` reads the
    reads, `gpu_needs` the memory. A green row keeps its key too, which `reused` reads it back by; a red
    row has none, so a later run never reuses an earlier green row of its file; nor has one that read a file
    the task's processes wrote (`changed`, `watched.changed`), whose key would hash the bytes they left, not
    those it read (row 2559). A check-1 site's judge keeps its map tasks' inputs with its own
    (`site_inputs`)."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return
    # Each file's record in the process's log (followed.py's `recorded`): none for a file it never reached.
    executed = followed.recorded(log)
    held = device_class(network)
    if opened is not None:
        recorded = {read for ran in executed.values() for read in ran["depends"] or ()}
        opened = opened | native_inputs(opened, recorded)
    for row_ in rows:
        name = parsed_row(row_["name"])[1]
        ran = executed.get(name, {"depends": None, "reads": [], "peak": None, "shares": 0, "pools": 0})
        if ran["depends"] is not None:
            inputs = None if opened is None else sorted({*ran["depends"], *opened})
            ran = {**ran, "depends": None if inputs is None or wrote_inputs(inputs, changed) else inputs}
        ran = {**ran, "depends": site_inputs(python, row_["name"], ran["depends"])}
        if row_["exitCode"] == 0 and name not in executed:
            continue
        green = row_["exitCode"] == 0 and ran["depends"] is not None
        # A group of a file the durations record does not wholly name (`unkeyed`, split.py) keeps no key: the
        # plan, which cuts by the record alone, may cut the file otherwise (`session.cut`).
        key = (cache_key(common, name, parsed_row(row_["name"])[2], ran["depends"])
               if green and name not in unkeyed else None)
        body = json.dumps({"key": key,
                           "depends": ran["depends"], "row": row_, "reads": ran["reads"],
                           "peak": ran["peak"], "shares": ran["shares"], "pools": ran["pools"]}).encode()
        # The record under these inputs, which `reused` reads, and the last run's, which the GPU tier and
        # lane read, only once the first is written, so it is never newer.
        if written(record_key(python, row_["name"], held, common), token, body):
            written(record_key(python, row_["name"], held), token, body)
            # The run of exactly these inputs, kept by their key (`green_key`): a green one, or a red one,
            # which replaces a green row of the same inputs so no later run reuses it (a flaky red).
            ran_on = key if key is not None else None if ran["depends"] is None or green else cache_key(
                common, name, parsed_row(row_["name"])[2], ran["depends"])
            if ran_on is not None:
                written(green_key(python, row_["name"], held, ran_on), token, body)


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
    # One reading of the checkout, which every record's key reads its inputs from.
    reading = Reading()

    def evidence_of(key: str) -> tuple[str, list[str]]:
        _, name, _ = key_parts(key)
        found = recorded(f"{prefix}{key}", token)
        if found is None:
            return name, ["unrecorded"]
        current = any(fresh(found, name, common, reading) for common in inputs)
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


def collected(python: str, marks: str, files: list[str] | None = None) -> dict[str, int]:
    """Each test file of this checkout (of `files`, where given) with tests pytest's `-m marks` selects, and
    how many, as pytest collects it on the CPU."""
    every = sorted(str(path) for path in Path("tests").glob("test_*.py")
                   if str(path) not in SKIPPED) if files is None else files
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


def borrowed_records(names: list[str], python: str, held: str, token: str,
                     strict: bool = False) -> dict[str, dict]:
    """For each of `names`, test files no run on class `held` measured, the last record of the whole
    file on another class whose run measured its GPU peak, the one that holds the most
    (`gpu_needs`): what a file has taken on a GPU, which a first lane on a class takes as its share.
    One no class measured is absent, as every one is where the Worker does not list its records. Its run on
    `held` records its own, which every later lane reads. With `strict`, a read the Worker does not answer is
    an `r2.R2Error`, never a file taken for unmeasured: what sizes a job before it runs (gpu_holds.py) is read
    whole or not at all (2026-11-11: one record's read did not come back as a GPU tier was composed, its part
    took the default for a file measured at 5.9 GiB, and the file ran out of a 4 GiB lane)."""
    if not names:
        return {}
    prefix = f"{CACHE_PREFIX}/{python}/"
    wanted = set(names)
    try:
        every = listed(prefix, token)
    # The Worker did not list its records: borrowing only packs a lane, so each such file runs alone with the
    # lane's whole share, as one measured on no class does.
    except r2.R2Error as error:
        if strict:
            raise
        print(f"the test cache did not list {prefix}: {error}; files unmeasured here run alone",
              file=sys.stderr)
        return {}
    keys = [key for key in every
            if (parts := key_parts(key))[0] != held and parts[2] == "whole" and parts[1] in wanted]

    def read(key: str) -> tuple[str, dict | None]:
        if strict:
            found = store(token).get(f"{prefix}{key}")
            return key_parts(key)[1], None if found is None else json.loads(found)
        return key_parts(key)[1], recorded(f"{prefix}{key}", token)

    def holds(found: dict) -> float:
        return gpu_needs(found, float("inf"))[1]

    borrowed: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for name, found in pool.map(read, keys):
            if found is None or found.get("depends") is None or found.get("peak") is None:
                continue
            if name not in borrowed or holds(found) > holds(borrowed[name]):
                borrowed[name] = found
    return borrowed


VARIANTS = {"users": ("users_process", {"DEW_TEST_USERS_ARITHMETIC": "1"})}
"""A GPU lane's processes of a file beside its own, by name: the mark of the tests one runs and the
environment it runs under. XLA reads a process's flags once, so tests that must open under a user's flags
(no deterministic ops, JAX's own matmul precision) are a process of their own: under its word the suite's
conftest runs the marked GPU tests and skips the others, as without it it skips the marked ones, so both
processes select and cut a file alike and each test runs in one."""
def lane_units(python: str, tests: list[str]) -> dict[str, tuple[str, str, str]]:
    """Each process a GPU lane runs for `tests`, by its name, as its file, its group and its variant: each
    test's own, and after it one for each of `VARIANTS` whose marked tests its file holds, as pytest collects
    the lane's files on the CPU, once a mark. A file with none of a variant's tests starts no process for it;
    one that holds some never goes without it (a variant no lane ran is tests that ran nowhere, silently)."""
    files = sorted({name.partition("#")[0] for name in tests})
    holding = {variant: collected(python, mark, files) for variant, (mark, _) in VARIANTS.items()}
    units = {}
    for name in tests:
        file, _, group = name.partition("#")
        units[name] = (file, group, "")
        units |= {f"{file}#{varied(group, variant)}": (file, group, variant)
                  for variant, files_ in holding.items() if file in files_}
    return units


PREPARES = "prepares"
"""The mark of a GPU test whose part's prepare runs it first, on the node's CPU (`gpu_prepared`)."""
PREPARING = {"DEW_TIER_PREPARE": "1", "JAX_PLATFORMS": "cpu"}
"""What a lane's prepare runs its files under: the word tests/conftest.py reads to skip every test but those
marked `PREPARES`, each of which then prepares rather than runs (tests/model_flows.py), on the CPU alone."""


def gpu_prepared(python: str, tests: list[str], out: Path, *, processes: int, deadline: float) -> int:
    """A GPU lane's prepare, on its node's CPU before its files' GPU run (`gpu_lane`) and with no GPU drawn:
    each of `tests` that holds a test marked `PREPARES` run once in a process of its own under `PREPARING`,
    at most `processes` at once, as its run will select and cut it (a group `<file>#G/N` holds in its prepare
    the tests it holds in its run: the same selection, the same weights, DEW_SPLIT_DURATIONS), so each such
    test computes what its run reads and must not compute while it holds the GPU, its float64 truths and
    its programs compiled for the GPU the node's topology names (DEW_NODE_TOPOLOGY), into the store its
    environment names (DEW_TIER_TRUTHS), which the run is given too. Their rows are written to `out` and
    nothing is recorded, no test having run to its end: a prepare that fails is said in its row, and its run
    is red for what it lacks, never computing it on the GPU. A file with nothing to prepare starts no
    process; a file's process under other flags (`lane_units`) prepares apart, under them, as it runs. First
    the pinned fixtures the tests read offline are fetched, as the run fetches them
    (tools/fetched_fixtures.py), so the run finds each whole and waits on no download with the GPU held."""
    subprocess.run([f".venv-{python}/bin/python", "tools/fetched_fixtures.py", "fetch"], check=True)
    units = lane_units(python, tests)
    holding = collected(python, PREPARES, sorted({file for file, _, _ in units.values()}))
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(units)
                if units[name][0] in holding}
    threads = str(max(1, len(os.sched_getaffinity(0)) // processes))
    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        for prepared in [pool.submit(tested, python, [units[name][0]], units[name][1], verdict, deadline,
                                     "off", session.GRACE, "",
                                     {**PREPARING, "OMP_NUM_THREADS": threads,
                                      **(VARIANTS[units[name][2]][1] if units[name][2] else {})},
                                     recorded=False, variant=units[name][2])
                         for name, verdict in verdicts.items()]:
            prepared.result()
    out.write_text(json.dumps({"rows": [row for verdict in verdicts.values()
                                        for row in json.loads(verdict.read_text())["rows"]]}))
    return 0


def gpu_lane(python: str, tests: list[str], out: Path, *, cache: str, processes: int,
             deadline: float, network: bool = False) -> int:
    """Run `tests` on this process's share of the one GPU it sees, each once, as a task of its own
    (`tested`), and write their verdict rows to `out`: a test file, or one group of one as CI names it
    (`<file>#G/N`, the Gth of N runs of the tests the file's collection here finds, split.py), whose row is
    that group's, so every test of a file cut into groups runs once, whatever an earlier run of it held
    (audit, 2026-11-08: the GPU tier's parts named tests an earlier tier ran, so a new one never ran, and a
    test named apart read "no evidence ran"). With `network`, the network lane: each file's
    network-marked tests alone, which every other task deselects (`session.SELECTION`), run with the Hub
    open on a node that reaches it, their records on a class of their own (`device_class`), so the test
    cache reruns only those whose inputs changed (the maintainer, 2026-10-25: no CI run reached the published
    checkpoints' parity tests).

    A file's process is given the share its last run on this GPU's class measured, green or red,
    and files start heaviest first while what they hold with their children (`gpu_needs`) fits
    the share this process was given (`XLA_PYTHON_CLIENT_MEM_FRACTION`, as gpu-run sets it), at
    most `processes` at once. A file that runs out of a measured share is red: it took more than
    its last run, whose peak its next run is given. A file no run on this class measured borrows its
    share from the largest any other GPU class's last run measured (`borrowed`), so a first lane on
    a class packs its files as a measured one does; one red under it is red, run once, as every red is
    (owner, 2026-11-07: a red under sharing is a red), and its measured peak is the next lane's. A file
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
    # The memory the GPU gives a process: all of it less what its driver and firmware keep, the total XLA
    # takes a pool's fraction of (within nvidia-smi's rounding to a MiB), and the one a node's session gave
    # this lane its fraction of. Counted of the GPU's whole memory, a file's share was a pool short of it by
    # what the driver keeps (2026-11-11: a GH200's 1,103 of 97,871 MiB made a 5.91 GiB share a 5.84 GiB pool,
    # which ran out).
    [total] = ((int(whole) - int(kept)) * 2 ** 20 for whole, kept in
               (answer.split(",") for answer in gpu_query("memory.total,memory.reserved")))
    # A job holding the whole GPU is given no fraction (a node sets one for a part of a GPU alone, dew-ops'
    # session `settings`): its processes may take what one process takes by JAX's own default (row 2570).
    budget = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", WHOLE_GPU)) * total
    held = device_class(network)
    threads = str(max(1, len(os.sched_getaffinity(0)) // processes))

    # Each process the lane runs (`lane_units`): a test's file and group (`<file>#G/N`; none, a whole
    # file), and its file's process under other flags, where it holds tests for one.
    units = lane_units(python, tests)
    tests = list(units)

    # A group the tier cut by its own seconds holds tests another cut's group of its name did not: its
    # file's whole measured run, which holds any group of it, is its measure (2026-11-10: a group measured
    # holding two tests was given their pool when it held five, one asking more than the pool).
    cut = bool(os.environ.get("DEW_SPLIT_DURATIONS"))

    def needs(name: str) -> dict | None:
        file, group, variant = units[name]
        return recorded(record_key(python, row(python, file, varied("" if cut else group, variant)), held),
                        token)

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        records = dict(zip(tests, pool.map(needs, tests), strict=True))
    # A group no run on this class measured borrows its whole file's measure, which holds at least its own;
    # a file's process under other flags, its tests a part of the file's, the measure of the lane's own first.
    borrowed = borrowed_records(sorted({units[name][0] for name, found in records.items() if found is None}),
                                python, held, token)
    measures = {name: records[name] or records.get(f"{file}#{group}" if group else file)
                or borrowed.get(file) for name, (file, group, _) in units.items()}
    needed = {name: gpu_needs(measures[name], budget) for name in tests}
    shares = {name: share for name, (share, _) in needed.items()}
    holds = {name: held for name, (_, held) in needed.items()}
    # A file measured holding more than the lane's share is given all of it, and its row says so: what it
    # was admitted at is the lane's reservation, not its need (2026-11-10: a 5.9 GiB need given a 4 GiB lane
    # without a word, and its run out of memory read as the test's).
    short = {name: f"admitted at {budget / 2 ** 30:.1f} GiB, below its measured need of "
                   f"{need / 2 ** 30:.1f} GiB"
             for name in tests
             if math.isfinite(need := gpu_needs(measures[name], math.inf)[1]) and need > budget}
    # Before any test writes under tests/, the inputs every file's key hashes.
    common = lane_inputs(python)
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(tests)}

    waiting = sorted(tests, key=lambda name: -holds[name])
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
                file, group, variant = units[fits]
                # A group cut by the tier's own GPU seconds (DEW_SPLIT_DURATIONS, split.py) is no group
                # any record's key holds: it reads no cached row, as it writes none, so no earlier cut's row
                # stands for it beside groups cut afresh (review of b3bcbe0fe).
                policy = "off" if group and os.environ.get("DEW_SPLIT_DURATIONS") else cache
                running[pool.submit(tested, python, [file], group, verdicts[fits], deadline, policy,
                                    session.GRACE, common,
                                    {**environ, **(VARIANTS[variant][1] if variant else {})}, None, network,
                                    variant=variant)] = fits
                continue
            done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                del running[future]
    rows = [{**verdict, "output": f"{short[name]}\n{verdict.get('output', '')}"} if name in short else verdict
            for name in tests for verdict in json.loads(verdicts[name].read_text())["rows"]]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def folded(record: dict[str, float], verdict: dict, target: float) -> dict[str, float]:
    """The durations `record` with main's CI `verdict` (`armada verdict <sha> --json`) folded in, one test
    file at a time, each whole, as its floor Python's rows ran it. A file whose rows are all green and
    complete (its whole row, or every group of its cut) takes the times they observed where its record is
    stale (`stale`); a file with a red or missing row keeps its entries, as does a file the verdict did
    not run, and one the checkout has changed since the verdict's commit, whose rows observed tests it may
    no longer hold (2026-10-31: train 370's fold replaced a head's 66 renamed test ids with main's old
    ones); a file the checkout no longer has loses them. A file a task's deadline cut off, run whole,
    takes the times its row observed of the tests it finished (all but the last, which the deadline
    stopped) where its record lacks them: no green run would ever time it, and the plan cuts it by them, a
    test it never reached weighing their mean (2026-11-02: base/kernels' test_algebra_laws, recorded
    nowhere, ran 46 of its ~70 tests and was cut off every run). A reused row (the test cache) carries the
    times of the run that recorded it. Each train folds main's CI so (the land queue), and nothing else
    writes the record. A site its file's marks shard (`sites`) keeps its entries: its judge's row times
    its pooling, no measure of its draws, so its map tasks are weighed by armada's measures of them, never
    by the record (`site_entries`), and a judge armada did not run is a row of its task's name. A
    whole_suite cell a row set aside (its `whole_suite` names it, its `tests` do not: a train's batch)
    keeps its entry too."""
    changed = set(subprocess.run(["git", "diff", "--name-only", verdict["sha"], "--", "tests"], check=True,
                                 capture_output=True, text=True).stdout.split())
    rows: dict[str, list[dict]] = {}
    for row_ in verdict["rows"]:
        if ":" not in row_["name"]:
            continue
        python, name, split_ = parsed_row(row_["name"])
        if (python == PYTHONS[0] and split_ != COLLECT and not session.SITE.fullmatch(split_)
                and VARIED not in split_ and Path(name).is_file() and name not in changed):
            rows.setdefault(name, []).append(row_)
    kept = {node: seconds for node, seconds in record.items() if Path(node.split("::")[0]).is_file()}
    for name, held in rows.items():
        splits = {parsed_row(row_["name"])[2] for row_ in held}
        counts = {int(split_.split("/")[1]) for split_ in splits if split_}
        complete = splits == {""} or (len(counts) == 1 and splits == {f"{group}/{count}" for count in counts
                                                                      for group in range(1, count + 1)})
        if splits == {""} and held[0].get("kind") == "deadline":
            # Its last test, the one the deadline stopped, is timed only as far as it ran.
            finished = list(held[0].get("tests", {}).items())[:-1]
            kept.update({node: round(seconds, 3) for node, seconds in finished if node not in kept})
            continue
        if not complete or any(row_["exitCode"] != 0 or "tests" not in row_ for row_ in held):
            continue
        observed = {node: seconds for row_ in held for node, seconds in row_["tests"].items()}
        marked = sites(name)
        # Its whole_suite cells its rows set aside (collected, not run: a train's batch) keep their times,
        # which only the whole suite's run observes (row 4377: a train's fold dropped a 1108 s cell the
        # whole run then packed cold).
        aside = {node for row_ in held for node in row_.get("whole_suite", [])} - observed.keys()
        own = {node: seconds for node, seconds in kept.items() if node.split("::")[0] == name
               and session.site_name(node, marked) not in marked and node not in aside}
        if stale(own, observed, sum(row_["seconds"] for row_ in held), max(counts, default=1), target):
            kept = {node: seconds for node, seconds in kept.items() if node not in own}
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
    recorded = session.unit_seconds(nodes, [own[node] for node in nodes])
    ran = session.unit_seconds(nodes, [observed[node] for node in nodes])
    setup = max(seconds - sum(ran), 0.0) / cut

    def heaviest(weights: list[float]) -> float:
        count = len(split(weights, seconds, target, cut))
        return max(sum(ran[run.start:run.stop]) + setup for run in session.groups(weights, count) if run)

    return heaviest(recorded) > KEPT * target and heaviest(ran) < heaviest(recorded)


ARTIFACTS = "/armada/task/artifacts"
"""Where an armada task's files are kept for `armada map --artifacts`."""
COVERAGE_DEADLINE = 5400.0
"""A coverage task's deadline. Branch tracing slows a task several times over, and a task cut short leaves
its files' coverage partial, which `coverage_report.report` refuses."""


def coverage_items(target: float, timings: dict) -> list[str]:
    """The plan's tasks for the oldest Python, as armada map items: each runs its files under coverage.py
    with the test cache off, its rows and data among its artifacts, by `COVERAGE_DEADLINE`. A check-1 site's
    map tasks run its code over every seed; its judge, which pools their draws, has none to pool here."""
    return [f"mkdir -p {ARTIFACTS} && python3 tools/armada/ci.py task --python={entry['python']} "
            f"--tests='{entry['tests']}' --split='{entry['split']}' --out={ARTIFACTS}/rows.json --cache=off "
            f"--deadline={COVERAGE_DEADLINE:.0f} --coverage={ARTIFACTS}"
            for entry in plan(target, timings)
            if entry["python"] == PYTHONS[0] and entry["split"] != COLLECT and "after" not in entry]


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
    marked whole_suite too. A check-1 site's map task past the target is reported whatever the record holds
    (its test is recorded whole): its mark gives a task too many units, which fewer a task (per_task,
    conditioning_per_task) bring within it (row 3729: one ran 1,366 s unreported)."""
    found = []
    for row_ in verdict["rows"]:
        cells = set(row_.get("whole_suite", []))
        drawn = session.SITE.fullmatch(parsed_row(row_["name"])[2]) if "name" in row_ else None
        mapped = drawn is not None and drawn["start"] is not None
        for test, seconds in (row_.get("tests") or {}).items():
            if seconds <= target or (base is not None and test in base and not mapped):
                continue
            found.append({"test": test, "file": test.partition("::")[0], "seconds": round(seconds, 1),
                          "target": target, **({"whole_suite": True} if test in cells else {}),
                          **({"map_task": row_["name"], "fix": "fewer units a task (per_task, "
                              "conditioning_per_task)"} if mapped else {})})
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


def releasing(args: argparse.Namespace) -> int:
    """`release-plan`'s tasks printed, or `release`'s checks run (`release_plan`, `release`)."""
    if args.operation == "release-plan":
        print(json.dumps(release_plan(json.loads(args.verdict.read_text()))))
        return 0
    return release(args.python, args.tests.split(), args.requirement, args.out)


def lane_asked(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """`gpu-lane` as its arguments ask: a lane's prepare, the network lane, or a CUDA lane over its files."""
    if args.tests is None and not args.network:
        parser.error("gpu-lane names its --tests; only the network lane (--network) takes every file "
                     "holding a network-marked test by default")
    if args.prepare:
        if args.network or args.cache is not None:
            parser.error("gpu-lane --prepare names its --tests, and takes neither --network nor --cache")
        return gpu_prepared(PYTHONS[0], args.tests.split(), args.out, processes=args.processes,
                            deadline=args.deadline)
    if args.cache is None:
        parser.error("gpu-lane names the test cache's policy (--cache)")
    tests = args.tests.split() if args.tests is not None else list(network_files(PYTHONS[0]))
    return gpu_lane(PYTHONS[0], tests, args.out, cache=args.cache, processes=args.processes,
                    deadline=args.deadline, network=args.network)


def main() -> int:
    parser = command_line.parser(PYTHONS, session.DEADLINE)
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
        # A GPU runs its marked files alone, a batch's and a day's alike (ledger 2452); --evidence prints that
        # tier, each file with its records' backend reads beside its marked tests.
        tier = marked(PYTHONS[0])
        if args.evidence:
            reads = gpu_tier(PYTHONS[0])
            for name, count in tier.items():
                evidence = [item for item in reads.get(name, []) if not item.startswith("mark:")]
                print(name, f"{count} marked", *evidence, sep="\t")
            return 0
        # Each shard's files on a line, spread by their tests' recorded times.
        recorded = recorded_tests()
        for _, names in packed({name: sum(recorded.get(name, [1.0])) for name in tier}, args.shards):
            print(*names)
        return 0
    if args.operation == "pytest":
        # argparse keeps the `--` that sets pytest's arguments apart from these.
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        return targeted(args.python, arguments, args.report, args.tests.split(), args.split, args.whole)
    if args.operation == "gpu-lane":
        return lane_asked(parser, args)
    if args.operation == "coverage":
        if args.stage == "report":
            return coverage_report.report(args.directory)
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps(coverage_items(args.target, timings)))
        return 0
    if args.operation == "policy":
        print(policy(args.base))
        return 0
    if args.operation == "plan":
        print(json.dumps({"include": plan(args.target, json.loads(args.timings.read_text())
                                          if args.timings.is_file() else {}, census=args.census)}))
        return 0
    if args.split == COLLECT:
        return collect(args.python, args.tests.split(), args.out, args.deadline)
    return task(args.python, args.tests.split(), args.split, args.out, args.deadline, args.cache,
                coverage=args.coverage, whole=args.whole == "1", census=args.census == "1")


def pinned(argv: list[str]) -> None:
    """This file run again under HASHED_BY, as uv finds or fetches it, with `argv`, where another Python
    runs it: on a planner's machine, a GitHub runner or a GPU node. Not `coverage report`, which keys nothing
    and reads coverage.py's data under the interpreter its caller installed coverage.py in
    (`coverage_report.report`)."""
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if argv[:2] != ["coverage", "report"] and running != HASHED_BY:
        os.execvp("uv", ["uv", "run", "--no-project", "--python", HASHED_BY, "python",
                         str(Path(__file__).resolve()), *argv])


if __name__ == "__main__":
    pinned(sys.argv[1:])
    raise SystemExit(main())

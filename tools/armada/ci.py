#!/usr/bin/env python3
"""Dew's CI on armada (.armada.json): the plan, and each task it maps.

    python3 tools/armada/ci.py plan --target=SECONDS --timings=FILE
    python3 tools/armada/ci.py task --python=3.12 --tests="tests/a.py tests/b.py" --split=[G/N] --out=FILE
    python3 tools/armada/ci.py durations VERDICT
    python3 tools/armada/ci.py gpu-tier [--evidence] [--shards=N]
    python3 tools/armada/ci.py gpu-lane --tests="tests/b.py#1/2" --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py gpu-lane --network [--tests=...] --out=FILE --cache=on --processes=N
    python3 tools/armada/ci.py gpu-lane --prepare --tests="tests/b.py#1/2" --out=FILE --processes=N
    python3 tools/armada/ci.py coverage items --target=SECONDS --timings=FILE [--scenarios] > ITEMS
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
fewest groups whose heaviest, setup included, holds to `target` (`cuts.split`). A file weighs armada's
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
measured seconds put a task past four group targets (`cut_draws`), which draw them into
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
inputs there. A row keeps no key, and is never reused, where its run was red,
where its task's watch of the files it opened was blind, where it read a file
its own processes wrote, or where it is a group of a file the record its groups
are cut by does not wholly name (`record`). The records live in the dew-xla-cache Worker
under `CACHE_PREFIX`. With DEW_TEST_CACHE=off where the plan runs, which its
entries carry to their tasks (`--cache`), or without the Worker's URL and
token, a task runs every file; with 'cold', it looks rows up as with 'on',
its plan weighing every task it runs cold (`policy`).

A row's record also says where its tests read the backend and which are
marked to run on a GPU (`gpu-tier --evidence` reports it); `gpu-tier` names the
files marked to run on a GPU, a day's and a batch's alike: the behavioural flows
(tests/test_gpu_*.py; no unit test runs on a GPU, ledger 2452), and after them,
on a line of JSON, the GPU generations each file's tests are pinned to (the
mark `generations`; none, a file every generation runs). A GPU
node runs them with `gpu-lane`, each as a task of a CUDA lane, whose rows are
recorded and reused on the GPU's class as a CPU task's are on the CPU's. With `--prepare`, a lane's files
run first on the node's CPU, no GPU drawn, for what their GPU run reads and must not compute while it holds
the GPU (`gpu_prepared`: every test of its part, under the prepare's word, each preparing its float64
truths and its programs compiled for the node's GPU or skipped, as tests/conftest.py decides), nothing of it
recorded. With `--network`, a GPU node that
reaches the Hub runs every file's network-marked tests, which every other task sets aside, as one lane of
their own, recorded on a class of their own, so each train reruns only the files whose inputs changed.

`coverage` measures src/dew's branch coverage over the suite, on demand and never
per batch: `items` prints the plan's tasks for the oldest Python as armada map
items, each running its files under coverage.py (pyproject.toml's
[tool.coverage.run], the Python processes its tests start included) with the
test cache off, since a reused row covers nothing, and writing its data among
the task's artifacts; a traced task records nothing in the cache either.
`report` combines the artifacts' data under DIR and writes DIR/coverage.json and
DIR/coverage.md, a module a line, the least covered first. `items --scenarios` measures the
scenario suite alone (coverage_report.py's `SCENARIO_SUITE`, docs/design/testing.md: a short run a cell
of each scenario):

    armada map --commit=SHA --artifacts=DIR --items=ITEMS --timeout=6000 --secrets=DEW_XLA_CACHE_TOKEN \
        -- sh -c {item}

A task reads the shared XLA cache as CI's do (--commit gives the checkout, its environments and the
cache's URL), and has coverage_report.py's `DEADLINE` and armada's timeout past it; a pass with a task
cut short is refused (`report`), its coverage partial. A target well under CI's keeps tasks short.
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
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Container, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from costs import Cost, Mark
    from run_keys import Reading
    from session import Entry


HERE = Path(__file__).resolve().parent
# The inputs, keys and records of a run (run_keys.py), by its path, and its loader (`run_keys.loaded`), which
# loads every other neighbour here, once each.
_spec = importlib.util.spec_from_file_location("armada_run_keys", HERE / "run_keys.py")
assert _spec is not None and _spec.loader is not None
run_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_keys)
_loaded = run_keys.loaded


@functools.cache
def check1_wire():
    """Check 1's map and judge contract, its one owner (tests/check1_wire.py): what runs a site's test as a
    map task (DRAWS, RECORDS), as their judge (POOL), a judge's ask for its later map tasks (LATER) and its
    stages log (STAGES). Loaded where a plan shards a site, so a checkout of the harness alone (a page
    generator's scratch copy) imports this file."""
    return _loaded(HERE.parents[1] / "tests" / "check1_wire.py")


@functools.cache
def tier_wire():
    """The GPU tier's prepare-to-lane contract, its one owner (tests/tier_wire.py): the prepare's word and
    unit, and where its truths and needs are filed. Loaded where a lane prepares, as `check1_wire` is."""
    return _loaded(HERE.parents[1] / "tests" / "tier_wire.py")


# The neighbours run_keys.py loaded, which this file takes from it, one module of each serving both.
session, records, followed, digests, grading, costs, cuts, rows = (
    run_keys.session, run_keys.records, run_keys.followed, run_keys.digests, run_keys.grading, run_keys.costs,
    run_keys.cuts, run_keys.rows)
child_groups = _loaded(HERE / "child_groups.py")
coverage_report = _loaded(HERE / "coverage_report.py")
r2 = records.r2
watched = _loaded(HERE / "watched.py")
progress = _loaded(HERE / "progress.py")
recycle = _loaded(HERE / "recycle.py")
natives = _loaded(HERE / "natives.py")
host, stalls, temp_roots = (_loaded(HERE / f"{name}.py") for name in ("host", "stalls", "temp_roots"))
partition = _loaded(HERE / "partition.py")
prepares = _loaded(HERE / "prepares.py")
VARIANTS = prepares.VARIANTS
fold = _loaded(HERE / "fold.py")
command_line = _loaded(HERE / "arguments.py")
row, parsed_row, row_place, key_parts = rows.row, rows.parsed_row, rows.row_place, rows.key_parts
grouped_rows = rows.grouped_rows
# The test cache's records (records.py), by the names this file and its tests read them.
(store, record_key, green_key, main_key, baseline_key, epoch_key, Unread, UNREAD, epoch, begun, standing,
 recorded, record_of, written, listed, CACHE_PREFIX, RECORD_TRIES, epochs_of, EPOCH_SECONDS, MAINLINE,
 mainline, merge_base, given_mainline) = (
    records.store, records.record_key, records.green_key, records.main_key, records.baseline_key,
    records.epoch_key, records.Unread, records.UNREAD, records.epoch, records.begun, records.standing,
    records.recorded, records.record_of, records.written, records.listed, records.CACHE_PREFIX,
    records.RECORD_TRIES, records.epochs_of, records.EPOCH_SECONDS, records.MAINLINE, records.mainline,
    records.merge_base, records.given_mainline)
# run_keys.py's names the land queue's tools read from this module (red_rows.py's and suspect_items.py's
# `sites`, `site_task`), and the constants this file's code reads by name; every other reader reads
# run_keys.py's names there.
(sites, site_task, COLLECT, SITE_MARK, INPUTS, ARTIFACTS) = (
    run_keys.sites, run_keys.site_task, run_keys.COLLECT, run_keys.SITE_MARK, run_keys.INPUTS,
    run_keys.ARTIFACTS)
VARIED, varied, cut_of, place_variant = rows.VARIED, rows.varied, rows.cut_of, rows.place_variant
shares, collection = (_loaded(HERE / f"{name}.py") for name in ("shares", "collection"))
gpu_measures = _loaded(HERE / "measures.py")
GPU_HEADROOM, gpu_needs = gpu_measures.GPU_HEADROOM, gpu_measures.gpu_needs
apportioned, deadline_shares, task_shares = shares.apportioned, shares.deadline_shares, shares.task_shares
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
"""The tested versions pyproject.toml's classifiers declare, oldest first: the floor runs every test on every
run, the newest every test on the census and on a pushed run, whatever its cache's policy, the files whose
behaviour changed with it (`NEWEST_FILES`). `pythons` prints them for tools/armada/install.sh, and `pythons
--check` holds .python-version and the environment's smoke (.armada.json) to them."""
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
"""The files the newest Python runs whole on a pushed run, its cache's policy on or off: those its changes
broke before, and the ones around the same machinery. It imports every other test file there (`COLLECT`), and
runs every file on the census: a policy-off train ran every file on both Pythons, doubling its suite (train
542's 3.14 rows were 43 container-hours of 87). The list is kept by hand, from those breaks, because no record
can derive it: the test cache records the checkout's files a run executed and read (executed.py), while each
break lived in the standard library's or a dependency's behaviour under the new Python (PEP 649's lazy
annotations read through tyro, forkserver as the default start method, threads ended at interpreter exit, a
process's allocations under tracemalloc), which no record sees; and the plan runs where the records' token is
not. A file red on the newest Python alone (the census's, a lane's run) joins the list."""
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
"""The files every test file's key covers whole, a Python module by its code (`code_of`), beside its
environment's locks (`shared`), the environment's own inputs, which .armada.json names (`environment_files`),
and .armada.json's parts a test process sees (`armada_seen`): what every test process runs or reads, the
session it runs in, the process group it runs as and the plugins loaded into it among them. The plan's own
code (this file) runs outside the tests, so a change to it reruns none of them; nor does the durations record
(`session.DURATIONS`), which only a split file's groups are cut from, so a group's key covers its own file's
entries alone (`cache_key`); nor does a fixture or a package's data, which is the input of the files whose
runs opened it, however they opened it (executed.py's `Opens`)."""
PREPARE_SIZE = "large"
"""The container every dry prepare task runs on, a kept one's among them (`prepares.entries`), which
compiles a kept row its lookup misses (`unrun`): a prepare compiles a GPU group's programs on the CPU,
which a medium's 10 GiB does not hold (test_gpu_recurrence.py#9/11's peaked at the whole 10 GiB, alone in
1,431 s, and was killed beside other tasks in train 1269). Its tests' recorded peaks are their CPU runs'
and size none of it (`session.sized`). Here, in the plan's code, not in the prepare's recipe
(`prepares.RECIPE`), so placing it keys no prepare's row."""
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


def shared(python: str, locks: tuple[str, ...] = ()) -> str:
    """The hash of what every test file's run shares: `python`, the `SHARED` files, `locks` beside the
    environment's own (a GPU lane's every lock, `lane_inputs`; a CPU task's lock is its environment's),
    .armada.json's parts a test process sees (`armada_seen`), and the names of tests/'s conftest.py and
    __init__.py files and of src/'s own entries, whose arrival can change what an import finds (a conftest.py
    or a top-level package that shadows another), and of every data file under tests/ but its fixtures and
    under src/ (an existence probe before the recorder starts). A module within a package under src/ is its
    importers' input alone, with its package's other modules (`inputs`), a helper module under tests/ the
    input of each run that imported a module of its name (followed.py's `shadowing`), a test module no other
    file's, and a fixture is its openers', listers' and probers' (executed.py's `Opens`, followed.py's
    `PROBES`): so a train that loses a head runs again only what that head's files reached (owner, 2026-10-27:
    8 fixtures changed reran all 485 tasks of train 222's second round)."""
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
            if (path.is_file() and path.suffix != ".py" and not digests.built(path) and path != session.CUTS
                    and digests.FIXTURES not in path.parents
                    and not digests.fetched.manifest(path).is_file()):
                digest.update(b"fixture:" + str(path).encode())
    armada = Path(".armada.json")
    digest.update(armada_seen(armada.read_text() if armada.is_file() else None).encode())
    for name in sorted({*SHARED, *locks, *environment_files(), "src"}):
        digest.update(name.encode() + (code_of(name) if name.endswith(".py") else digests.contents(name)))
    return digest.hexdigest()


def lane_inputs(python: str) -> str:
    """The hash of what a GPU lane's file shares (`gpu_lane`): what every test file's run shares (`shared`),
    and this file and prepares.py, which set the environment (a user's process's, `VARIANTS`), share and
    deadline each file runs under on a GPU, where a CPU task's are the session's."""
    locks = tuple(map(str, Path("locks").glob("*.txt")))
    own = code_of(__file__).hex() + code_of(str(HERE / "prepares.py")).hex()
    return hashlib.sha256((shared(python, locks) + own).encode()).hexdigest()


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


def reusable(python: str, rows: list[tuple[str, str]], common: str,
             missed: dict[str, int]) -> dict[tuple[str, str], str]:
    """The rows of `rows` (each a file and its group, "" for the whole file) that ran green on exactly these
    inputs (`common`, and each file's own, `fresh`) on any class the tasks run on (`pinned_classes`),
    which a task of theirs could land on, read at plan time with the cache's read token (`READ_TOKEN`): a
    task of their own would only look each up and exit. A heavy file's groups are rows like any other,
    each looked up by its own (2026-10-30: the plan looked up whole files alone, and a docs-only change's
    1,034 green groups of heavy files each booted a container to read its record). Each with where its
    green run is kept (`green_key`'s class and key, `session.KEPT_AT`'s token), which the plan hands the
    task that holds it, so no run of another commit replacing a row's last record between plan and task
    can hide it (2026-10-30: a kept task found 23 of its 259 files and ran the rest). None where the plan
    has no token or no class (the maintainer, 2026-10-28: a lane run with most files unchanged booted a
    container a task; 2026-10-29: the suite pinned to 25/17 and 26/17). Why each other row was not,
    counted in `missed` (`green_record`'s word)."""
    token, held = os.environ.get(READ_TOKEN), pinned_classes()
    if not token or not held or "DEW_XLA_CACHE_URL" not in os.environ:
        return {}
    # The plan's one reading of the checkout, which every row's key reads its inputs from.
    reading = run_keys.Reading()

    misses: dict[tuple[str, str], list[str]] = {}
    def hit(each_row: tuple[str, str]) -> str | None:
        name, group = each_row
        for each in held:
            found = green_record(python, name, group, common, token, [each], reading,
                                 misses.setdefault(each_row, []))
            # Kept only where its green run is kept by its key, as the task reads it.
            if found is not None and recorded(
                    green_key(python, row(python, name, group), each, found[0]["key"]), token) is not None:
                return f"{each}:{found[0]['key']}"
            if found is not None:
                misses[each_row].insert(0, records.MISSED["kept"])
        return None

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        found = {each: kept for each, kept in zip(rows, pool.map(hit, rows), strict=True) if kept}
    missed.update(Counter(misses[e][0] for e in rows if e not in found))
    return found


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


def kept_token(name: str, group: str, at: str) -> str:
    """The token of file `name`'s row in `group` ("" for the whole file), kept where `at` says
    (`session.KEPT_AT`)."""
    return f"{name}{session.GROUP_AT + group if group else ''}{session.KEPT_AT}{at}"


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
    """A task's `tests`, each a file's name, or a kept row's token, its file's name with its group where
    it has one and where its green run is kept (`session.KEPT_AT`): the files the task runs whole, those
    kept among them, and each kept row's class and key by its file and group."""
    names, kept = [], {}
    for test in tests:
        # Where its green run is kept follows the last mark: a row's group may hold one of its own (a process
        # variant's, `varied`).
        at_row, _, at = test.rpartition(session.KEPT_AT) if session.KEPT_AT in test else (test, "", "")
        name, _, group = at_row.partition(session.GROUP_AT)
        if not group:
            names.append(name)
        if at:
            held, _, key = at.partition(":")
            kept[(name, group)] = (held, key)
    return names, kept


def green_record(python: str, name: str, split: str, common: str, token: str, classes: list[str],
                 reading: "Reading | None", why: list[str]) -> tuple[dict, str] | None:
    """The record of test file `name`'s row (in `split`'s group) that ran green on exactly these inputs on one
    of `classes`, the first found, with its class; else None. The plan (`reusable`) and a task (`reused`) read
    it alike, from one reading of the inputs (`Reading`). A row's key hashes the inputs its run executed, so
    these inputs are keyed with the executed set of the last run on these shared inputs: its own record where
    its key holds here, else the green run of exactly that key (`green_key`), whatever ran since; and where
    neither holds, with the set of main's own run of it at this checkout's main (`mainline`, `main_key`),
    which a lane's later run on its branch re-recording the row hides from the last record (Main,
    2026-11-14: a walk over each row's earlier sets found it, at about seven reads a row, 139 s of a plan;
    two reads now). A hit is sound: its key equals a green run's only where every input that run executed,
    its units among them, holds here what it held then. A miss says why in `why`, a class at a time
    (records.py's `MISSED`)."""
    reading = run_keys.Reading() if reading is None else reading
    place = row(python, name, split)
    for each in classes:
        try:
            current = epoch(python, place, each, common, token)[0]
        except Unread:
            why.append(records.MISSED["unread"])
            continue
        if current is None:
            why.append(records.MISSED["shared"])
            continue
        found = recorded(record_key(python, place, each, common), token)
        # The last run's inputs hashed once: its own key where fresh (`fresh`), else the green run of exactly
        # these, kept by that key (`green_key`) whatever ran since; why it was not, the class's word.
        recorded_set = None if found is None else found["depends"]
        key = None if recorded_set is None else run_keys.cache_key(common, name, split, recorded_set, reading)
        why.append(records.missed(found, key, current))
        if key is not None:
            if found["key"] is not None and found["key"] == key and standing(found, current):
                return found, each
            kept = recorded(green_key(python, place, each, key), token)
            if kept is not None and kept["key"] == key and standing(kept, current):
                return kept, each
        base = mainline()
        named = {} if base is None else recorded(main_key(base, python, place, each), token) or {}
        main_run = None if not named else recorded(green_key(python, place, each, named["key"]), token)
        if (main_run is not None and main_run["depends"] is not None and main_run["key"] == named["key"]
                and run_keys.cache_key(common, name, split, main_run["depends"], reading) == named["key"]
                and standing(main_run, current)):
            return main_run, each
    return None


def reused_classes(network: bool = False) -> list[str]:
    """The classes whose green rows a task on this host reuses: every class the tasks are pinned to, where
    this host's is one of them, as the plan counted a row green on any of them reusable (`reusable`), so the
    task holding the files it kept reuses them wherever it lands; else this host's class alone."""
    own, held = run_keys.device_class(network), pinned_classes()
    return held if own in held else [own]


def reused(python: str, name: str, split: str, common: str, network: bool = False,
           kept: tuple[str, str] | None = None, reading: "Reading | None" = None,
           named: dict[str, tuple[str, str]] | None = None) -> dict | None:
    """The green row of test file `name` (in `split`'s group) recorded on a class this host reuses
    (`reused_classes`), if it ran green there on exactly these inputs, `cached` with that run's key (armada
    counts it reused and times no file by it); else None. A file its plan kept
    (`kept`: its green run's class and key, `reusable`) reads that run, fresh here; any other, the record of
    the last run on these shared inputs: a key hashes them (`cache_key`), so no run on others, whose record
    the last run's may be, can be fresh here, all from one reading of the inputs (`Reading`). The run it
    reads is the commit's to name (`named_at_commit`) once its row is graded: `named` takes its row's name to
    that run's class and key (`reused_rows`)."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return None
    place = row(python, name, split)
    if kept is not None:
        found, current = recorded(green_key(python, place, *kept), token), None
        try:
            current = epoch(python, place, kept[0], common, token)[0]
        except Unread:
            found = None
        if found is not None and found["key"] == kept[1] and fresh(found, name, common, reading) and standing(
                found, current):
            if named is not None:
                named[found["row"]["name"]] = kept
            return {**found["row"], "cached": found["key"]}
    hit = green_record(python, name, split, common, token, reused_classes(network), reading, [])
    if hit is None:
        return None
    if named is not None:
        named[hit[0]["row"]["name"]] = (hit[1], hit[0]["key"])
    return {**hit[0]["row"], "cached": hit[0]["key"]}


def fresh(recorded: dict, name: str, common: str, reading: "Reading | None" = None) -> bool:
    """Whether test file `name`'s record was made by a green run on exactly these inputs, as `reading` finds
    them (`cache_key`). A run that keeps no key (`record`: a red one among them) is fresh on none."""
    return recorded["key"] is not None and recorded["key"] == run_keys.cache_key(
        common, name, parsed_row(recorded["row"]["name"])[2], recorded["depends"], reading)


def recorded_tests() -> dict[str, list[float]]:
    """Each file's tests' times recorded (`own_tests`), one shared setup's one weight (`session.units`)."""
    found = {name: run_keys.own_tests(name, tests) for name, tests in run_keys.parted_record().items()}
    return {name: session.unit_seconds(list(own), list(own.values())) for name, own in found.items() if own}


def stored_weights() -> dict[str, list[float]]:
    """Each split file's stored cut (`session.CUTS`) that still holds its recorded tests (`own_tests`,
    `session.holds`) as its groups' recorded seconds, which the plan runs it as (`cuts.weighed`)."""
    stored, parted, held = session.stored_cuts(), run_keys.parted_record(), {}
    for name, bounds in stored.items():
        if (own := run_keys.own_tests(name, parted.get(name, {}))) and session.holds(
                groups := session.cut_by(list(own), bounds), own, session.held_seconds()):
            held[name] = [sum(own[node] for node in group) for group in groups]
    return held


def weights(files: list[str], timings: dict, python: str = PYTHONS[0]) -> dict[str, float | None]:
    """Each file's seconds on armada under `python`: its median, which armada sums over a split file's groups
    (`task`), or its tests' sum in tests/test_durations.json where that is more: each test's record is its
    recorded time (`durations`), so a sum past the median may mean the measured runs predate the file's
    current tests, and the sum is a floor on what it takes. base/kernels' route census, 1,576 tests recorded
    at 2,654 s in a file main ran in 6 s, was packed by that median with six other files into one task, which
    ran past its deadline (train 18). Else the recorded sum, scaled by how much slower armada ran the files
    measured both ways, and never below it, a floor as a measured file's is (a test's cold time outweighs a
    warm run's measure); else None, for a file nothing has timed."""
    recorded = {name: sum(tests) for name, tests in recorded_tests().items()}
    # A file armada holds no timing of weighs its rows' sum at the cut they name (`measured_cut`), which is
    # what its timing would say: armada keeps a plan's rows from run to run, and drops the file timings a run
    # did not itself measure (ledger 5579).
    measured = {}
    for name in files:
        held = timings.get("files", {}).get(row(python, name, ""))
        cut = measured_cut(timings, python, name) if held is None else None
        if held is not None or cut is not None:
            measured[name] = held if cut is None else cut[0]
    ratios = sorted(measured[name] / recorded[name] for name in files
                    if name in measured and recorded.get(name, 0) > 1)
    slower = ratios[len(ratios) // 2] if ratios else 1.0
    return {name: max(measured[name], recorded.get(name, 0.0)) if name in measured
            else recorded[name] * max(slower, 1.0) if name in recorded else None for name in files}


def measured_cut(timings: dict, python: str, name: str) -> tuple[float, int] | None:
    """What `name` took under `python` at the cut its rows in `timings` name, and that cut's count
    (`cuts.measured_cut`, over its rows by their groups, `grouped_rows`)."""
    return cuts.measured_cut(grouped_rows(timings.get("rows", {})).get((python, name), {}))


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
"""Where a changed file is placed by the records (executed.py): a Python file a test ran, a page, a notebook
or a workflow it read (followed.py's `HIDDEN_INPUTS`), a fixture its openers read. A native kernel package's
sources (kernels/), which the environment installs as a built wheel, never as these files, a type stub, a
license, an image and thresh's annotations are placed alike: each is the input of the files whose runs read
it, of none where none does. A change elsewhere is placed by none, and runs the whole suite (`policy`): a
workflow change reran every file while no record placed it (the maintainer, 2026-10-28: 5 of 10 trains ran
with the cache off, 0.75% of rows reused), and a change to a kernel package's wrapper ran train 537's whole
suite on both Pythons."""
PLACED_FILES = ("README.md", "CONTRIBUTING.md", "AGENTS.md", "LICENSE", "CHANGELOG.md", "MANIFEST.in",
                ".gitattributes")
"""The checkout's own files at its root that tests read as a page is read, and the changelog, which no test
reads (test_changelog_merge writes its own) and most heads add an entry to: placed by none, it turned the
cache off for every train holding such a head, which then ran its whole suite on every round (train 149: 30
task-hours, twice)."""


def policy(base: str) -> str:
    """The test cache's policy (DEW_TEST_CACHE) for this checkout's change from `base`: 'off', every file run,
    for a change to a file the records do not place (`PLACED`), which no key covers; 'cold' for a change to
    what every test process runs or reads (`SHARED`, a Python module of it by its code; a lock, the
    environment's files), which every key covers (`shared`), so no row of main's is found and every file runs,
    the plan weighing each task cold, while a train composed again on the same inputs, and the next on them,
    finds the rows the first ran green (owner, 2026-11-13: train 793's second round reran all 4,113 tasks its
    first had run under the same inputs); else 'on', each file run where its inputs changed. A tool is such an
    input of each file whose run executed or read it, which its key hashes, so a changed tool runs those files
    again and no other (train 115 reran its whole suite for five tools no pulled head had touched, under a
    policy that reran every file that read any tool)."""
    changed = subprocess.run(["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"], check=True,
                             capture_output=True, text=True).stdout.split()
    if ".armada.json" in changed:
        before = at_commit(base, ".armada.json")
        armada = Path(".armada.json")
        if armada_seen(None if before is None else before.decode()) == armada_seen(
                armada.read_text() if armada.is_file() else None):
            changed.remove(".armada.json")  # where and when tasks run, never what a test sees
    whole = {*SHARED, *environment_files(), ".armada.json", ".python-version"}
    # A lock the environment is not built from is a GPU lane's alone (`lane_inputs`): it reruns no CPU file.
    changed = [name for name in changed if not name.startswith("locks/") or name in whole]
    # A SHARED module whose code `base` holds as it is (a docstring's or a comment's edit) keys every row as
    # it did (`code_of`, `shared`), so it leaves the cache on (Main, 2026-11-07: 12 of the queue's last 20
    # train runs ran cold on harness edits).
    changed = [name for name in changed
               if not (name in SHARED and name.endswith(".py") and same_code(base, name))]
    if any(not (name.startswith(PLACED) or name in PLACED_FILES) for name in changed if name not in whole):
        return "off"
    return "cold" if any(name in whole for name in changed) else "on"


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
    (the most of theirs): 3.14 runs every file only on the census, and a file weighed by its
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
"""What a task that runs a file adds to its estimate when it runs cold, the test cache's policy off or cold (a
lock, JAX's build or the recorder changed): the compiles and imports no warm run paid. Train 254's cold suite
ran about 470 s a task where its plan estimated about 120 (1,389 tasks in 2,700 s on 242 containers,
2026-10-28)."""

LATER_JUDGED = "check1_later"
"""A check-1 site judge's row's word that its judgement needed its later map tasks (`later_judged`), which the
next plan reads to draw them in the first wave (`eager_sites`)."""


def rate(python: str, name: str, test: str, spans: Sequence[tuple[int, int]], measured: Mapping[str, float],
         timings: Mapping[str, float], *, weighed: bool) -> "Cost | None":
    """What a map task of test file `name`'s site `test` takes under `python`: its setup and a unit's draws as
    its last judge timed them from its map tasks' records (`timings`, armada's file timings, `site_costs`),
    whatever their sizes; else over those of its map tasks' `spans` that armada measured (under their task
    names, `site_task`, `costs.fitted`); None where it measured none. Tasks of one size, cut by their seconds
    over units, charged each draw the setup, so a site was cut finer and measured at one size again, run after
    run (2026-11-11: 565 map tasks to 1,344), which the records' setup ends. Once a site runs as map tasks,
    the durations record holds no whole run of it (its test's seconds are one task's, its judge's pooling
    among them), so its map tasks are what tell its cost."""
    found = costs.recorded(run_keys.site_task(python, name, test), timings, weighed=weighed)
    return found or costs.fitted([
        (stop - start, seconds) for start, stop in spans
        if (seconds := measured.get(run_keys.site_task(python, name, test, (start, stop))))])


def site_rate(python: str, name: str, test: str, mark: "Mark", timings: Mapping[str, Mapping[str, float]], *,
              weighed: bool) -> "Cost | None":
    """What a map task of test file `name`'s site `test` takes under `python` (`rate`), by its judge's records
    of its map tasks (`timings`' files), else over its map tasks of any cut armada measured, by their names
    (`site_task`), each span read from its name's end: its weighed check's units apart from the rest
    (`weighed`), as they cost their own."""
    measured = timings.get("rows", {})
    prefix = f"{run_keys.site_task(python, name, test)}-draws-"
    spans = [(int(start), int(stop)) for task in measured if task.startswith(prefix)
             for start, _, stop in [task[len(prefix):].partition("-")]]
    return rate(python, name, test, [span for span in spans if (span[0] >= mark.units) == weighed], measured,
                timings.get("files", {}), weighed=weighed)


def site_costs(python: str, name: str, test: str) -> dict[str, float]:
    """A judge's timings of its site's map tasks (`costs.timed`), from the records each wrote among its
    artifacts beside its draws (map_judge's `record_units`: `<site>.<start>-<stop>.seconds.json`), which its
    inputs hold (ARMADA_INPUTS); nothing where they hold none of the site's."""
    held = run_keys.sites(name)
    if test not in held:
        return {}
    records = []
    for path in run_keys.armada_dir("ARMADA_INPUTS", INPUTS).rglob("*.seconds.json"):
        record = json.loads(path.read_text())
        python_, _, rest = record["site"].partition("::")
        node = rest.rpartition("::")[0]
        if (python_, grading.owning(node)) == (python, name) and session.site_name(node, held) == test:
            # The record's own words, those of what its draws paid once among them where it keeps them.
            records.append(costs.Drawn(**{word: record[word] for word in costs.Drawn.__annotations__
                                          if word in record}))
    return costs.timed(run_keys.site_task(python, name, test), records, held[test].units)


def site_entries(python: str, marked: "Mapping[str, Mapping[str, Mark]]",
                 kept: Mapping[tuple[str, str], str], timings: dict, target: float,
                 eager: Container[tuple[str, str]] = ()) -> list[dict]:
    """The plan's entries for each site of `marked` (`sites`, by its file) under `python`: a map task each
    `per_task` of its units, or fewer where their measured rate puts a task past four of `target`
    (`cut_draws`), which runs its test over them alone, records their draws among its artifacts and judges
    nothing; and its judge, which runs once they all passed (armada's `after`), pools
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
            spans = costs.cut_draws(mark, rates, target)
            maps = []
            for start, stop in spans:
                group = f"{test}.draws-{start}-{stop}"
                task_name = run_keys.site_task(python, name, test, (start, stop))
                # A weighed check's units cost their own, as those of them armada measured did.
                cost = rates[start >= mark.units]
                weight = measured.get(task_name) or (
                    cost.seconds(start, stop) if cost is not None else target)
                # A check-1 site's second-stage seeds, drawn only where its judge's first stage asks for
                # them (armada's `later`), or at once where its last judgement needed them (`eager`).
                maps.append({"name": task_name, "python": python, "tests": name, "split": group,
                             "rows": [row(python, name, group)], "weight": weight,
                             **({"later": True} if start >= mark.timed and (name, test) not in eager
                                else {})})
            entries += [*maps, {"name": run_keys.site_task(python, name, test), "python": python,
                                "tests": name, "split": test, "after": [each["name"] for each in maps],
                                "weight": measured.get(row(python, name, test)) or target}]
    return entries


def plan(target: float, timings: dict, census: bool = False, reach: dict | None = None) -> list[dict]:
    """The matrix's entries: about `target` seconds of files each, for each Python. A file nothing has timed
    runs alone, so however long it takes it holds up no other file, and its time is known from then on. Files
    marked `torchax` (`APART`) share tasks only with one another. Each entry carries the test cache's policy,
    DEW_TEST_CACHE here (`policy`), to its task. Under the policy "on", the files whose rows the cache holds
    for exactly these inputs (`reusable`) share tasks of the target's lookups each, which look each up again
    in their containers, so the verdict names every file and containers scale with the change. Under the
    policy "off" or "cold" every task that runs a file carries COLD_SECONDS, so armada sizes the pool by the
    work, not the longest task, as warm estimates did. A site marked one of `session.SHARDS` runs as armada's
    map and judge, apart from its file's tasks (`site_entries`). The `census` (check 1's CENSUS) is the whole
    suite with the cache off, whatever this environment says, each entry carrying it to its task."""
    cache = "off" if census else os.environ.get("DEW_TEST_CACHE") or "on"
    # What every entry carries to its task (.armada.json's command): the cache's policy, the census, and
    # main's commit, which a task's checkout cannot find (`mainline`).
    stamped = {"cache": cache, "census": "1" if census else "0", "main": mainline() or ""}
    every = test_files()
    # Every file's check-1 sites its marks shard, refused where one site's map tasks outnumber what its
    # judges' job takes (`refused_sites`).
    shards = {name: held for name in every if (held := run_keys.sites(name))}
    costs.refused_sites(shards)
    # Every file's shared setups, read so a mark the plan cannot read refuses its file now
    # (`session.shared_setups`), not where a cut first reads it.
    for name in every:
        session.shared_setups(name)
    recorded = recorded_tests()
    stored = stored_weights()
    heavy = session.heavy_files()
    entries = []
    for python in PYTHONS:
        # Every file on the floor, and on the census every file on every Python; on a pushed run, its cache's
        # policy on or off, the newest takes `NEWEST_FILES` whole.
        files = every if python == PYTHONS[0] or census else [name for name in every if name in NEWEST_FILES]
        timings = borrowed(timings, python)
        timed = weights(files, timings, python)
        # Each file's rows: a file heavier than the target runs as the groups it is cut into, each a row of
        # its own; a file one group holds runs whole, alone, under its own row.
        groups: dict[str, list[tuple[str, float]]] = {}
        for name, seconds in timed.items():
            if seconds is not None and seconds > target:
                cut = cuts.weighed(session, recorded.get(name, []), seconds, target,
                                   measured_cut(timings, python, name), stored.get(name, ()))
                groups[name] = [(f"{group}/{len(cut)}" if len(cut) > 1 else "", weight)
                                for group, weight in enumerate(cut, start=1)]
        # Each check-1 site its file's marks shard is a row of its own, its judge's.
        marked = {name: shards[name] for name in files if name in shards}
        rows_of = [(name, "") for name in timed if name not in groups] + [
            (name, group) for name, cut in groups.items() for group, _ in cut] + [
            (name, test) for name, held in marked.items() for test in held]
        missed: dict[str, int] = {}
        found = reusable(python, rows_of, shared(python), missed) if cache != "off" else {}
        if reach is not None:  # reach.py's: the rows and those kept (None off "on")
            reach[python] = (rows_of, found if cache == "on" else None)
        print(f"plan: {python}: kept {len(found)} of {len(rows_of)}; not kept: {missed}", file=sys.stderr)
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
                            "weight": weight + (COLD_SECONDS if cache != "on" and any(
                                each not in found for each in held_rows) else 0.0), **stamped})
        # The sites to run whose judge's last run needed its later map tasks draw them at once
        # (`eager_sites`).
        eager = eager_sites(python, [(name, test) for name, held in marked.items() for test in held
                                     if (name, test) not in found])
        entries += [{**entry, "weight": entry["weight"] + (COLD_SECONDS if cache != "on" else 0.0),
                     **stamped}
                    for entry in site_entries(python, marked, found, timings, target, eager)]
        # The GPU tier's files prepared dry on the floor Python's CPU (`prepare_entries`), beside its rows,
        # apart from what reach and the census read, each task on a `PREPARE_SIZE` container.
        if python == PYTHONS[0] and not census:
            entries += [{**entry, "size": PREPARE_SIZE}
                        for entry in prepares.entries(sys.modules[__name__], python, timings, target, cache,
                                                      stamped)]
    # What the newest Python runs no test of, it imports (`COLLECT`): on a pushed run, every file but its own.
    rest = [] if census else [name for name in every if name not in NEWEST_FILES]
    collecting = collect_entries(rest, timings, target)
    return keyed([session.sized(entry, heavy)
                  for entry in entries + [{**entry, **stamped} for entry in collecting]])


def keyed(entries: list[dict]) -> list[dict]:
    """`entries`, each with the `key` armada keeps its recorded use under (its `keyOf`): the rows it runs (a
    site's judge by its name), whether it was kept (it looks its rows up and runs none) and how it runs them
    (the cache's policy and the census's compiles, each its own use), never an entry's
    name or weight, which each plan cuts anew. A task that recurs then packs on what it used, not on its
    class's estimate (2026-11-07: no record of a task ever matched the next plan's, and each container's
    CPU was about a quarter used). Hashed: a kept task's entry, which travels whole as one environment
    string, holds up to `KEPT_BYTES` of row names already."""
    def key(entry: dict) -> str:
        identity = {"rows": entry.get("rows") or [entry["name"]], "kept": session.KEPT_AT in entry["tests"],
                    **{field: entry.get(field) for field in ("cache", "census")}}
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


KEPT_UNFOUND = "stale, not run: the plan kept it on a record this task did not find, "
"""How a kept row this task did not find or could not run begins what it says (`stale_row`)."""


def stale_row(python: str, name: str, group: str, why: str) -> dict:
    """Row `name` (in `group`'s group) with no verdict of its tests (`grading.STALE`), not run, saying `why`:
    a kept row this task did not find, which the queue runs again, or a unit its GPU lane cannot hold
    (`gpu_lane`), which Platform's harness evidence says, never a red of a test nor its file's verdict (Main's
    audit, row 7614: as a red of its own kind, the tier read it as its file's verdict and its tests' red)."""
    return {"name": row(python, name, group), "exitCode": 1, "seconds": 0.0, "output": why,
            "tests": {}, "ynn_switched": [], "cells": {}, "timings": {row(python, name, ""): 0.0},
            **grading.red_as(grading.STALE, [])}


def unbaselined(python: str, rows: list[dict], unnamed: Container[str]) -> list[dict]:
    """`rows`, each dry prepare's green row whose baseline the CI store did not take (`unnamed`,
    `baselined`) said stale: it runs again rather than stand green with no baseline."""
    why = "its prepare's baseline was not written to the CI store; it runs again"
    named = {found["name"]: parsed_row(found["name"]) for found in rows if found["name"] in unnamed}
    # Stale, keeping what its prepare left unprepared and its output (a capped row's banner).
    return [{**stale_row(python, named[found["name"]][1], named[found["name"]][2], why),
             **{field: found[field] for field in ("unprepared", "capped") if field in found},
             "output": "\n".join([why, found.get("output") or ""])} if found["name"] in named
            else found for found in rows]


def stale_cost(name: str, group: str, reading: "Reading") -> float:
    """What a kept row of test file `name` (in `group`'s group) costs a task that runs it, its record not
    found: its tests' recorded seconds, its check-1 sites' aside, which run apart (`own_tests`; a group's,
    `members`), each its recorded cost, and what a run cold adds to them (`COLD_SECONDS`), as the plan weighs
    a cold row; a file the record does not name, that alone."""
    reading.cut(name, 1)
    own = run_keys.own_tests(name, (reading.durations or {}).get(name, {}))
    evenly = prepares.prepared_group(group, VARIED)
    nodes = run_keys.members(name, cut, reading, evenly=evenly) if (cut := cut_of(group)) else list(own)
    return sum(own.get(node, 0.0) for node in nodes) + COLD_SECONDS


def afforded(python: str, tests: list[str], group: str, kept: Container[str],
             left: float) -> tuple[list[str], list[dict]]:
    """Of `tests` a task still has to run, those it runs and the rows of those it does not: a row its plan
    kept (`kept`) that this task did not find (its record replaced since) runs here only where what it costs
    run cold fits what is `left` of the task's deadline, in turn (`stale_cost`); any other is said stale, not
    run (`grading.STALE`), which the queue runs again as a row of its own, neither a red of its tests nor
    flaky (Main, 2026-11-07; row 4468: 163 of a kept task's 165 rows ran out its deadline one after
    another)."""
    fitting, stale_rows, reading = [], [], run_keys.Reading()
    for name in tests:
        cost = stale_cost(name, group, reading)
        if name in kept and cost > left:
            stale_rows.append(stale_row(python, name, group,
                                        f"{KEPT_UNFOUND}its {cost:.0f} s past the {left:.0f} s left"))
            continue
        fitting.append(name)
        left -= cost
    return fitting, stale_rows


def collect(python: str, tests: list[str], out: Path, deadline: float) -> int:
    """Import `tests` under `python` without running them and write one verdict row a file: red
    where collecting it errs or skips on a missing import, which running it would have."""
    began = time.monotonic()
    command = [f".venv-{python}/bin/python", "-m", "pytest", "--collect-only", "-q", "-rs", *tests]
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
             split: str = "") -> int:
    """pytest's exit status on `arguments` (files, node ids and options) and `tests` in `split`'s group, run
    as a CI task runs them (`session.pytest_run`, `selection`, `site_environment`): a lane's targeted run
    (tmp/run_tests.py) and the gangs select as CI does, and an option of their own (a `-m` after CI's)
    chooses instead. A file's group sets aside its sites as CI's does (`selected`), which run in their own
    tasks alone (2026-11-02: a targeted group ran a site's every seed itself, 2233 s against CI's 8); a node
    id named with no split runs what it names, a site too (row 8963: it selected nothing)."""
    site, named = session.SITE.fullmatch(split), any("::" in each for each in arguments) and not split
    run = session.selected(session.pytest_run(python, report), network=False, site=site is not None or named)
    return subprocess.run([*run, *arguments, *session.selection(list(tests), split)],
                          env=session.pytest_environment(site_environment(site)), check=False).returncode


def task(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str = "on",
         grace: float = session.GRACE, coverage: Path | None = None, census: bool = False) -> int:
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
    # Each pytest the task starts, in order: the files one process runs together where they share one (a
    # group's, a site's, coverage's, the census's) or there is one; else each whole file its own, so its
    # record holds what its own process ran, never what files before it imported or executed (2026-11-23:
    # tests/test_records.py imports 9 dew modules, and a packed process's record of it held 802); then each
    # kept group its own. One process start a file, about 2 s (armada, 25/17).
    together = bool(split) or coverage is not None or census or len(names) < 2
    runs = ([(names, split, whole_files)] if together
            else [([name], "", {name: whole_files[name]} if name in whole_files else {}) for name in names])
    runs += [([name], group, {name: at}) for (name, group), at in kept.items() if group]
    # Every reuse found before any of them starts: a process may leave the checkout changed, and a later
    # key read after it would hash what it left (row 3924's review).
    reading = run_keys.Reading()
    found = [unrun(python, files, group, session.SITE.fullmatch(group), cache, common, network=False,
                   kept=held, judged=None, deadline=deadline, reading=reading, afford=False)
             for files, group, held in runs]
    rows = [each for reused, _ in found for each in reused]
    code, spent, written = 0, "", []
    for index, ((_, group, held), (_, left_tests)) in enumerate(zip(runs, found, strict=True)):
        # A kept row whose record was not found runs where its cold cost fits what is left now (`afforded`).
        spare = deadline - (time.monotonic() - began)
        stale_rows: list[dict] = []
        if held:
            left_tests, stale_rows = afforded(python, left_tests, group, held, spare)
        rows += stale_rows
        if not left_tests:
            continue
        # The checkout's untracked entries, scanned only where a later process will run (`restored`).
        later = index < len(runs) - 1
        before = untracked() if later else set()
        if (left := deadline - (time.monotonic() - began)) <= 0:
            # Never started: stale, run again, at no seconds armada weighs a plan by (`unobserved`), as a
            # process's file it never reached is (`tested`).
            why = (f"not graded: its task's {deadline:.0f} s were spent {spent or 'finding reuse'}, "
                   "before it began")
            rows += [{**stale_row(python, name, group, f"{why}; it runs again"), "unobserved": True}
                     for name in left_tests]
            continue
        spent = "in " + " ".join(left_tests) + (f" ({group})" if group else "")
        part = out.with_name(f"{out.name}.{index}")
        code = max(code, tested(python, left_tests, group, part, left, "off", grace, common, environ,
                                coverage, written=written))
        rows += json.loads(part.read_text())["rows"]
        # The next process sees the commit's checkout, not what this one left in it (row 3932's review).
        if later:
            restored(written[-1] if written else set(), before, out)
    out.write_text(json.dumps({"rows": rows}))
    # A map task's red row fails its task: armada runs no judge on draws a map task did not pass, and reads
    # the judge red, naming it (`site_entries`).
    site = session.SITE.fullmatch(split)
    if site and site["start"] is not None and any(found["exitCode"] for found in rows):
        return told(out, 1)
    return told(out, code)


def told(out: Path, code: int) -> int:
    """`code`, else its commit's .armada.json's task.redExit (0 where it names none: the rows its verdict),
    which armada reads as the rows written, a packed task rerun alone, where a row is red; the task's verdict
    `out` said last: each red row, its name and its output's first lines (why it is red, its signal and the
    file it reached: `tested`) and last (what pytest said last). armada keeps a task's log tail beside its
    exit, and a map task's verdict reaches no report but its judge's red row, which names that tail (Main,
    2026-11-09: a map task "exited 1 after 437 s", its row's "pytest exited -9" kept in its output alone)."""
    rows = json.loads(out.read_text())["rows"]
    for found in rows:
        if found["exitCode"]:
            lines = (found.get("output") or "").strip().splitlines()
            shown = (lines if len(lines) <= 2 * TOLD_LINES
                     else [*lines[:TOLD_LINES], "...", *lines[-TOLD_LINES:]])
            print("\n".join([f"RED {found['name']}", *(f"  | {line}" for line in shown)]), flush=True)
    config = REPO / ".armada.json"
    red = int(json.loads(config.read_text()).get("task", {}).get("redExit", 0)) if config.is_file() else 0
    return code or (red if any(found["exitCode"] for found in rows) else 0)


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
        return {wire.POOL: str(run_keys.armada_dir("ARMADA_INPUTS", INPUTS)), wire.LATER: str(later_file()),
                wire.STAGES: str(stages_file())}
    wire = check1_wire()
    return {wire.DRAWS: f"{site['start']}:{site['stop']}",
            wire.RECORDS: str(run_keys.armada_dir("ARMADA_ARTIFACTS", ARTIFACTS))}


def asked(site: re.Match | None) -> dict[str, bool]:
    """A row's ask for more, where its task is a check-1 site's judge that asked for its later map tasks
    (`ask_later`): armada runs them and it judges again (its `more`); nothing else."""
    return {"more": True} if site and site["start"] is None and later_file().is_file() else {}


def later_file() -> Path:
    """Where a judge writes that its judgement needs its later map tasks (LATER, `ask_later`)."""
    return run_keys.armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / "check1-later.json"


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
    return run_keys.armada_dir("ARMADA_ARTIFACTS", ARTIFACTS) / "check1-stages.jsonl"


def reused_rows(python: str, tests: list[str], group: str, common: str, network: bool,
                kept: Mapping[str, tuple[str, str]] | None = None,
                named: dict[str, tuple[str, str]] | None = None,
                reading: "Reading | None" = None) -> list[dict]:
    """The green rows of `tests` in `group` recorded on these inputs on this host's class (`reused`), those
    its plan kept read where it found them (`kept`), said, each one's run (its class and key) in `named` by
    its row's name. They are looked up before any test runs, from one reading of the checkout (`Reading`)."""
    kept = kept or {}
    reading = run_keys.Reading() if reading is None else reading
    rows = [found for name in tests
            if (found := reused(python, name, group, common, network, kept.get(name), reading, named))]
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
    opens = watched.Opens(os.getcwd(), frozenset({str(session.DURATIONS), str(session.CUTS)}))
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
    if left := child_groups.stop([done], grace=max(grace - spent, 0.0) if expired else 0.0,
                                 interrupt=signal.SIGINT, settle=SETTLE):
        print(f"processes {left} of the task's group still ran {SETTLE:.0f} s after SIGKILL", flush=True)
    # A process of the group still running may yet open a file: no record of this task keeps a key.
    closed = opens.closed()
    reader.join(10)
    wrote = watched.changed(before, watched.stamps(os.getcwd(), followed.HIDDEN_INPUTS))
    return done.returncode, "".join(lines), expired, dump, None if left else closed, wrote


def evidence_name(python: str, tests: list[str], group: str) -> str:
    """The name a run's evidence is kept under among the task's artifacts: the rows it grades, so a task's
    several runs (a GPU lane's) each keep their own."""
    graded = "\n".join(row(python, name, group) for name in tests)
    return f"{python}-{hashlib.sha256(graded.encode()).hexdigest()[:16]}"


def watched_pytest(command: list[str], environment: dict[str, str], deadline: float, grace: float, pid: Path,
                   python: str, name: str, began: float
                   ) -> tuple[int, str, bool, str, set[str] | None, set[str], dict]:
    """pytest run as a task runs it (`ran_pytest`), with the task environment's py-spy reading a stalled run's
    stacks (`stalls.dumped`) and a collector outside it reading those of any process of it a signal ends
    (`natives.Capture`): its exit, output, whether it expired, a stall's stacks, the files it opened and
    wrote, and what a red row of it says of its evidence: its capture's (`Capture.evidence`), and among the
    task's artifacts, named `name`, a stall's record, or a red run's whole output with the native stacks where
    a signal ended it or a process of it (a worker its test started), as one record (`stalls.kept`)."""
    spy = Path(f".venv-{python}/bin/py-spy")
    capture = natives.Capture(spy, dict(environment))
    try:
        returncode, output, expired, dump, opened, wrote = ran_pytest(
            command, {**environment, **capture.variables}, deadline, grace, pid, spy)
    finally:
        stacks = capture.stacks()
    root, evidence = run_keys.armada_dir("ARMADA_ARTIFACTS", ARTIFACTS), capture.evidence(stacks)
    if expired:
        kept = stalls.kept(root, name, dump, time.time() - (time.monotonic() - began))
    elif returncode != 0 and (returncode < 0 or "ended by signal" in stacks):
        kept = stalls.kept(root, f"signalled-{name}", f"{output}\n== native stacks\n{stacks}",
                           time.time() - (time.monotonic() - began))
    else:
        return returncode, output, expired, dump, opened, wrote, evidence
    return returncode, output, expired, dump, opened, wrote, {**evidence,
                                                              "artifacts": [str(kept.relative_to(root))]}


def unrun(python: str, tests: list[str], group: str, site: re.Match[str] | None, cache: str, common: str,
          network: bool, kept: Mapping[str, tuple[str, str]] | None,
          judged: Callable[[list[dict]], list[dict]] | None, deadline: float,
          reading: "Reading | None" = None, afford: bool = True) -> tuple[list[dict], list[str]]:
    """`tested`'s rows it need not run (reused, no site's map task; kept, `afforded` where `afford`),
    graded by `judged`, and the tests it must, from `reading`; a reused row named the commit's only as its
    grade leaves it green."""
    began = time.monotonic()
    named: dict[str, tuple[str, str]] = {}
    rows = ([] if cache == "off" or (site is not None and site["start"] is not None)
            else reused_rows(python, tests, group, common, network, kept, named, reading))
    tests = [name for name in tests if row(python, name, group) not in {found["name"] for found in rows}]
    if kept and afford:
        tests, stale_rows = afforded(python, tests, group, kept, deadline - (time.monotonic() - began))
        rows += stale_rows
    rows = judged(rows) if judged else rows
    if token := os.environ.get("DEW_XLA_CACHE_TOKEN"):
        unnamed = {found["name"] for found in rows if found["name"] in named and found["exitCode"] == 0
                   and not run_keys.baselined(python, found["name"], named[found["name"]][0], token, found)}
        for found in rows:
            if found["name"] in named and found["exitCode"] == 0 and found["name"] not in unnamed:
                run_keys.named_at_commit(python, found["name"], *named[found["name"]], token)
        rows = unbaselined(python, rows, unnamed)
    return rows, tests


def unstarted(out: Path, rows: list[dict], python: str, tests: list[str], group: str) -> int:
    """`rows`, and each of `tests` stale (never red for time not given), where the task's time went first."""
    why = "its task's time was spent before pytest began (its reuse and epochs read), so it ran no test"
    out.write_text(json.dumps({"rows": rows + [stale_row(python, name, group, why) for name in tests]}))
    return 0


def epochs_first(python: str, tests: list[str], group: str, common: str, network: bool, ends: float,
                 recording: bool) -> Mapping[str, str | None] | None:
    """A recording run's rows' epochs (`epochs_of`) within EPOCH_SECONDS and its task's end (`ends`)."""
    if not recording:
        return None
    return epochs_of(python, [row(python, name, group) for name in tests], common,
                     run_keys.device_class(network), min(time.monotonic() + EPOCH_SECONDS, ends))


def tested(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str, grace: float,
           common: str, environ: Mapping[str, str], coverage: Path | None = None,
           network: bool = False, kept: Mapping[str, tuple[str, str]] | None = None, recorded: bool = True,
           variant: str = "", judged: Callable[[list[dict]], list[dict]] | None = None,
           written: list[set[str]] | None = None, recycle_at: int | None = None) -> int:
    """`task` on the inputs `common` hashes, its pytest under `environ` beside this process's, a green row
    reused as the cache's policy (`policy`) allows. With `network`, the network-marked tests alone, with the
    Hub open, recorded on the network lane's class (`device_class`). A check-1
    site's map task (`session.SITE`) is never reused: its draws, among its artifacts, are what its judge
    reads. A run that is no test of its files (a GPU lane's prepare, `recorded` false) records no row. A
    process of the files under other flags than the lane's (`variant`, a GPU lane's `VARIANTS`) runs the
    same cut of them, its rows and their records named apart (`varied`); `judged` grades its rows first.
    `written`: each earlier process's checkout writes, read by none of its records; it adds its own."""
    began = time.monotonic()
    ends = began + deadline  # The task's one end: its reuse's lookups, its epochs and its pytest within it.
    group = varied(split, variant)
    site = session.SITE.fullmatch(group)
    rows, tests = unrun(python, tests, group, site, cache, common, network, kept, judged, deadline)
    if not tests:
        out.write_text(json.dumps({"rows": rows}))
        return 0
    log, collection, report, unrecorded, pid = task_files(out)
    # coverage.py follows the processes the tests start (pyproject.toml's patch), into COVERAGE_FILE.
    measured = [] if coverage is None else ["coverage", "run"]
    environ = environ if coverage is None else {**environ, "COVERAGE_FILE": str(coverage / ".coverage")}
    run = session.selected(session.pytest_run(python, report, *measured), network, site is not None)
    # A resuming task sets aside the tests its recalled attempt passed (`resumed`); a site's map task, a trace
    # and a run that records none (a prepare's, whose tests still prepare into this store) resume none.
    resumes = recorded and site is None and coverage is None and not variant.startswith(prepares.PREPARE)
    done_before, cold_before, peak_before = (progress.resumed(tests, grading.owning, not group) if resumes
                                             else ({}, set(), None))
    (aside := out.with_name(f"{out.name}.set-aside.json")).write_text(json.dumps(sorted(done_before)))
    # A traced run's rows, timed under tracing, are no rows to reuse, nor the census's, nor a resumed one's,
    # whose earlier attempt's opens no record holds; no other read their epochs (`epochs_first`).
    recording = (recorded and coverage is None and not (environ and check1_wire().CENSUS in environ)
                 and not done_before)
    began_in = epochs_first(python, tests, group, common, network, ends, recording)
    # A dry prepare's pytest past `recycle_at` resident ends between two tests (recycle.py), and the tests
    # left run in a fresh process: only a process's end returns what its tests compiled. Each such
    # process's cases kept (`earlier`), the tests they reported set aside from the next (progress.py's
    # SET_ASIDE).
    recycled = out.with_name(f"{out.name}.recycled")
    # What its first process selected (recycle.py's `SELECTED`), written once: every test the row must report.
    selected = out.with_name(f"{out.name}.selected")
    selected.unlink(missing_ok=True)
    sets_aside = resumes or recycle_at is not None
    command = [*session.pytest_command(run, tests, split), *(["-p", "progress"] if sets_aside else []),
               *(["-p", "recycle"] if recycle_at is not None else [])]
    earlier: list[ET.Element] = []
    outputs: list[str] = []
    segments = 1
    with temp_roots.temp_root() as root:
        environment = task_environment(environ, site, root, (log, collection, unrecorded, pid),
                                       aside if sets_aside else None, group, network)
        if recycle_at is not None:
            environment = {**environment, recycle.RECYCLE_AT: str(recycle_at),
                           recycle.RECYCLED: str(recycled), recycle.SELECTED: str(selected)}
        while True:
            # Each process's own sidecars: none an earlier one left reads as this one's.
            for sidecar in (recycled, report, pid, collection):
                sidecar.unlink(missing_ok=True)
            # What is left of the task's end, read once as its pytest starts (`unstarted` where none is).
            if (left := ends - time.monotonic()) <= 0:
                if not earlier:
                    return unstarted(out, rows, python, tests, group)
                # Its time spent before the tests left could start: they are the deadline's.
                expired = True
                break
            returncode, output, expired, dump, opened, wrote, evidence = watched_pytest(
                command, environment, left, grace, pid, python, evidence_name(python, tests, group), began)
            done, _ = reported_cases(report)
            # Ended by recycle.py alone: pytest's interrupted exit, its report holding the test it ended
            # after. Any other end (a crash after its report, a kill) is graded as it is, never run past.
            stopped = recycled.read_text() if recycled.is_file() else None
            if (expired or stopped is None or returncode != INTERRUPTED
                    or stopped not in {grading.node_of(case) for case in done}):
                break
            earlier += done
            outputs.append(output)
            segments += 1
            aside.write_text(json.dumps(sorted({*done_before, *(grading.node_of(case) for case in earlier)})))
            environment = {**environment, recycle.CONTINUES: "1"}
    output = "\n".join([*outputs, output])
    # This attempt's peak, and an earlier attempt's where it ran tests this row carries (`progress.resumed`).
    peaks = [found for found in (progress.task_peak(host), peak_before) if found is not None]
    peak = max(peaks, default=None)
    # Its last process's report is graded as a run of the tests left, by its own cases alone (`terminal`): an
    # earlier process's cases are no evidence it ran them. Each earlier process's failures are added after.
    terminal, reported = reported_cases(report)
    cases = earlier + terminal
    # How collection found each file: a file with no row is green only where every test it holds was set aside
    # by the selection, as pytest reported it (executed.py); a run that wrote no report greens nothing.
    counts = read_json(collection) or {}
    ran = []
    stood = followed.visited(followed.log_events(log)) if expired or not reported else []
    # The file its tests were in where it stopped; a process of one file stalled before any test, that file.
    running = stood[-1] if stood else (grading.owning(tests[0]) if expired and len(tests) == 1 else None)
    shares = task_shares(tests, cases, time.monotonic() - began, running if expired else None, expired,
                         grading)
    # Where the task ran out of time, the file whose test started last is the one it was spent in (executed.py
    # logs each visit): its row is the deadline's red; a file pytest reported is graded by its own cases, the
    # exit the interrupt's, not its; and one with none is not graded but run again (`grading.STALE`), never
    # a red of its tests (Main, 2026-11-09: train 745 filed six files packed beside a hung one as reds
    # "at 0.0 s").
    # A task stalled before any test started (its collection) has no such file: each row is the deadline's.
    # A row named by a test node (`file::test`) is graded by its own cases (`grading.of`), its file's
    # collection and visits its own: a GPU lane runs each such row in a task of its own (`lane_units`).
    for name in tests:
        own = [case for case in terminal if grading.of(case, name)]
        red, failed, cause = grading.graded(name, own, terminal, counts.get(grading.owning(name), {}),
                                            reported, returncode, deadline if expired else None,
                                            running, pid.is_file(), running)
        if segments > 1:
            # Run past a recycle: every test its first process selected (recycle.py's `SELECTED`) reported by
            # one of its processes, whatever a later collection found (a test a later import no longer
            # defines, or one a later collection dropped unreported, never ran).
            file = grading.owning(name)
            chosen = {node for node in (read_json(selected) or []) if grading.owning(node) == file}
            reported_all = {grading.node_of(case) for case in [*earlier, *own] if grading.of(case, name)}
            if not chosen or (never_ran := sorted(chosen - reported_all)):
                red.append("its processes, run past a recycle, reported not every test its first selected: "
                           + (", ".join(never_ran[:6]) if chosen else "its first process's selection unread"))
        if before := [case for case in earlier if grading.of(case, name)]:
            # Its earlier processes each reported, recycled (`tested`'s loop): their failures are its too,
            # and their cases its tests, timings and cells beside the last's.
            red_before, failed_before, _ = grading.graded(name, before, earlier, {}, True, 0, None, None)
            red, failed, own = red_before + red, failed_before + failed, before + own
        # A file that never started in a task that ran out of time has no seconds (`deadline_shares`): its
        # row is `unobserved`, and armada weighs no plan by it.
        unobserved = name not in shares
        seconds = shares.get(name, 0.0)
        # armada times a file at the sum of its rows' timings, a split file's groups together, each
        # Python's apart; a check-1 site's judge apart from its file, its pooling no part of the file's tasks.
        ran.append({"name": row(python, name, group), "exitCode": 1 if red else 0, "seconds": seconds,
                     # A stall's stacks whole, then the end of what pytest wrote; a red's capture and its
                     # stall's or signal's record among its artifacts (`watched_pytest`).
                     "output": "\n".join(red + ([dump[:12000]] if dump else [])
                                         + ([output[-12000:]] if red else [])),
                     **(evidence if red else {}),
                     # Its first failed test's exception as pytest gave it ("RuntimeError: ...").
                     **({"first_failure": failed[0][2].get("message") or ""} if failed else {}),
                     # Where the task ran out of time, a file that never started: no seconds to time it by.
                     **({"unobserved": True} if unobserved else {}),
                     # A pytest a signal ended, its file's red whole: the signal, for whoever it routes to.
                     **({"signal": -returncode} if red and returncode < 0 and not expired else {}),
                     "tests": {**{node: seconds for node, seconds in done_before.items()
                                  if grading.holds(name, node)},
                               **{grading.node_of(case): float(case.get("time") or 0) for case in own}},
                     # The tests that compiled a program the shared cache lacked (`grading.compiled_cold`),
                     # an earlier attempt's among them (`progress.resumed`): their seconds are a cold run's,
                     # which the durations record keeps apart (`folded_entries`).
                     **({"cold": cold} if (cold := sorted(set(grading.compiled_cold(own)) | {
                         node for node in cold_before if grading.holds(name, node)})) else {}),
                     "ynn_switched": grading.ynn_switched(own), "cells": grading.flow_cells(own),
                     # The task's resident peak, its tests' bound of their memory (`task_peak`).
                     **({"peak": peak} if peak is not None else {}),
                     # A judge's, beside its pooling, what its map tasks' records say they paid
                     # (`site_costs`).
                     "timings": {row(python, name, group if site or variant.startswith(prepares.PREPARE)
                                     else ""): seconds,
                                 **(site_costs(python, name, site["test"]) if site and site["start"] is None
                                    else {})}, **cause, **asked(site),
                     **later_judged(site)})
    ran = judged(ran) if judged else ran
    out.write_text(json.dumps({"rows": rows + ran}))
    if resumes:  # Sealed once graded: a recall sets a whole file's passes aside where it graded green alone.
        progress.ended(read_text(pid), int(any(found["exitCode"] for found in ran)))
    unlisted = set(json.loads(unrecorded.read_text())) if unrecorded.is_file() else set()
    carried = set(wrote).union(*(written or []))
    if written is not None:
        written.append(set(wrote))
    # A row run in more than one process keeps no key, as a blind watch's (`opened` None): what each read is
    # no one record's. Its red still begins its next epoch and its green still writes its baseline.
    if recording and (unnamed := run_keys.record(python, ran, log, common, network,
                                                 opened if segments == 1 else None, unlisted, carried,
                                                 began_in)):
        out.write_text(json.dumps({"rows": rows + unbaselined(python, ran, unnamed)}))
    return 0


INTERRUPTED = 2
"""pytest's exit where a plugin stopped its session (`session.shouldstop`, pytest's ExitCode.INTERRUPTED)."""


def task_environment(environ: Mapping[str, str], site: re.Match[str] | None, root: str,
                     files: tuple[Path, Path, Path, Path], aside: Path | None, group: str,
                     network: bool) -> dict[str, str]:
    """The environment of the pytest `tested` starts under `environ`: its task's `files` (`task_files`: the
    recorder's log, collection's, the unrecorded files' and its process id), its temporary root `root`, and
    `aside`, the tests a resumed attempt sets aside, where it resumes; with `network`, the Hub open."""
    log, collection, unrecorded, pid = files
    environment = session.pytest_environment({
        **environ, **site_environment(site), "DEW_EXECUTED_LOG": str(log), "PYTEST_DEBUG_TEMPROOT": root,
        "DEW_COLLECTED": str(collection), "DEW_UNRECORDED": str(unrecorded), "DEW_PYTEST_PID": str(pid),
        "DEW_STACKS_READER": str(os.getpid()), **({progress.SET_ASIDE: str(aside)} if aside else {}),
        **({session.SPLIT_EVENLY: "1"} if prepares.prepared_group(group, VARIED) else {})})
    if network:
        return {key: value for key, value in environment.items() if key not in session.CLOSED_HUB}
    return environment


def reported_cases(report: Path) -> tuple[list[ET.Element], bool]:
    """The test cases of a pytest's JUnit `report`, and whether it wrote one: a report its killed pytest left
    cut short is none, its files graded as if none was written (`tested`)."""
    try:
        return (list(ET.parse(report).getroot().iter("testcase")), True) if report.is_file() else ([], False)
    except ET.ParseError:
        return [], False


def read_json(path: Path) -> dict | None:
    """The JSON object at `path`, or None where none is or what is there is no whole object (a write its
    killed process cut short, said): no evidence, never a task's end."""
    if not path.is_file():
        return None
    try:
        found = json.loads(path.read_text())
    except json.JSONDecodeError as cut:
        print(f"ci: {path} holds no whole JSON ({cut}): read as none", flush=True)
        return None
    return found if isinstance(found, dict) else None


def read_text(path: Path) -> str | None:
    """`path`'s text stripped, None where it has none."""
    return path.read_text().strip() if path.is_file() else None


def untracked() -> set[str]:
    """The checkout's entries its repository neither tracks nor ignores (none where it is no repository):
    each file, and each such directory as `name/` (an empty one among them)."""
    found: set[str] = set()
    for collapsed in ([], ["--directory"]):
        listed = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", *collapsed, "-z"],
                                capture_output=True, text=True, check=False)
        found |= set(listed.stdout.split("\0")) - {""} if listed.returncode == 0 else set()
    return found


def restored(wrote: set[str], before: set[str], out: Path) -> None:
    """The checkout as its commit holds it again after a process that wrote `wrote` (`tested`'s), so the next
    process runs on the commit's files, not on what this one left (row 3932's review: a file rewriting a
    later file's source, or making a conftest.py or a link under it, ran it green on code the commit does
    not hold): each tracked file it wrote checked out again, and each entry the checkout neither tracks nor
    ignores that was not there before it (`before`, `untracked`) removed, then each directory it made that
    this leaves empty (one imports as a namespace package), but the task's own outputs: its verdict `out`,
    each process's beside it (`out.N` and theirs) and its artifacts (ARMADA_ARTIFACTS), which outlive it."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", *sorted(wrote)], capture_output=True, text=True,
                            check=False) if wrote else None
    tracked = set(listed.stdout.split("\0")) - {""} if listed and listed.returncode == 0 else set()
    if tracked:
        subprocess.run(["git", "checkout", "-q", "--", *sorted(tracked)], check=True)
    artifacts = os.environ.get("ARMADA_ARTIFACTS")
    roots = [Path(artifacts).resolve()] if artifacts else []
    verdicts = out.resolve()
    ours = re.compile(rf"{re.escape(verdicts.name)}(\.\d+(\..+)?)?")
    checkout = Path.cwd().resolve()
    emptied: set[Path] = set()
    for made in sorted(untracked() - before):
        # The entry itself, a link among them, not what it names; a directory's own entry (`name/`) aside.
        place = Path(made).parent.resolve() / Path(made).name
        beside = place.relative_to(verdicts.parent).parts[0] if verdicts.parent in place.parents else None
        if made.endswith("/") or any(place == root or root in place.parents for root in roots) or \
                (beside is not None and ours.fullmatch(beside)):
            continue
        Path(made).unlink(missing_ok=True)
        emptied.update(parent for parent in place.parents if checkout in parent.parents)
    for directory in sorted(emptied, key=lambda each: len(each.parts), reverse=True):
        if f"{directory.relative_to(checkout).as_posix()}/" not in before and not any(directory.iterdir()):
            directory.rmdir()


def gpu_tier(python: str) -> dict[str, list[str]]:
    """The test files that take another path on a GPU, each with its evidence: where its tests read the
    backend and its marks of `session.MARKS`, over every record of its rows, on every class, a split file's
    under each count of groups it was planned as. A file is out only where every record is of these inputs
    (`fresh`), a CPU task's (`shared`) or a GPU lane's (`lane_inputs`), and says neither, so a file whose
    inputs changed since it ran is in as "changed", and one without a record, or with one the Worker did not
    return, as "unrecorded": a new file, or one no green run of these tools has recorded."""
    token = os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        raise SystemExit("the GPU tier is chosen from the test cache's records, which need its URL and token")
    inputs = (shared(python), lane_inputs(python))
    prefix = f"{CACHE_PREFIX}/{python}/"
    # One reading of the checkout, which every record's key reads its inputs from.
    reading = run_keys.Reading()

    def evidence_of(key: str) -> tuple[str, list[str]]:
        _, name, _ = key_parts(key)
        # A test node's record (a GPU lane's, `rows.row_place`) is no file's evidence: its file's runs are.
        if "::" in name:
            return name, []
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
    every = test_files()
    unrecorded = {name: ["unrecorded"] for name in every if name not in evidence}
    return {name: sorted(evidence[name]) for name in every if evidence.get(name)} | unrecorded


def marked(python: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Each test file of this checkout with tests marked one of `session.MARKS`, and how many, as pytest
    collects it on the CPU: what a batch's GPU run holds, whatever the records say; and each such file's
    GPU generations, those its marked tests are pinned to (the mark `generations`, pins.py), none where a
    test of it runs on every one."""
    return collection.collected(python, " or ".join(session.MARKS), test_files(), pinned=True)


def test_files() -> list[str]:
    """The checkout's test files CI's tasks run (`SKIPPED` run apart)."""
    return sorted(str(path) for path in Path("tests").glob("test_*.py") if str(path) not in SKIPPED)


def network_files(python: str) -> dict[str, int]:
    """Each test file of this checkout with tests marked network, and how many: what the network lane runs
    (`gpu_lane`)."""
    return collected(python, "network")


def collected(python: str, marks: str, files: list[str] | None = None,
              within: float | None = None) -> dict[str, int]:
    """Each test file of this checkout (of `files`, where given) with tests pytest's `-m marks` selects, and
    how many, as pytest collects it on the CPU (collection.py): of `files`, one that does not collect has
    none and stops no other's reading, within `within` seconds where a lane has a deadline."""
    return collection.collected(python, marks, test_files() if files is None else files, within,
                                tolerant=files is not None)[0]


def largest_records(python: str, token: str, measures, under: str = "", strict: bool = False) -> dict:
    """For each thing `measures` says a record's class, test file and group (`key_parts`) measures (None:
    nothing asked for), of this Python's last records under `under` (a class's `<class>/`; none, every
    class's), the most those whose runs measured it held, as one record (measures.py's `largest`); none
    where the Worker does not list its records. With `strict`, a read the Worker does not answer is an
    `r2.R2Error`, never a file taken for unmeasured: what sizes a job before it runs (gpu_holds.py) is read
    whole or not at all (2026-11-11: one record's read did not come back as a GPU tier was composed, its part
    took the default for a file measured at 5.9 GiB, and the file ran out of a 4 GiB lane)."""
    prefix = f"{CACHE_PREFIX}/{python}/"
    try:
        keys = [under + key for key in listed(prefix + under, token)]
    # The Worker did not list its records: a measure only packs a lane, so each such file runs alone with the
    # lane's whole share, as one measured on no class does.
    except r2.R2Error as error:
        if strict:
            raise
        print(f"the test cache did not list {prefix}{under}: {error}; files unmeasured here run alone",
              file=sys.stderr)
        return {}

    def read(key: str) -> dict | None:
        if strict:
            found = store(token).get(f"{prefix}{key}")
            return None if found is None else json.loads(found)
        return recorded(f"{prefix}{key}", token)

    return gpu_measures.largest(keys, read, lambda key: measures(*key_parts(key)))


def borrowed_records(names: list[str], python: str, held: str, token: str,
                     strict: bool = False) -> dict[str, dict]:
    """For each of `names`, test files no run on class `held` measured, the most its last records on the
    other classes measured (`largest_records`), any group's and any of its processes' (`varied`: a file's
    process under a user's flags may hold more than the lane's own): what a file has taken on a GPU, which
    a first lane on a class takes as its share. One no class measured is absent. Its run on `held`
    records its own, which every later lane reads."""
    wanted = set(names)
    # A test node's records (`file::test`) count as its file's.
    return largest_records(python, token, lambda at, name, _: (
        grading.owning(name) if at != held and grading.owning(name) in wanted else None), strict=strict) \
        if names else {}


def prepare_shared(python: str) -> str:
    """What every dry prepare's row shares: every row's (`shared`) and the prepare's own recipe
    (`prepares.RECIPE`), which runs beside what its tests record."""
    digest = hashlib.sha256(shared(python).encode())
    for path in prepares.RECIPE:
        digest.update(path.encode() + Path(path).read_bytes())
    return digest.hexdigest()


def prepared_row(name: str) -> bool:
    """Whether a verdict's row `name` is a dry prepare's (`dry_prepared`), no run of its file's tests."""
    return ":" in name and prepares.prepared_group(parsed_row(name)[2], VARIED)


def dry_prepared(python: str, tests: list[str], split: str, out: Path, deadline: float, cache: str) -> int:
    """A task's dry prepares of GPU tier files on this CPU (prepares.py's `run`)."""
    return prepares.run(sys.modules[__name__], python, tests, split, out, deadline, cache)


def lane_units(python: str, tests: list[str], within: float | None = None) -> dict[str, tuple[str, str, str]]:
    """Each process a GPU lane runs for `tests`, by its name, as its file, its group and its variant: each
    test's own, and after it one for each of `VARIANTS` whose marked tests its file holds, as pytest collects
    the lane's files on the CPU, once a mark. A file with none of a variant's tests starts no process for it;
    one that holds some never goes without it (a variant no lane ran is tests that ran nowhere, silently)."""
    files = sorted({name.partition("#")[0] for name in tests})
    holding = {variant: collected(python, mark, files, within) for variant, (mark, _) in VARIANTS.items()}
    units = {}
    for name in tests:
        file, _, group = name.partition("#")
        units[name] = (file, group, "")
        # A test node's (`file::test`) collection counts under its file.
        units |= {f"{file}#{varied(group, variant)}": (file, group, variant)
                  for variant, files_ in holding.items() if grading.owning(file) in files_}
    return units


def gpu_prepared(python: str, tests: list[str], out: Path, *, processes: int, deadline: float) -> int:
    """A GPU lane's prepare, on its node's CPU before its files' GPU run (`gpu_lane`) and with no GPU drawn:
    each of `tests` run once in a process of its own on the CPU alone, under the prepare's word
    (tests/tier_wire.py's PREPARE, under which tests/conftest.py has each test prepare its check-1 sites'
    truths and arms for the node's GPU rather than run, one that prepares none skipped) and its unit's name
    (UNIT): every test of the part, so a site its run reaches never goes unprepared, and its run calls stored
    executables and compiles nothing while it holds the GPU (Main, 2026-11-17: the day's GPU jobs left 2034 of
    3238 held GPU-minutes idle, tracing and compiling on the host), at most `processes` at once, as its run
    will select and cut it (a group `<file>#G/N` holds in its prepare the tests it holds in its run: the same
    selection, the same weights, DEW_SPLIT_DURATIONS), so each such test computes what its run reads and must
    not compute while it holds the GPU, its float64 truths and its programs compiled for the GPU the node's
    topology names (DEW_NODE_TOPOLOGY), into the store its environment names (DEW_TIER_TRUTHS), which the run
    is given too. Their rows are written to `out` and nothing is recorded, no test having run to its end: a
    prepare that fails is said in its row and exit; its run is red for what it lacks, never computing it on
    the GPU. A file's process under other flags (`lane_units`) prepares apart, under them, as it runs. First
    the pinned fixtures the tests read offline are fetched, as the run fetches them
    (tools/fetched_fixtures.py), so the run finds each whole and waits on no download with the GPU held.
    `deadline` is the whole prepare's: each process is given what is left of it as its turn comes, its own
    end's allowance kept back, so the prepare ends within it however many files and waves it holds, and a file
    it leaves no time is a red row of its own, never started (review: a deadline a process let five files
    outlast the job's own prepare bound, whose session then ended the job and every file's prepare with it);
    its fetch and its collection are within it too."""
    began = time.monotonic()

    def left() -> float:
        return deadline - (time.monotonic() - began)

    subprocess.run([f".venv-{python}/bin/python", "tools/fetched_fixtures.py", "fetch"], check=True,
                   timeout=max(1.0, left()))
    units = lane_units(python, tests, max(1.0, left()))
    verdicts = {name: out.with_name(f"{out.name}.{index}") for index, name in enumerate(units)}
    threads = str(max(1, len(os.sched_getaffinity(0)) // processes))

    def prepared(name: str, verdict: Path) -> int:
        file, group, variant = units[name]
        return prepares.process(sys.modules[__name__], python, file, group, verdict, left(), unit=name,
                                threads=threads, users=bool(variant), variant=variant)

    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        for done in [pool.submit(prepared, name, verdict) for name, verdict in verdicts.items()]:
            done.result()
    rows = [row_ for verdict in verdicts.values() for row_ in json.loads(verdict.read_text())["rows"]]
    out.write_text(json.dumps({"rows": rows}))
    return int(any(row_["exitCode"] != 0 for row_ in rows))  # Row 9541: an uncollected file's exit 0 hid it.


def lane_measures(python: str, units: dict[str, tuple[str, str, str]], held: str,
                  token: str) -> dict[str, dict | None]:
    """What each of a GPU lane's processes (`lane_units`: its file, group and variant) has held on class
    `held`, a record `gpu_needs` reads: its last run's there, a group the tier cut the most its file's runs
    measured there; else its file's on the other classes (`borrowed_records`); None where no run measured
    it."""
    # A group the tier cut by its own seconds holds tests another cut's group of its name did not, so its own
    # last record is not its measure (2026-11-10: a group measured holding two tests was given their pool when
    # it held five, one asking more than the pool): it reads its file's whole run's.
    cut = bool(os.environ.get("DEW_SPLIT_DURATIONS"))

    def needs(name: str) -> dict | None:
        file, group, variant = units[name]
        found = recorded(record_key(python, row(python, file, varied("" if cut else group, variant)), held),
                         token)
        # A test node (`file::test`) no run measured holds no more than its file's process.
        if found is None and (whole := grading.owning(file)) != file:
            found = recorded(record_key(python, row(python, whole, varied("", variant)), held), token)
        return found

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        records = dict(zip(units, pool.map(needs, units), strict=True))
    # Nor is the whole run alone such a group's measure, which a test added since has outgrown (2026-11-11: a
    # whole run's 3.92 GiB gave a newer test a 5.9 GiB lane, which ran out): it is the most the runs of its
    # file's process on this class measured, whole or a group of any cut. A whole run that measured nothing (a
    # child it did not follow, no report) stays the file's, which so takes the lane's whole share.
    if cut:
        files = {grading.owning(file) for file, _, _ in units.values()}
        # A test node's records (`file::test`) among its file's: the most any of its runs held.
        most = largest_records(python, token, under=f"{held}/", measures=lambda _, name, group: (
            (grading.owning(name), place_variant(group)) if grading.owning(name) in files else None))
        records = {name: found if found is not None and not gpu_measures.measured(found)
                   else most.get((grading.owning(units[name][0]), units[name][2]), found)
                   for name, found in records.items()}
    # A group no run on this class measured borrows the most its file held on another, which holds at least
    # its own; a file's process under other flags, its tests a part of the file's, the lane's own's first.
    borrowed = borrowed_records(sorted({grading.owning(units[name][0]) for name, found in records.items()
                                        if found is None}),
                                python, held, token)
    return {name: records[name] or records.get(f"{file}#{group}" if group else file)
            or borrowed.get(grading.owning(file)) for name, (file, group, _) in units.items()}


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

    A file's process is given the share its last run on this GPU's class measured, green or red (a group
    the tier cut, the most any run of its file on the class did), and files start heaviest first while
    what they hold with their children (`gpu_needs`) fits the share this process was given
    (`XLA_PYTHON_CLIENT_MEM_FRACTION`, as gpu-run sets it), at most `processes` at once. A file that runs
    out of a measured share is red: it took more than its last run, whose peak its next run is given. A
    file no run on this class measured borrows its share from the largest any other GPU class's last run
    measured (`borrowed`), so a first lane on a class packs its files as a measured one does; one red
    under it is red, run once, as every red is (owner, 2026-11-07: a red under sharing is a red), and its
    measured peak is the next lane's. A file given the lane's whole share (measured on no class, measured
    that large, or with a child the lane did not follow, as the fla reference's in another environment's
    Python) grows its pool as it uses it rather than taking the share at its first GPU op
    (`XLA_PYTHON_CLIENT_PREALLOCATE`), so such a child has what the file has not yet taken; a file that
    grows to the whole share still leaves its child none. Each file's process takes its share of the CPUs
    this process may use, as of the files that run at once (`OMP_NUM_THREADS`): a torch reference arm's
    small ops on all of a node's 30 CPUs ran 13 times slower than on 4 (test_gemma3n's gradient reference
    on an A100 node, 1175 s against armada's 326 s), and the more so as files share them.

    First the pinned fixtures the tests read offline are fetched while the Hub is reachable
    (tools/fetched_fixtures.py): a store file already whole is kept, and a pin whose upload is due stops
    the lane before any file runs."""
    subprocess.run([f".venv-{python}/bin/python", "tools/fetched_fixtures.py", "fetch"], check=True)
    token = os.environ["DEW_XLA_CACHE_TOKEN"]
    # The memory the GPU gives a process, less what its driver and firmware keep: the total XLA and a node's
    # session take this lane's fraction of (within nvidia-smi's MiB). Of the GPU's whole memory, a share was
    # a pool short by what the driver keeps (2026-11-11: a GH200's 1,103 of 97,871 MiB made a 5.91 GiB share
    # a 5.84 GiB pool, which ran out).
    [total] = ((int(whole) - int(kept)) * 2 ** 20 for whole, kept in
               (answer.split(",") for answer in run_keys.gpu_query("memory.total,memory.reserved")))
    # A job holding the whole GPU is given no fraction (a node sets one for a part of a GPU alone, its
    # session's settings): its processes may take what one process takes by JAX's own default (row 2570).
    budget = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", WHOLE_GPU)) * total
    held = run_keys.device_class(network)
    threads = str(max(1, len(os.sched_getaffinity(0)) // processes))

    # Each process the lane runs (`lane_units`): a test's file and group (`<file>#G/N`; none, a whole
    # file), and its file's process under other flags, where it holds tests for one.
    units = lane_units(python, tests)
    tests = list(units)

    measures = lane_measures(python, units, held, token)
    prepared = gpu_measures.prepared_needs(python, tests, tier_wire())  # No less than its programs' need.
    unheld = gpu_measures.unschedulable(prepared, budget, float(WHOLE_GPU) * total, held)  # Nor below it.
    tests = [name for name in tests if name not in unheld]
    needed = {name: gpu_needs(measures[name], budget, prepared.get(name, 0.0)) for name in tests}
    shares = {name: share for name, (share, _) in needed.items()}
    holds = {name: held for name, (_, held) in needed.items()}
    short = gpu_measures.measured_short({name: measures[name] for name in tests}, budget)
    common = lane_inputs(python)  # Before any test writes under tests/, the inputs every file's key hashes.
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
                running[pool.submit(tested, python, [file], group, verdicts[fits], deadline, cache,
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
    rows += [stale_row(python, file, varied(group, variant), unheld[name])
             for name, (file, group, variant) in units.items() if name in unheld]
    out.write_text(json.dumps({"rows": rows}))
    return 0


def verdict_rows(verdict: dict) -> dict[str, list[dict]]:
    """Main's CI `verdict`'s rows by test file, as its floor Python ran each file's groups or whole: no
    collection's, site's or variant's row, and none of a file the checkout no longer has or has changed since
    the verdict's commit, whose rows observed tests it may no longer hold (`folded`, `cuts.ran`)."""
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
    return rows


def folded_entries(record: dict[str, "Entry"], verdict: dict, target: float,
                   rows: dict[str, list[dict]]) -> dict:
    """The durations `record`'s entries with main's CI `verdict` folded in, its `rows` (`verdict_rows`): each
    file whole (`folded`), each cold-timed test seen at its warm record where it has one, a cold run moving
    no warm time (a test no warm run has timed takes its cold time as its warm one, which a warm run lowers),
    each test's warm time lowered first to what its green runs take (fold.py's `lowered`, so the same verdict
    folded again moves nothing), then its cold time and peak (fold.py's `entries`); each file's sites read
    once for both."""
    marks = {name: run_keys.sites(name) for name in rows}
    warm = fold.lowered(session, {node: session.warm_cold(held)[0] for node, held in record.items()}, rows,
                        marks, target)
    # A green row that is no planning evidence (fold.py's `evidence`: reused, or run alone after a failure
    # beside other tasks) and each test a row timed cold keep its record's time: they count for whether a
    # file is whole, never as its time; a red row's raises, isolated or not, as `raised` takes them (review).
    calm = {name: [{**row_, "tests": {node: warm[node] if node in warm and (
        node in row_.get("cold", ()) or (row_["exitCode"] == 0 and not fold.evidence(row_))) else seconds
        for node, seconds in row_.get("tests", {}).items()}} for row_ in held] for name, held in rows.items()}
    return fold.entries(session, record, folded(warm, verdict, target, calm, marks), rows, grading.owning)


def folded(record: dict[str, float], verdict: dict, target: float, rows: dict[str, list[dict]] | None = None,
           marks: dict[str, dict] | None = None) -> dict[str, float]:
    """The durations `record` with main's CI `verdict` (`armada verdict <sha> --json`) folded in, one test
    file at a time, each whole, as its floor Python's rows ran it. A file whose rows are all green and
    complete (its whole row, or every group of its cut) takes the times they observed where its record is
    stale (`stale`, its total drifted among it). A group's key holds its tests, not their seconds
    (`cache_key`), and its cut stays as stored while its groups hold (`session.stored_bounds`), so a refreshed
    file reruns only the groups whose tests move, and every group of it where its group count changes. A file
    with a red or missing row keeps its entries, but for what its rows timed past them (fold.py's `raised`),
    as does a file the verdict did not run, and one the checkout has changed since the verdict's commit, whose
    rows observed tests it may no longer hold (2026-10-31: train 370's fold replaced a head's 66 renamed test
    ids with main's old ones); a file the checkout no longer has loses them. A file a task's deadline cut off
    takes the times its rows observed of the tests it finished (all but the last, which the deadline stopped)
    where its record lacks them: no green run would ever time it, and the plan cuts it by them, a test it
    never reached weighing their mean (2026-11-02: base/kernels' test_algebra_laws, recorded nowhere, ran 46
    of its ~70 tests and was cut off every run). A reused row (the test cache) carries the times of the run
    that recorded it. Each train folds main's CI so (the land queue), and nothing else writes the record. A
    site its file's marks shard (`sites`) keeps its entries: its judge's row times its pooling, no measure of
    its draws, so its map tasks are weighed by armada's measures of them, never by the record
    (`site_entries`), and a judge armada did not run is a row of its task's name. Its `rows` and each file's
    sites (`marks`) are read where not handed in."""
    rows = verdict_rows(verdict) if rows is None else rows
    kept = {node: seconds for node, seconds in record.items() if Path(grading.owning(node)).is_file()}
    for name, held in rows.items():
        splits = {parsed_row(row_["name"])[2] for row_ in held}
        counts = {int(split_.split("/")[1]) for split_ in splits if split_}
        complete = splits == {""} or (len(counts) == 1 and splits == {f"{group}/{count}" for count in counts
                                                                      for group in range(1, count + 1)})
        marked = marks[name] if marks is not None else run_keys.sites(name)
        if not complete or any(row_["exitCode"] != 0 or "tests" not in row_ for row_ in held):
            kept.update(fold.raised(session, kept, held, marked))
            continue
        observed = {node: seconds for row_ in held for node, seconds in row_["tests"].items()}
        own = {node: seconds for node, seconds in kept.items() if grading.owning(node) == name
               and session.site_name(node, marked) not in marked}
        took = sum(row_["seconds"] for row_ in held)
        if cuts.stale(session, own, observed, took, max(counts, default=1), target):
            kept = {node: seconds for node, seconds in kept.items() if node not in own}
            kept.update({node: round(seconds, 3) for node, seconds in observed.items()})
    return dict(sorted(kept.items()))


def coverage_items(target: float, timings: dict, scenarios: bool = False) -> list[str]:
    """The coverage pass's armada map items (`coverage_report.items`): the plan's tasks for the oldest Python,
    but its collections, a check-1 site's judge, which pools its map tasks' draws and has none to pool here,
    and a dry prepare, which runs no test of its file (`dry_prepared`)."""
    entries = [entry for entry in plan(target, timings)
               if entry["python"] == PYTHONS[0] and entry["split"] != COLLECT and "after" not in entry
               and not entry["tests"].startswith(prepares.TOKEN)]
    return coverage_report.items(entries, ARTIFACTS, scenarios)


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
    gates = [line.split("\t", 1) for line in (REPO / "tools" / "check_gates").read_text().splitlines()
             if line.startswith("collect-")]
    collected = tuple(name.removeprefix("collect-") for name, command in gates
                      if f".venv-{name.removeprefix('collect-')}/" in command)
    if collected != PYTHONS or len(gates) != len(PYTHONS):
        problems.append(f"tools/check_gates collects on {collected}; the tested versions are {PYTHONS}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def slow(verdict: dict, target: float, base: dict[str, float] | None = None) -> list[dict]:
    """Each test of `verdict`'s rows (`armada verdict <sha> --json`) whose own seconds pass `target`, the
    group target, slowest first: no cut brings its group within the target (bible §7: every test past the
    target is reported). With `base` (main's durations record), only the tests it lacks: a new slow test
    (owner, 2026-10-28). A check-1 site's map task past the plan's bound on one (`cut_draws`: four
    targets) is reported whatever the record holds (its test is recorded whole): its mark gives a task too
    many units, which fewer a task (per_task, conditioning_per_task) bring within it (row 3729: 1,366 s). A
    dry prepare's row (`dry_prepared`) times its tests' prepares, not their runs, and reports none."""
    found = []
    for row_ in verdict["rows"]:
        if "name" in row_ and prepared_row(row_["name"]):
            continue
        drawn = session.SITE.fullmatch(parsed_row(row_["name"])[2]) if "name" in row_ else None
        mapped = drawn is not None and drawn["start"] is not None
        most = target * costs.SITE_TARGETS if mapped else target
        for test, seconds in (row_.get("tests") or {}).items():
            if seconds <= most or (base is not None and test in base and not mapped):
                continue
            found.append({"test": test, "file": grading.owning(test), "seconds": round(seconds, 1),
                          "target": most,
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
    if args.operation in ("task", "pytest", "gpu-tier"):  # each collects tests: the pins they read, fetched
        args.deadline -= session.fetched_absent(args.python, args.deadline, digests.fetched.absent(Path()))
    if args.operation == "pythons":
        return pythons(check=args.check)
    if args.operation == "gangs":
        print(json.dumps(gangs()))
        return 0
    if args.operation == "slow":
        return flagged(args.verdict, args.new)
    if args.operation == "partition":
        nodes = partition.collected(line for listed in args.collected
                                    for line in listed.read_text().splitlines())
        wrong = partition.compared(nodes, [found for found in json.loads(args.verdict.read_text())["rows"]
                                           if not prepared_row(found["name"])], PYTHONS[0], SKIPPED)
        print(f"{sum(map(len, nodes.values()))} tests of {len(nodes)} files collected: "
              + ("each in one row" if not wrong else f"{len(wrong)} lines of rows that are not their tests"),
              *wrong, sep="\n")
        return 1 if wrong or not nodes else 0
    if args.operation == "durations":
        target = json.loads(Path(".armada.json").read_text())["target"]
        before, verdict = session.entries(), json.loads(args.verdict.read_text())
        # The record main's CI cut its files by, read before its times change it (`cuts.refolded`).
        cut_by = {name: run_keys.own_tests(name, tests) for name, tests in run_keys.parted_record().items()}
        rows_of = verdict_rows(verdict)
        recorded = folded_entries(before, verdict, target, rows_of)
        if recorded != before:
            session.DURATIONS.write_text(json.dumps(recorded, indent=0) + "\n")
        after = {name: run_keys.own_tests(name, tests) for name, tests in run_keys.parted_record().items()}
        runs = cuts.ran(rows_of)
        stored = cuts.refolded(session, session.stored_cuts(), runs, target, cut_by, after,
                               cuts.held(rows_of))
        if stored != session.stored_cuts():
            session.CUTS.write_text(json.dumps(stored, indent=0, sort_keys=True) + "\n")
        changed = sorted({grading.owning(node) for node in before.keys() ^ recorded.keys()}
                         | {grading.owning(node) for node in before.keys() & recorded.keys()
                            if before[node] != recorded[node]})
        print(f"{len(changed)} files refolded in {session.DURATIONS}: {' '.join(changed) or 'none'}")
        return 0
    if args.operation == "gpu-tier":
        # A GPU runs its marked files alone, a batch's and a day's alike (ledger 2452); --evidence prints that
        # tier, each file with its records' backend reads beside its marked tests.
        tier, pinned = marked(args.python)
        if args.evidence:
            reads = gpu_tier(args.python)
            for name, count in tier.items():
                evidence = [item for item in reads.get(name, []) if not item.startswith("mark:")]
                print(name, f"{count} marked", *evidence, sep="\t")
            return 0
        # Each shard's files on a line, spread by their tests' recorded times.
        recorded = recorded_tests()
        for _, names in packed({name: sum(recorded.get(name, [1.0])) for name in tier}, args.shards):
            print(*names)
        # And each file's pinned generations, one line of JSON: what a tier's composer cuts each kind's by.
        print(json.dumps(pinned))
        return 0
    if args.operation == "pytest":  # argparse keeps the `--` that sets pytest's arguments apart from these
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        return targeted(args.python, arguments, args.report, args.tests.split(), args.split)
    if args.operation == "gpu-lane":
        return lane_asked(parser, args)
    if args.operation == "coverage":
        if args.stage == "report":
            return coverage_report.report(args.directory)
        timings = json.loads(args.timings.read_text()) if args.timings.is_file() else {}
        print(json.dumps(coverage_items(args.target, timings, args.scenarios)))
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
    # A dry prepare's task, or its row run again by its file and its group, as the land queue runs one.
    with given_mainline(args.main):
        if args.tests.startswith(prepares.TOKEN) or prepares.prepared_group(args.split, VARIED):
            return dry_prepared(args.python, args.tests.split(), args.split, args.out, args.deadline,
                                args.cache)
        return task(args.python, args.tests.split(), args.split, args.out, args.deadline, args.cache,
                    coverage=args.coverage, census=args.census == "1")


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

"""A CI task's pytest session, as the plan (ci.py) and the plugins a task loads into it (split.py,
executed.py) share it: how pytest runs and what it sees (`pytest_run`, `pytest_environment`), how long it runs
(`DEADLINE`, `GRACE`), which tests take another path on a GPU (`MARKS`), and how a split file's tests are cut
(`cut`, `groups`), those of one shared setup together (`SHARED_SETUP`). What it decides changes a task's
outcome, so the test cache keys every file's record on it (ci.py's `SHARED`); the plan's own code does not,
and a change to it reruns nothing. The standard library alone: the plan imports it where pytest is not
installed."""

import ast
import bisect
import functools
import json
import os
import re
from collections.abc import Container, Iterator, Mapping
from pathlib import Path

DURATIONS = Path("tests/test_durations.json")
CUTS = Path("tests/test_cuts.json")
"""Each split test file's cut as main's CI last ran it, kept while the file's groups hold (`stored_cuts`): the
node id each group after the first begins at, so a group's name means the same tests from run to run whatever
their times do, and its record is found again (2026-11-13: 43% of the plan's rows had no record, a split
file's groups renamed whenever its times moved; Main). Written by the durations fold alone (ci.py
`durations`), a record the land queue owns as it owns `DURATIONS`; a second file rather than a key of that
one, whose readers all read node ids to seconds (split.py, ci.py, the queue's merge of a head's new times, a
GPU tier's record of the same shape)."""


SELECTION = ("-m", "not network")
"""Which tests CI runs: every one but those marked network, which reach the Hub (`CLOSED_HUB`)."""

CLOSED_HUB = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_ENDPOINT": "http://127.0.0.1:9"}
"""What a task's tests see of the Hugging Face Hub, which none of them may reach (`SELECTION`): the
libraries offline, and the Hub's address a closed port for any that would ask it regardless. A test
that needs the Hub is marked network; one that reads a model or dataset otherwise brings it."""

DEADLINE = 1680.0
"""Seconds a task's tests run before they are interrupted and the files they hold are red: below armada's own
task timeout (.armada.json's), so the verdict is written."""
GRACE = 60.0
"""Seconds an interrupted task's tests have to end before their process group is killed."""


BOOT = "import lifetime; lifetime.boot()"
"""How a task's Python starts (`pytest_run`): with no site (-S), the test cache's recorder first
(lifetime.py), which then makes the site and runs pytest, so it records the process's whole life."""


def pytest_run(python: str, report: Path, *measured: str) -> list[str]:
    """CI's pytest under `python`, as every task runs it: the recorder booted before the interpreter's site
    (`BOOT`), CI's `SELECTION`, failures and errors summarized with short tracebacks, a file that does not
    collect beside those that do, unbuffered (-u) so the output holds every test that finished when a run is
    cut short, and pytest's JUnit report written to `report`; `measured`, a module and its arguments that run
    pytest in turn (coverage's `run`). No cache of pytest's own (its cacheprovider), which CI never reads: its
    write makes a directory of a random name in the checkout, which the recorder saw a run probe and keyed its
    row by, so no later run's inputs matched it (2026-10-30: 192 records a docs-only change could not reuse).
    No timed dump of every thread's stack runs beside the tests, as walking threads that run killed the
    process mid-dump (exit 245, a SIGSEGV, in 3 of 5 runs of a multithreaded interpreted kernel's test,
    2026-10-25): a caller's deadline ends a hang (ci.py's `tested`: SIGINT, whose report pytest prints where
    it can handle it, then the group killed past the grace), its stacks read from outside it once as it passes
    (ci.py's `dumped`), and a fatal signal still prints the stacks (pytest's faulthandler)."""
    runner = [*measured, "-m", "pytest"] if measured else ["pytest"]
    return [f".venv-{python}/bin/python", "-S", "-u", "-c", BOOT, *runner, "-q", *SELECTION, "-rfE",
            "--tb=short", "--continue-on-collection-errors", "-p", "no:cacheprovider", f"--junitxml={report}"]


def pytest_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What CI's pytest runs under: this process's environment but what pytest reads as its own arguments
    (`UNKEYED`), `environ` over it, the Hub closed, and tools/armada on the path, where the plugins a task
    loads (split.py, executed.py) import from: what a run selects and loads is its command's alone, which
    every row's key holds (row 2558: an inherited PYTEST_ADDOPTS could reuse a green row its run would not
    give)."""
    path = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")]))
    inherited = {name: value for name, value in os.environ.items() if name not in UNKEYED}
    return {**inherited, **environ, **CLOSED_HUB, "PYTHONPATH": path}


UNKEYED = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
"""The variables pytest reads as arguments and plugins to load, which a supervisor's environment may carry
and no row's key holds: a task's pytest runs without them (`pytest_environment`)."""


SHARDS = {"check1_shards": "seeds", "shards": "units"}
"""The marks of a site too long for one task (pyproject.toml's markers), each by the count it declares:
`check1_shards(seeds=S, per_task=P)` on a test whose `assert_as_exact` sites (tests/reference_error.py)
declare S seeds, which check 1's two stages draw (1 + REDRAW) times over; and `shards(units=N,
per_task=P)` on a test whose work is N units of its own, split the same way (a witness's items). CI runs
either as armada's map and judge (ci.py's `site_entries`): map tasks of P units each over the units its
stages or its work take, and one judge."""

SITE = re.compile(r"(?P<test>(?:\w+\.)*test\w*(?:\[[\w-]+\])?)"
                  r"(?:\.draws-(?P<start>\d+)-(?P<stop>\d+))?")
"""A check-1 site's group in its rows' names (ci.py's `row`), which its tasks' split carries: its test's name
in its file (ci.py's `sites`), with its parameter's id where the mark is a `pytest.param`'s (`test[id]`), its
judge's, whose row is the site's verdict; or that and the seeds one of its map tasks draws,
`<test>.draws-<start>-<stop>`."""
PARAM_ID = re.compile(r"[\w-]+")
"""A marked parameter's id (`pytest.param(..., id=...)`), as a site's name carries it (`SITE`): letters,
digits, `_` and `-`, so its node id, its rows' names and its tasks' split hold it whole."""


def selected(command: list[str], network: bool, whole: bool = False, release: bool = False,
             site: bool = False) -> list[str]:
    """CI's pytest `command` (`pytest_run`), or with `network` the same selecting the network-marked tests
    alone, every one the rest of CI sets aside. It sets aside the tests marked whole_suite too, but on the
    whole suite's run (`whole`: DEW_WHOLE_SUITE, CI's schedule and its whole dispatch), whatever the cache's
    policy: a policy-off train runs every file, never a whole-suite cell (the maintainer, 2026-10-28: train
    250's policy-off suite ran G03's GRPO cells, 826-1,430 s each, to its deadline). The release checks run in
    their own tasks alone (`release`), every one whatever else marks it, under a requirement the suite's
    environment lacks (ci.py's `release_plan`), and on no other run, the network lane's among them. A site
    marked one of `SHARDS` runs in its own tasks alone (`site`: its map tasks and its judge, ci.py's
    `site_entries`), and every other run sets it aside. This module is every row's key's (ci.py's `SHARED`),
    so a change to what a run selects re-keys every row (row 2558)."""
    at = command.index(SELECTION[1])
    if release:
        chosen = "release_env"
    elif network:
        chosen = "network and not release_env"
    else:
        chosen = (SELECTION[1] + " and not release_env" + ("" if whole else " and not whole_suite")
                  + ("" if site else "".join(f" and not {kind}" for kind in SHARDS)))
    return [*command[:at], chosen, *command[at + 1:]]


def selection(tests: list[str], split: str, release: str = "") -> list[str]:
    """pytest's arguments that select `tests` as a CI task does: one group of a split file (`split`,
    split.py), a release check's tests (`release`, release.py), or a check-1 site's test alone (`split` its
    group, `SITE`)."""
    site = SITE.fullmatch(split)
    grouping = (["-p", "split", f"--ci-group={split}"] if split and not site else []) + (
        ["-p", "release", f"--ci-release={release}"] if release else [])
    named = [f"{name}::{site['test'].replace('.', '::')}" for name in tests] if site else tests
    return [*named, *grouping]


def pytest_command(run: list[str], tests: list[str], split: str, release: str) -> list[str]:
    """CI's pytest `run` over `tests`, with the plugin that records what ran (executed.py), as `selection`
    selects them. A release check's pytest runs with its requirement over the suite's environment, whose
    torch and jax it keeps: uv's overlay."""
    command = [*run, "-p", "executed", *selection(tests, split, release)]
    return ["uv", "run", "--no-project", "--python", command[0], "--with", release, "--", "python",
            *command[1:]] if release else command


MARKS = ("gpu",)
"""The marks of tests a GPU runs: its behavioural flows' (tests/test_gpu_*.py, the owner's rule of no unit
tests on a GPU, ledger 2452), and the few gpu-marked tests not yet a flow's cells. A kernel's tests run on the
CPU alone, interpreted; each kernel's native run is a flow's cell."""


SHARED_SETUP = "shared_setup"
"""The mark of tests that share one expensive setup (pyproject.toml's markers), a module- or
class-scoped fixture's making: `@pytest.mark.shared_setup("name")` on each test's own decorator, of a
literal name, as the plan reads it from the source (`shared_setups`). A split file's tests of one name
run in one group (`cut`), so the setup is paid once; a fixture that only cleans up, or one every test
takes (autouse), is no reason to mark (Main, 2026-11-02: the joint release's fit and an admitted plan,
45-95 s each, paid twice where a cut parted their tests)."""


def test_name(node: str) -> str:
    """Test node `node`'s test's name in its file: its function's, or a class's method's
    `Class.test_name`, its parameters' ids aside."""
    return node.partition("::")[2].split("[")[0].replace("::", ".")


def declared_marks(name: str, text: str) -> dict[str, tuple[str, dict[str, int], int]]:
    """The sites test file `name`, whose source is `text`, declares (ci.py's `sites`), each by its name with
    its mark's kind of `SHARDS`, its literal counts and the line that holds it: a test's own decorator, or a
    parameter's of the test's one literal parametrize list, `pytest.param(..., id="<id>", marks=...)`, its
    site `test[id]` (Main's ruling, 2026-11-09: test_recurrence's ~60 hook combinations, each a check-1 site
    of its own seeds). A mark written otherwise would set its site aside from its file's tasks and plan none
    of its own, so nothing would run it: its file is refused, by the lines that hold it. None for a source
    that does not parse, whose collection fails its row, saying why."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    found: dict[str, tuple[str, dict[str, int], int]] = {}
    read: set[int] = set()

    def marked(site: str, mark: ast.Call) -> None:
        kind = ast.unparse(mark.func).removeprefix("pytest.mark.")
        if kind not in SHARDS:
            return
        counts = {keyword.arg: keyword.value.value for keyword in mark.keywords
                  if isinstance(keyword.value, ast.Constant)}
        allowed = [{SHARDS[kind], "per_task"}]
        if kind == "check1_shards":
            allowed.append({SHARDS[kind], "per_task", "conditioning", "conditioning_per_task"})
        if (not mark.args and len(counts) == len(mark.keywords) and set(counts) in allowed
                and site not in found and all(type(count) is int and count > 0 for count in counts.values())):
            found[site] = (kind, counts, mark.lineno)
            read.add(id(mark.func))

    decorated = list(decorated_tests(tree.body))
    for test, mark in decorated:
        marked(test, mark)
    lists = {test: [mark for each, mark in decorated if each == test
                    and ast.unparse(mark.func) == "pytest.mark.parametrize"] for test, _ in decorated}
    for test, parametrized in lists.items():
        if len(parametrized) != 1:
            continue
        given = parametrized[0].args[1] if len(parametrized[0].args) > 1 else next(
            (keyword.value for keyword in parametrized[0].keywords if keyword.arg == "argvalues"), None)
        for value in given.elts if isinstance(given, (ast.List, ast.Tuple)) else ():
            if not (isinstance(value, ast.Call) and ast.unparse(value.func) == "pytest.param"):
                continue
            keywords = {keyword.arg: keyword.value for keyword in value.keywords}
            named, marks = keywords.get("id"), keywords.get("marks")
            if not (isinstance(named, ast.Constant) and isinstance(named.value, str)
                    and PARAM_ID.fullmatch(named.value)):
                continue
            for mark in marks.elts if isinstance(marks, (ast.List, ast.Tuple)) else [marks]:
                if isinstance(mark, ast.Call) and test not in found:
                    marked(f"{test}[{named.value}]", mark)
    unread = sorted(node.lineno for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and node.attr in SHARDS
                    and ast.unparse(node.value).rpartition(".")[2] == "mark" and id(node) not in read)
    if unread:
        raise SystemExit(f"{name}, line {', '.join(map(str, unread))}: check1_shards and shards mark a test "
                         f"once, on its own decorator, as @pytest.mark.check1_shards(seeds=S, per_task=P), "
                         f"with conditioning=C, conditioning_per_task=Q where it declares a weighed check, "
                         f"or @pytest.mark.shards(units=N, per_task=P), of literal counts above zero, or a "
                         f"parameter of its one literal parametrize list once, as pytest.param(..., "
                         f"id=\"<id>\", marks=pytest.mark.check1_shards(...)) of a literal id of letters, "
                         f"digits, _ and -, which the plan reads from the source "
                         f"(tools/armada/ci.py `sites`)")
    return found


def site_name(node: str, sites: Container[str]) -> str:
    """Test node `node`'s name among a file's `sites` (ci.py's `sites`): its parameter's site's, `test[id]`,
    where that parameter is marked one, else its test's (`test_name`)."""
    named = test_name(node)
    held = f"{named}[{node.partition('[')[2][:-1]}]" if "[" in node else named
    return held if held in sites else named


def shared_setups(name: str) -> dict[str, str]:
    """Test file `name`'s tests marked `SHARED_SETUP`, each by its name in the file (`test_name`) with
    the setup it names. Read from the source, as the plan runs under a Python with no pytest and imports
    no test: a mark written elsewhere than a test's own decorator (on a class or a module, in a
    parameter's marks), or of a name not a literal string, would be cut apart by the plan and kept
    together by nothing it can read, so the file is refused, by the line that holds it."""
    path = Path(name)
    text = path.read_text() if path.is_file() else ""
    return marked_setups(name, text) if SHARED_SETUP in text else {}


def decorated_tests(body: list[ast.stmt], prefix: str = "") -> Iterator[tuple[str, ast.Call]]:
    """Each test function of a test file's `body` (`ast.parse`'s), by its name in the file (`test_name`'s
    form), with each call its own decorators make (`@pytest.mark.<mark>(...)`): what the plan reads a test's
    marks from, as it imports no test (`shared_setups`, ci.py's `sites`)."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from decorated_tests(node.body, f"{prefix}{node.name}.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name[:4] == "test":
            for mark in node.decorator_list:
                if isinstance(mark, ast.Call):
                    yield prefix + node.name, mark


@functools.cache
def marked_setups(name: str, text: str) -> dict[str, str]:
    """`shared_setups` of test file `name`, whose source is `text`, read once a text."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Its collection fails its row, saying why.
        return {}
    found: dict[str, str] = {}
    read: set[int] = set()
    for test, mark in decorated_tests(tree.body):
        if (ast.unparse(mark.func) == f"pytest.mark.{SHARED_SETUP}" and len(mark.args) == 1
                and not mark.keywords and isinstance(mark.args[0], ast.Constant)
                and isinstance(mark.args[0].value, str) and test not in found):
            found[test] = mark.args[0].value
            read.add(id(mark.func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == SHARED_SETUP and id(node) not in read:
            raise SystemExit(f"{name}, line {node.lineno}: {SHARED_SETUP} is read on a test's own "
                             f"decorator alone, once, as @pytest.mark.{SHARED_SETUP}(\"name\") of a literal "
                             "name, which the plan reads from the source (tools/armada/session.py "
                             "`shared_setups`)")
    return found


def units(nodes: list[str]) -> list[list[int]]:
    """The indexes of `nodes` (test node ids, in node-id order) a cut keeps together: each test alone, but
    those of one file marked with one `SHARED_SETUP`, together, at the place of the first of them. Each
    file's marks are read once."""
    marks: dict[str, dict[str, str]] = {}
    held: dict[tuple[str, str] | int, list[int]] = {}
    for index, node in enumerate(nodes):
        name = node.split("::")[0]
        if name not in marks:
            marks[name] = shared_setups(name)
        setup = marks[name].get(test_name(node))
        held.setdefault(index if setup is None else (name, setup), []).append(index)
    return list(held.values())


def unit_seconds(nodes: list[str], seconds: list[float]) -> list[float]:
    """Each of `units`' weights: its tests' `seconds` (each node's, in `nodes`' order) summed."""
    return [sum(seconds[index] for index in unit) for unit in units(nodes)]


def cut(nodes: list[str], seconds: list[float], count: int) -> list[list[str]]:
    """`nodes` (node-id order, each weighing its `seconds`) as `count` groups (`groups`) of `units`, a shared
    setup's tests in one group: how the plan cuts a split file's recorded tests (ci.py's `members`), which
    every task of it keeps whatever else it collects (`kept`)."""
    held = units(nodes)
    runs = groups([sum(seconds[index] for index in unit) for unit in held], count)
    return [[nodes[index] for unit in held[run.start:run.stop] for index in unit] for run in runs]


def stored_cuts() -> dict[str, list[str]]:
    """The stored cuts (`CUTS`): each split file's group boundaries, none where the checkout has no record."""
    return json.loads(CUTS.read_text()) if CUTS.is_file() else {}


def cut_by(nodes: list[str], bounds: list[str]) -> list[list[str]]:
    """`nodes` (node-id order) as the `len(bounds) + 1` groups a stored cut names: each of `units` in the
    group whose interval of node ids, [bounds[k - 1], bounds[k]), holds its first test, so a test added later
    falls in the interval it sorts into and no test leaves its group while the cut stands."""
    held: list[list[str]] = [[] for _ in range(len(bounds) + 1)]
    for unit in units(nodes):
        held[bisect.bisect_right(bounds, nodes[unit[0]])].extend(nodes[index] for index in unit)
    return [sorted(members) for members in held]


def planned(name: str, nodes: list[str], seconds: list[float], count: int,
            stored: Mapping[str, list[str]] | None = None) -> list[list[str]]:
    """Test file `name`'s recorded `nodes` as the plan's `count` groups: its stored cut where it has `count`
    groups and the groups are cut by the durations record (`cut_record`; a GPU tier cuts by its own seconds),
    else cut by their `seconds` (`cut`). `stored` is `stored_cuts()` where a caller has read it once."""
    bounds = (stored_cuts() if stored is None else stored).get(name) if cut_record() == DURATIONS else None
    if bounds is not None and len(bounds) + 1 == count:
        return cut_by(nodes, bounds)
    return cut(nodes, seconds, count)


@functools.cache
def bound(text: str) -> frozenset[str]:
    """The names a test file's source `text` binds at its top level (a function's, a class's, an assignment's,
    an import's), which a test's node id begins with."""
    names: set[str] = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        else:
            names.update(target.id for target in ast.walk(node) if isinstance(target, ast.Name)
                         and isinstance(target.ctx, ast.Store))
    return frozenset(names)


def standing(name: str, recorded: Mapping[str, float]) -> dict[str, float]:
    """Of test file `name`'s `recorded` tests (node id to seconds), those whose function or class the file
    still binds at its top level (`bound`): a test a lane renamed or cut leaves its record behind until main's
    CI folds the file again, and cut by it the file ran as groups that held nothing (2026-11-12:
    tests/test_recurrence.py, 37 tests, was planned as 133 groups by a record of 262, 244 of them a renamed
    function's; 113 tasks ran nothing, 2,720 task-seconds). A class's tests stay while its class does, whoever
    defines them; a file that is gone or does not parse keeps its record as it is. What the plan cuts a file
    by (ci.py's `own_tests`) and every task of it cuts alike (split.py)."""
    path = Path(name)
    if not path.is_file():
        return dict(recorded)
    try:
        names = bound(path.read_text())
    except SyntaxError:
        # Its collection fails its row, saying why (ci.py's `collect`); its record stands meanwhile.
        return dict(recorded)
    return {node: seconds for node, seconds in recorded.items()
            if node.partition("::")[2].split("[")[0].split("::")[0] in names}


def own(recorded: Mapping[str, float], sites: Container[str]) -> dict[str, float]:
    """Of a test file's `recorded` tests (node id to seconds, in node-id order), those but its `sites'`
    (`declared_marks`, `site_name`), which run apart: what the plan cuts the file by (ci.py's `own_tests`)
    and every task of it cuts alike (split.py)."""
    return {node: seconds for node, seconds in recorded.items() if site_name(node, sites) not in sites}


def cut_record() -> Path:
    """The record a task's groups are cut by (split.py) and keyed by (ci.py's `members`): the durations record
    (`DURATIONS`), or where the task is a GPU tier's, the seconds its GPUs took (DEW_SPLIT_DURATIONS, a record
    of the same shape dew-ops' maintier writes: the CPU suite skips a file's GPU tests, so its record times
    none of them)."""
    weights = os.environ.get("DEW_SPLIT_DURATIONS")
    return Path(weights) if weights else DURATIONS


def kept(nodes: list[str], planned: list[list[str]], mean: float) -> list[list[str]]:
    """`nodes`, a task's collected tests of one split file in node-id order, as the plan's groups
    `planned` (the file's recorded tests cut, `cut`): each test the record names in its planned group, so no
    group runs a recorded test its plan did not weigh (2026-11-11: two new parameters moved test_recurrence's
    963 s test into a group planned at 77 s); a test the record lacks that shares a setup with a recorded
    one (`units`) in that one's group; and the rest, `mean` seconds each, cut into as many runs of
    consecutive tests as there are groups (`groups`), the kth run joining the kth group, so a new
    parametrization spreads across the groups from its first run, never into one group past its deadline
    (2026-11-03: test_attention_mosaic's ~190 new ids, weighed nothing, packed into one). A group of a file
    whose record misses a test it collects is run, never reused (ci.py's `record`)."""
    group_of = {node: index for index, members in enumerate(planned) for node in members}
    held: list[list[str]] = [[] for _ in planned]
    new: list[list[str]] = []
    for unit in units(nodes):
        placed = next((group_of[nodes[index]] for index in unit if nodes[index] in group_of), None)
        if placed is None:
            new.append([nodes[index] for index in unit])
        else:
            held[placed].extend(nodes[index] for index in unit)
    for index, run in enumerate(groups([mean * len(unit) for unit in new], len(planned))):
        held[index].extend(node for unit in new[run.start:run.stop] for node in unit)
    return [sorted(members) for members in held]


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

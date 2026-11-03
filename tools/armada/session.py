"""A CI task's pytest session, as the plan (ci.py) and the plugins a task loads into it (split.py,
executed.py) share it: how pytest runs and what it sees (`pytest_run`, `pytest_environment`), how long it runs
(`DEADLINE`, `GRACE`), which tests take another path on a GPU (`MARKS`), and how a split file's tests are cut
(`cut`, `groups`), those of one shared setup together (`SHARED_SETUP`). What it decides changes a task's
outcome, so the test cache keys every file's record on it (ci.py's `SHARED`); the plan's own code does not,
and a change to it reruns nothing. The standard library alone: the plan imports it where pytest is not
installed."""

import ast
import functools
import os
from collections.abc import Container, Iterator, Mapping
from pathlib import Path

DURATIONS = Path("tests/test_durations.json")


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
    it can handle it, then the group killed past the grace), its stacks written once as it passes
    (executed.py), and a fatal signal still prints the stacks (pytest's faulthandler)."""
    runner = [*measured, "-m", "pytest"] if measured else ["pytest"]
    return [f".venv-{python}/bin/python", "-S", "-u", "-c", BOOT, *runner, "-q", *SELECTION, "-rfE",
            "--tb=short", "--continue-on-collection-errors", "-p", "no:cacheprovider", f"--junitxml={report}"]


def pytest_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What CI's pytest runs under: this process's environment, `environ` over it, the Hub closed, and
    tools/armada on the path, where the plugins a task loads (split.py, executed.py) import from."""
    path = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")]))
    return {**os.environ, **environ, **CLOSED_HUB, "PYTHONPATH": path}


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


def units(nodes: list[str], recorded: Container[str] | None = None) -> list[list[int]]:
    """The indexes of `nodes` (test node ids, in node-id order) a cut keeps together: each test alone, but
    those of one file marked with one `SHARED_SETUP`, together, whether the durations record names them yet
    or not, at the place of the first of them `recorded` names (the nodes the record names, where given),
    else of the first: the plan, which sees the recorded tests alone, then orders the units as a task does.
    Each file's marks are read once."""
    marks: dict[str, dict[str, str]] = {}
    held: dict[tuple[str, str] | int, list[int]] = {}
    for index, node in enumerate(nodes):
        name = node.split("::")[0]
        if name not in marks:
            marks[name] = shared_setups(name)
        setup = marks[name].get(test_name(node))
        held.setdefault(index if setup is None else (name, setup), []).append(index)

    def place(unit: list[int]) -> int:
        return next((index for index in unit if recorded is None or nodes[index] in recorded), unit[0])

    return sorted(held.values(), key=place)


def unit_seconds(nodes: list[str], seconds: list[float]) -> list[float]:
    """Each of `units`' weights: its tests' `seconds` (each node's, in `nodes`' order) summed."""
    return [sum(seconds[index] for index in unit) for unit in units(nodes)]


def cut(nodes: list[str], seconds: list[float], count: int,
        recorded: Container[str] | None = None) -> list[list[str]]:
    """`nodes` (node-id order, each weighing its `seconds`) as `count` groups (`groups`) of `units`: how
    every task of a split file and the plan alike cut its tests (split.py, ci.py's `members`), a shared
    setup's tests in one group, at its first test `recorded` names (the nodes the durations record names,
    where given). A test the record lacks weighs what its caller gives it (split.py: its file's mean), so a
    new parametrization spreads across the groups from its first run, never into one group past its
    deadline (2026-11-03: test_attention_mosaic's ~190 new ids, weighed nothing, packed into one). The plan,
    which reads the record alone, cuts as the task does where the record names every test the task collects;
    a group of a file it does not is run, never reused (ci.py's `record`)."""
    held = units(nodes, recorded)
    runs = groups([sum(seconds[index] for index in unit) for unit in held], count)
    return [[nodes[index] for unit in held[run.start:run.stop] for index in unit] for run in runs]


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

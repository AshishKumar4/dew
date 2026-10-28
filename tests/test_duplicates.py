"""tools/duplicates.py, the duplicate-code gate: it refuses code written twice, read as text (pylint's
similarity) and as structure (a function, or six statements, equal once names are renamed in order of use
and constants read as their types), and passes the same files once the code has one owner its callers
share. Declarations alike (fields, imports, stubs, runs of one call with other data) are no copy. Held to
a base, it refuses only a copy the base did not have, wherever it is written, and holds one that moved."""

import os
import subprocess
import sys
from pathlib import Path

from tool_support import load

GATE = Path(__file__).resolve().parents[1] / "tools" / "duplicates.py"
BASE_HISTORY = GATE.with_name("base_history.sh")
duplicates = load("duplicates")

BLOCK = """\
    total = 0
    for index, value in enumerate(values):
        if value > limit:
            total += value * index
        else:
            total -= value
    return total
"""
"""Seven lines of logic, one past the text reading's six."""

SPREAD = '''\
def spread(values, limit):
    """How far apart the values past a limit lie, over how many there are, and where their middle falls."""
    kept = [value for value in values if value > limit]
    if not kept:
        raise ValueError(f"no value past {limit}")
    low, high = min(kept), max(kept)
    middle = (low + high) / 2
    above = sum(1 for value in kept if value > middle)
    return {"spread": (high - low) / len(kept), "above": above / len(kept)}
'''

RANGE_OVER_COUNT = '''\
def range_over_count(items, floor):
    """The range of the items above a floor, per item, and the share above the range's midpoint."""
    chosen = [item for item in items if item > floor]
    if not chosen:
        raise ValueError(f"nothing above {floor}")
    lo, hi = min(chosen), max(chosen)
    centre = (lo + hi) / 2
    over = sum(1 for item in chosen if item > centre)
    return {"range": (hi - lo) / len(chosen), "over": over / len(chosen)}
'''
"""`SPREAD` renamed, its message, keys and docstring reworded: no line is SPREAD's; it is the same logic."""


def gate(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GATE), *arguments, "."], cwd=root, capture_output=True,
                          text=True, timeout=300)


def sources(root: Path) -> dict[str, str]:
    """`root`'s files as the gate reads them, named from `root`."""
    return {str(path.relative_to(root)): path.read_text() for path in sorted(root.rglob("*.py"))}


def copies(root: Path) -> list:
    """The copies the gate names in `root`'s files."""
    return duplicates.structural(sources(root))


def added(root: Path, base: str = "main") -> list:
    """The copies `root`'s checkout adds to `base`'s, which the gate held to it refuses."""
    merged = duplicates.merge_base(root, base)
    return duplicates.named(duplicates.judged(duplicates.found(sources(root)), merged, root)[0])


def owners(hits: list) -> list[tuple[str, list[str]]]:
    return [(hit.kind, [f"{place.file} {place.owner}" for place in hit.places]) for hit in hits]


def tree(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    # duplicates_*: CI puts tools/armada on its children's PYTHONPATH, where a plain name can be a module.
    root = tmp_path / f"duplicates_{name}"
    root.mkdir()
    for name, text in files.items():
        (root / name).write_text(text)
    return root


STEPS = ["kept = [value for value in values if value > limit]",
         "low, high = min(kept), max(kept)",
         "middle = (low + high) / 2",
         "above = [value for value in kept if value > middle]",
         "below = [value for value in kept if value <= middle]",
         "balance = (len(above) - len(below)) / len(kept)"]
"""Six statements of logic, past the structural reading's bar."""

MOMENTS = ["total = sum(values)",
           "count = len(values)",
           "mean = total / count",
           "squares = [(value - mean) ** 2 for value in values]",
           "variance = sum(squares) / count",
           "deviation = variance ** 0.5"]
"""Six other statements, another algorithm."""


def renamed(steps: list[str]) -> list[str]:
    return [step.replace("kept", "chosen").replace("value", "item") for step in steps]


def body(*runs: list[str]) -> str:
    return "".join(f"    {step}\n" for run in runs for step in run)


def before_return(function: str, *runs: list[str]) -> str:
    return function.replace("    return", body(*runs) + "    return")


def test_seven_lines_two_files_copy_fail_the_gate_and_six_or_one_owner_pass_it(tmp_path):
    copied = tree(tmp_path, "copied", {f"{name}.py": f"def {name}(values, limit):\n{BLOCK}"
                                       for name in ("first", "second")})
    assert gate(copied).returncode == 1
    owned = tree(tmp_path, "owned", {"first.py": f"def first(values, limit):\n{BLOCK}",
                                     "second.py": "from first import first\n\n\ndef second(values, limit):\n"
                                                  "    return first(values, limit)\n"})
    assert gate(owned).returncode == 0
    # The last line differs, so the two share six lines, not past the bar, and differ in what they do.
    differing = BLOCK.replace("return total", "return -total")
    sharing_six = tree(tmp_path, "sharing_six", {"first.py": f"def first(values, limit):\n{BLOCK}",
                                                 "second.py": f"def second(values, limit):\n{differing}"})
    assert gate(sharing_six).returncode == 0


def test_each_reading_alone_fails_its_own_copies_and_passes_the_others(tmp_path):
    """One reading alone (tools/check_gates' two gates; check.sh --changed runs the structure): seven
    settings two files write alike, a copy only the text reading sees, fail the text and pass the structure;
    a function two files write under other names, a copy only the structure sees, fails the structure and
    passes the text."""
    settings = "".join(f"{name} = {value}\n" for value, name in enumerate(
        ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta")))
    texts = tree(tmp_path, "texts", {"first.py": settings, "second.py": settings})
    assert gate(texts).returncode == 1
    assert gate(texts, "--reading=text").returncode == 1
    assert gate(texts, "--reading=structure").returncode == 0
    functions = tree(tmp_path, "functions", {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT})
    assert gate(functions).returncode == 1
    assert gate(functions, "--reading=structure").returncode == 1
    assert gate(functions, "--reading=text").returncode == 0


def test_a_tree_without_a_python_file_passes_both_readings(tmp_path):
    """The script reads its files in forked processes (tools/forked.py), which an empty tree gives none."""
    empty = tree(tmp_path, "empty", {})
    assert gate(empty, "--reading=text").returncode == 0
    assert gate(empty).returncode == 0


def test_no_pragma_holds_a_text_copy(tmp_path):
    """The text reading reads no `pylint: disable=duplicate-code`: a copy is fixed, never silenced."""
    settings = "".join(f"{name} = {value}\n" for value, name in enumerate(
        ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta")))
    silenced = "# pylint: disable=duplicate-code\n" + settings
    texts = tree(tmp_path, "silenced", {"first.py": silenced, "second.py": silenced})
    assert gate(texts, "--reading=text").returncode == 1


def test_a_function_two_files_write_under_other_names_fails_and_one_owner_passes(tmp_path):
    copied = tree(tmp_path, "copied", {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["ranges.py range_over_count", "spread.py spread"])]
    calling = ("from spread import spread\n\n\n"
               "def range_over_count(items, floor):\n    return spread(items, floor)\n")
    owned = tree(tmp_path, "owned", {"spread.py": SPREAD, "ranges.py": calling})
    assert gate(owned).returncode == 0


def test_six_statements_two_functions_share_fail_where_their_functions_differ(tmp_path):
    first = "def first(values, limit, scale):\n" + body(STEPS) + "    return balance * scale\n"
    second = ("def second(values, limit):\n    values = sorted(values)\n" + body(renamed(STEPS))
              + "    return {'balance': balance, 'kept': len(chosen)}\n")
    copied = tree(tmp_path, "copied", {"first.py": first, "second.py": second})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("statements", ["first.py first", "second.py second"])]
    # Seven shared statements are two windows over one copy: the gate names the copy once.
    longer = [*STEPS, "spread = max(kept) - min(kept)"]
    first = "def first(values, limit, scale):\n" + body(longer) + "    return balance * scale\n"
    second = ("def second(values, limit):\n    values = sorted(values)\n" + body(renamed(longer))
              + "    return {'balance': balance, 'kept': len(chosen)}\n")
    longer_copy = tree(tmp_path, "longer", {"first.py": first, "second.py": second})
    assert gate(longer_copy).returncode == 1
    assert owners(copies(longer_copy)) == [("statements", ["first.py first", "second.py second"])]


def test_two_copies_between_two_functions_are_two_copies(tmp_path):
    """Two algorithms both functions copy, apart in each, are two copies, each named."""
    left = ("def left(values, limit, scale):\n" + body(STEPS, ["scale = scale * 2"], MOMENTS)
            + "    return balance * scale + deviation\n")
    right = ("def right(values, limit):\n" + body(renamed(STEPS), ["values = list(values)"], renamed(MOMENTS))
             + "    return {'balance': balance, 'deviation': deviation}\n")
    copied = tree(tmp_path, "copied", {"left.py": left, "right.py": right})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("statements", ["left.py left", "right.py right"])] * 2


def test_one_file_writing_a_function_twice_fails(tmp_path):
    copied = tree(tmp_path, "copied", {"both.py": SPREAD + "\n\n" + RANGE_OVER_COUNT})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["both.py spread", "both.py range_over_count"])]


def test_a_local_annotation_hides_no_copy(tmp_path):
    annotated = RANGE_OVER_COUNT.replace("centre = ", "centre: float = ")
    copied = tree(tmp_path, "copied", {"spread.py": SPREAD, "ranges.py": annotated})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["ranges.py range_over_count", "spread.py spread"])]


def test_an_inner_function_copied_with_its_outer_one_is_one_copy(tmp_path):
    def nested(outer: str, arguments: str, inner: str, function: str) -> str:
        return (f"def {outer}({arguments}):\n" + "".join(f"    {line}\n" for line in function.splitlines())
                + f"    return {inner}({arguments})\n")

    copied = tree(tmp_path, "copied", {"outer.py": nested("summary", "values, limit", "spread", SPREAD),
                                       "inner.py": nested("report", "items, floor", "range_over_count",
                                                          RANGE_OVER_COUNT)})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["inner.py report", "outer.py summary"])]


def test_two_files_evaluating_alike_fail(tmp_path):
    """Two objectives that score their own batch fields through one compiled program the same way: a small
    function (ledger 932's size), written twice."""
    evaluate = ("class {0}:\n    def evaluate(self, params, batch, step):\n"
                "        {1} = {{{2!r}: batch[{2!r}], {3!r}: batch[{3!r}]}}\n"
                "        return self.compiled(self.variables(params, step), {1}, step.key)\n")
    copied = tree(tmp_path, "copied", {"grpo.py": evaluate.format("Grpo", "prompts", "prompt", "length"),
                                       "dpo.py": evaluate.format("Dpo", "pairs", "ids", "mask")})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["dpo.py Dpo.evaluate", "grpo.py Grpo.evaluate"])]


def test_two_files_closing_alike_on_exit_fail(tmp_path):
    """Two context managers whose exits close the same way, the error noted on the one in flight: a small
    function (21 nodes, training/distributed.py's and tracker.py's), written twice."""
    exit_ = ("class {0}:\n    def close(self):\n        self.closed = True\n\n"
             "    def __exit__(self, kind, {1}, traceback):\n        try:\n            self.close()\n"
             "        except BaseException as {2}:\n            if {1} is None:\n                raise\n"
             "            {1}.add_note(f'{0} cleanup failed: {{{2}!r}}')\n")
    copied = tree(tmp_path, "copied", {"prefetch.py": exit_.format("Prefetch", "exc", "error"),
                                       "tracker.py": exit_.format("Tracker", "pending", "failure")})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [
        ("function", ["prefetch.py Prefetch.__exit__", "tracker.py Tracker.__exit__"])]


def test_a_copy_written_on_one_line_fails_held_to_a_base_or_not(tmp_path):
    """A function and its six statements on one line span the same lines: the copy is still named, and
    still fails."""
    first = "def first(x): a = x + 1; b = a * 2; c = b - 3; d = c / 4; e = d ** 2; f = e % 5; return f\n"
    second = ("def second(y): p = y + 11; q = p * 12; r = q - 13; s = r / 14; t = s ** 12; u = t % 15;"
              " return u\n")
    copied = tree(tmp_path, "copied", {"first.py": first, "second.py": second})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["first.py first", "second.py second"])]
    root = repository(tmp_path, {"first.py": first})
    (root / "second.py").write_text(second)
    commit(root, "a head that copies it")
    assert gate(root, "--base=main").returncode == 1
    assert owners(added(root)) == [("function", ["first.py first", "second.py second"])]


def test_a_constructor_computing_its_parent_is_logic(tmp_path):
    """A constructor that only hands its arguments to its parent's declares nothing when it computes which
    parent: written twice, it is a copy."""
    computed = ("class {0}:\n    def __init__(self, {1}):\n        super(choose({1}, min({1}), max({1}),"
                " len({1}), sum({1}), tuple({1})), self).__init__()\n")
    copied = tree(tmp_path, "copied", {"left.py": computed.format("Left", "values"),
                                       "right.py": computed.format("Right", "items")})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["left.py Left.__init__", "right.py Right.__init__"])]


def test_a_copied_run_of_one_call_fails(tmp_path):
    """Six calls to one function declare data: copied with their data, they are a copy, though no line reads
    alike as text."""
    emits = [("loss", 1, "mean"), ("accuracy", 2, "mean"), ("tokens", 3, "sum"), ("rows", 4, "sum"),
             ("seconds", 5, "max"), ("memory", 6, "max")]
    left = ("def left(sink):\n    opened = sink.open()\n"
            + "".join(f"    sink.emit('{name}', {order}, '{reduce}')\n" for name, order, reduce in emits)
            + "    return opened\n")
    right = ("def right(sink):\n    sink.flush()\n"
             + "".join(f'    sink.emit("{name}", {order}, "{reduce}")\n' for name, order, reduce in emits)
             + "    return None\n")
    copied = tree(tmp_path, "copied", {"left.py": left, "right.py": right})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("statements", ["left.py left", "right.py right"])]


def test_declarations_alike_are_no_copy(tmp_path):
    """Two records' annotated fields and two parsers' arguments, alike in shape, declare different things;
    so do two modules' imports, two protocols' stubs, two runs of values one function reads and two
    constructors that store their own fields. Two functions threading a value through other numbers of one
    step are other logic."""
    record = "from dataclasses import dataclass\n\n\n@dataclass\nclass {0}:\n" + "".join(
        f"    {{1}}{index}: tuple[int, ...] | None = None\n" for index in range(8))
    parser = ("import argparse\n\n\ndef {0}():\n    parser = argparse.ArgumentParser()\n" + "".join(
        f"    parser.add_argument('--{{1}}{index}', type=int, default={index},"
        f" help='the {{1}} option {index}')\n" for index in range(8)) + "    return parser.parse_args()\n")
    imports = "".join(f"from {{0}}{index} import {{0}}a, {{0}}b, {{0}}c, {{0}}d, {{0}}e\n"
                      for index in range(6))
    stubs = "class {0}:\n" + "".join(f"    @property\n    def {{1}}{index}(self) -> int: ...\n"
                                     for index in range(6))
    reads = "".join(f"{{0}}{index} = {{1}}('{{0}}{index}')\n" for index in range(6))
    fields = ("class {0}:\n    def __init__(self, {1}, {2}):\n        self.{1}, self.{2} = {1}, {2}\n"
              "        self.{3}, self.{4}, self.{5} = [], [], []\n")
    parameters = ", ".join(f"{{1}}{index}" for index in range(6))
    annotated = f"class {{0}}:\n    def __init__(self, {parameters}):\n" + "".join(
        f"        self.{{1}}{index}: object = {{1}}{index}\n" for index in range(6))
    threading = ("def {0}(x):\n    y = normalize(x)\n    scale = y * 2\n"
                 + "    x = layer(x, depth=3)\n" * 8 + "    return x + scale\n")
    alike = tree(tmp_path, "alike", {
        "left.py": record.format("Left", "a") + parser.format("left", "x"),
        "right.py": record.format("Right", "b") + parser.format("right", "y"),
        "left_imports.py": imports.format("left"), "right_imports.py": imports.format("right"),
        "left_stubs.py": stubs.format("LeftNames", "token"),
        "right_stubs.py": stubs.format("RightNames", "field"),
        "left_reads.py": "values = {}\n" + reads.format("width", "values.get"),
        "right_reads.py": "import re\n" + reads.format("PATTERN", "re.compile"),
        "harness.py": fields.format("Harness", "failure", "paid", "opened", "closed", "calls"),
        "artifact.py": fields.format("Artifact", "name", "kind", "dirs", "files", "references"),
        "left_fields.py": annotated.format("LeftFields", "a"),
        "right_fields.py": annotated.format("RightFields", "b"),
        "deep.py": threading.format("deep"),
        "shallow.py": threading.format("shallow").replace("    x = layer(x, depth=3)\n" * 6, "")})
    assert copies(alike) == []
    assert gate(alike).returncode == 0


def commit(root: Path, message: str) -> None:
    for command in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message]):
        subprocess.run(["git", *command], cwd=root, check=True)


def repository(tmp_path: Path, files: dict[str, str]) -> Path:
    """A repository whose main holds `files`, checked out on a head branch."""
    root = tree(tmp_path, "repository", files)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    commit(root, "main")
    subprocess.run(["git", "checkout", "-q", "-b", "head"], cwd=root, check=True)
    return root


def branch(root: Path, name: str) -> None:
    subprocess.run(["git", "checkout", "-q", "-b", name, "main"], cwd=root, check=True)


def test_held_to_a_base_only_a_copy_the_base_lacked_fails(tmp_path):
    """Main holds a copy (spread.py, ranges.py): a head that leaves it passes with the base and fails
    without; a head that writes the function a third time fails with the base too, naming the new copy,
    however the gate is given the files."""
    root = repository(tmp_path,
                      {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT, "other.py": "VALUE = 1\n"})
    (root / "other.py").write_text("VALUE = 2\n")
    commit(root, "a head that leaves the copy")
    assert gate(root, "--base=main").returncode == 0 and added(root) == []
    assert gate(root).returncode == 1
    third = RANGE_OVER_COUNT.replace("range_over_count", "third").replace("chosen", "picked")
    (root / "third.py").write_text(third)
    commit(root, "a head that copies it again")
    assert gate(root, "--base=main").returncode == 1
    assert owners(added(root)) == [
        ("function", ["ranges.py range_over_count", "spread.py spread", "third.py third"])]
    absolute = subprocess.run([sys.executable, str(GATE), "--base=main", str(root)], cwd=root, timeout=300,
                              capture_output=True)
    assert absolute.returncode == 1


def test_a_checkout_carrying_its_base_is_held_to_it_and_fetches_nothing(tmp_path):
    """A gate's container checkout carries its commit's history (armada's history full) and is given BASE,
    its head's merge-base with main, with no origin/main and no reachable origin: tools/base_history.sh
    points origin/main at BASE from the history it holds, fetching nothing, and the gate then fails the
    head's new copy; a checkout whose history lacks BASE fails, naming it, fetching nothing (Main,
    2026-10-28: the gates' fetches of main over HTTPS failed lanes' lint, curl 92)."""
    root = repository(tmp_path, {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT})
    base = subprocess.run(["git", "rev-parse", "main"], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()
    third = RANGE_OVER_COUNT.replace("range_over_count", "third").replace("chosen", "picked")
    (root / "third.py").write_text(third)
    commit(root, "a head that copies it again")
    carried = tmp_path / "carried"
    subprocess.run(["git", "clone", "-q", "--branch", "head", root.as_uri(), str(carried)], check=True)
    for command in (["remote", "set-url", "origin", str(tmp_path / "unreachable")],
                    ["update-ref", "-d", "refs/remotes/origin/main"]):
        subprocess.run(["git", *command], cwd=carried, check=True)
    held = subprocess.run(["sh", str(BASE_HISTORY)], cwd=carried, timeout=120, capture_output=True, text=True,
                          env={**os.environ, "BASE": base})
    assert held.returncode == 0, held.stderr
    assert gate(carried, "--base=origin/main").returncode == 1
    assert owners(added(carried, "origin/main")) == [
        ("function", ["ranges.py range_over_count", "spread.py spread", "third.py third"])]
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "head", root.as_uri(), str(shallow)],
                   check=True)
    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "unreachable")], cwd=shallow,
                   check=True)
    lacking = subprocess.run(["sh", str(BASE_HISTORY)], cwd=shallow, timeout=120, capture_output=True,
                             text=True, env={**os.environ, "BASE": base})
    assert lacking.returncode != 0 and base in lacking.stderr and "unreachable" not in lacking.stderr


def test_held_to_a_base_a_new_copy_beside_a_held_one_fails(tmp_path):
    """Main holds one copy between two functions. A head that copies another algorithm between the same two
    functions fails, as does one that writes the held copy a third time in one of them."""
    left = "def left(values, limit, scale):\n" + body(STEPS) + "    return balance * scale\n"
    right = ("def right(values, limit):\n    values = sorted(values)\n" + body(renamed(STEPS))
             + "    return {'balance': balance}\n")
    root = repository(tmp_path, {"left.py": left, "right.py": right})
    assert gate(root, "--base=main").returncode == 0
    (root / "left.py").write_text(before_return(left, ["scale = scale * 2"], MOMENTS))
    (root / "right.py").write_text(before_return(right, ["values = list(values)"], renamed(MOMENTS)))
    commit(root, "a head that copies a second algorithm")
    assert gate(root, "--base=main").returncode == 1
    assert [hit.kind for hit in added(root)] == ["statements"]
    branch(root, "again")
    (root / "left.py").write_text(before_return(left, ["scale = scale * 2"], renamed(STEPS)))
    commit(root, "a head that writes the held copy again")
    assert gate(root, "--base=main").returncode == 1
    assert [len(hit.places) for hit in added(root)] == [3]


def test_held_to_a_base_a_moved_or_shrunk_copy_passes(tmp_path):
    """Main holds a function two files write. A head that renames one file and its function holds the same
    copy; so does one that changes one function's ending, which leaves six statements copied."""
    first = "def first(values, limit, scale):\n" + body(STEPS) + "    return balance * scale\n"
    second = "def second(values, limit, scale):\n" + body(renamed(STEPS)) + "    return balance * scale\n"
    root = repository(tmp_path, {"first.py": first, "second.py": second})
    subprocess.run(["git", "mv", "second.py", "moved.py"], cwd=root, check=True)
    (root / "moved.py").write_text(second.replace("second", "moved"))
    commit(root, "a head that moves the copy")
    assert gate(root, "--base=main").returncode == 0 and added(root) == []
    branch(root, "shrunk")
    (root / "second.py").write_text(second.replace("balance * scale", "-balance"))
    commit(root, "a head that changes one copy's ending")
    assert gate(root, "--base=main").returncode == 0 and added(root) == []


def test_held_to_a_base_an_uncommitted_copy_fails(tmp_path):
    root = repository(tmp_path, {"spread.py": SPREAD})
    (root / "ranges.py").write_text(RANGE_OVER_COUNT)
    assert gate(root, "--base=main").returncode == 1
    assert owners(added(root)) == [("function", ["ranges.py range_over_count", "spread.py spread"])]

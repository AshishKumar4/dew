"""tools/duplicates.py, the duplicate-code gate: it refuses code two files write, read as text (pylint's
similarity) and as structure (a function, or six statements, equal once names are renamed in order of use
and constants read as their types), and passes the same files once the code has one owner its callers
share. Declarations alike (fields, imports, stubs, runs of one call) are no copy. Held to a base, it
refuses only a copy the base did not have."""

import subprocess
import sys
from pathlib import Path

from tool_support import load

GATE = Path(__file__).resolve().parents[1] / "tools" / "duplicates.py"
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


def copies(root: Path) -> list:
    """The structural copies the gate reads in `root`'s files, as it names them from `root`."""
    return duplicates.structural({str(path.relative_to(root)): path.read_text()
                                  for path in sorted(root.rglob("*.py"))})


def owners(hits: list) -> list[tuple[str, list[str]]]:
    return [(hit.kind, [f"{place.file} {place.owner}" for place in hit.places]) for hit in hits]


def tree(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    # duplicates_*: CI puts tools/armada on its children's PYTHONPATH, where a plain name can be a module.
    root = tmp_path / f"duplicates_{name}"
    root.mkdir()
    for name, text in files.items():
        (root / name).write_text(text)
    return root


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


def test_a_function_two_files_write_under_other_names_fails_and_one_owner_passes(tmp_path):
    copied = tree(tmp_path, "copied", {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("function", ["ranges.py range_over_count", "spread.py spread"])]
    calling = ("from spread import spread\n\n\n"
               "def range_over_count(items, floor):\n    return spread(items, floor)\n")
    owned = tree(tmp_path, "owned", {"spread.py": SPREAD, "ranges.py": calling})
    assert gate(owned).returncode == 0


def test_six_statements_two_functions_share_fail_where_their_functions_differ(tmp_path):
    steps = ["kept = [value for value in values if value > limit]",
             "low, high = min(kept), max(kept)",
             "middle = (low + high) / 2",
             "above = [value for value in kept if value > middle]",
             "below = [value for value in kept if value <= middle]",
             "balance = (len(above) - len(below)) / len(kept)"]
    renamed = [step.replace("kept", "chosen").replace("value", "item") for step in steps]
    first = ("def first(values, limit, scale):\n" + "".join(f"    {s}\n" for s in steps)
             + "    return balance * scale\n")
    second = ("def second(values, limit):\n    values = sorted(values)\n"
              + "".join(f"    {s}\n" for s in renamed)
              + "    return {'balance': balance, 'kept': len(chosen)}\n")
    copied = tree(tmp_path, "copied", {"first.py": first, "second.py": second})
    assert gate(copied).returncode == 1
    assert owners(copies(copied)) == [("statements", ["first.py first", "second.py second"])]
    # Seven shared statements are two windows over one copy: the gate names the copy once.
    steps.append("spread = max(kept) - min(kept)")
    renamed.append("spread = max(chosen) - min(chosen)")
    first = ("def first(values, limit, scale):\n" + "".join(f"    {s}\n" for s in steps)
             + "    return balance * scale\n")
    second = ("def second(values, limit):\n    values = sorted(values)\n"
              + "".join(f"    {s}\n" for s in renamed)
              + "    return {'balance': balance, 'kept': len(chosen)}\n")
    longer = tree(tmp_path, "longer", {"first.py": first, "second.py": second})
    assert gate(longer).returncode == 1
    assert owners(copies(longer)) == [("statements", ["first.py first", "second.py second"])]


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


def test_declarations_alike_are_no_copy(tmp_path):
    """Two records' annotated fields and two parsers' arguments, alike in shape, declare different things;
    so do two modules' imports, two protocols' stubs and two runs of values one function reads."""
    record = "from dataclasses import dataclass\n\n\n@dataclass\nclass {0}:\n" + "".join(
        f"    {{1}}{index}: tuple[int, ...] | None = None\n" for index in range(8))
    parser = ("import argparse\n\n\ndef {0}():\n    parser = argparse.ArgumentParser()\n" + "".join(
        f"    parser.add_argument('--{{1}}{index}', type=int, default={index})\n" for index in range(8))
        + "    return parser.parse_args()\n")
    imports = "".join(f"from {{0}}{index} import {{0}}a, {{0}}b, {{0}}c, {{0}}d, {{0}}e\n"
                      for index in range(6))
    stubs = "class {0}:\n" + "".join(f"    @property\n    def {{1}}{index}(self) -> int: ...\n"
                                     for index in range(6))
    reads = "".join(f"{{0}}{index} = {{1}}('{{0}}{index}')\n" for index in range(6))
    alike = tree(tmp_path, "alike", {
        "left.py": record.format("Left", "a") + parser.format("left", "x"),
        "right.py": record.format("Right", "b") + parser.format("right", "y"),
        "left_imports.py": imports.format("left"), "right_imports.py": imports.format("right"),
        "left_stubs.py": stubs.format("LeftNames", "token"),
        "right_stubs.py": stubs.format("RightNames", "field"),
        "left_reads.py": "values = {}\n" + reads.format("width", "values.get"),
        "right_reads.py": "import re\n" + reads.format("PATTERN", "re.compile")})
    assert copies(alike) == []
    assert gate(alike).returncode == 0


def commit(root: Path, message: str) -> None:
    for command in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message]):
        subprocess.run(["git", *command], cwd=root, check=True)


def test_held_to_a_base_only_a_copy_the_base_lacked_fails(tmp_path, monkeypatch):
    """Main holds a copy (spread.py, ranges.py): a head that leaves it passes with the base and fails
    without; a head that writes the function a third time fails with the base too, naming the new copy."""
    root = tree(tmp_path, "repository",
                {"spread.py": SPREAD, "ranges.py": RANGE_OVER_COUNT, "other.py": "VALUE = 1\n"})
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    commit(root, "main")
    subprocess.run(["git", "checkout", "-q", "-b", "head"], cwd=root, check=True)
    (root / "other.py").write_text("VALUE = 2\n")
    commit(root, "a head that leaves the copy")
    monkeypatch.chdir(root)
    assert gate(root, "--base=main").returncode == 0 and duplicates.new(copies(root), "main") == []
    assert gate(root).returncode == 1
    (root / "third.py").write_text(SPREAD.replace("spread", "third"))
    commit(root, "a head that copies it again")
    assert gate(root, "--base=main").returncode == 1
    assert owners(duplicates.new(copies(root), "main")) == [
        ("function", ["ranges.py range_over_count", "spread.py spread", "third.py third"])]

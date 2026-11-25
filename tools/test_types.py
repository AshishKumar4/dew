"""No test a change touches reads an attribute its class never defines where it read none before: pyright's
attribute check (reportAttributeAccessIssue alone) over each test file HEAD adds, changes or renames since its
merge-base with BASE, run at HEAD and at that merge-base (a renamed file at its old path), each tree read
with its own pyproject.toml's pyright settings and its own source; a finding HEAD has more often than the
merge-base, by file and message, fails (owner, 2026-11-22: a dead field's read, `miss.frame`, passed every
run that never reached it). The tests are no declared type boundary and hold findings of their own (1,867
attribute reads pyright cannot prove on Dew's classes at 2026-11-22: a narrowed return type, a union's
members), which this reads as the base it compares with, never as a fault; a change that removes one read
and adds another of the same attribute on the same class in one file nets none.

From the checkout's root:

    python -m tools.test_types BASE

tools/check_gates runs it as the test-types gate.
"""

import argparse
import collections
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.check_tool_types import pyright_report

RULE = "reportAttributeAccessIssue"


def git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True, cwd=cwd).stdout


def touched(merge: str, cwd: Path | None = None) -> dict[str, str | None]:
    """Each test file HEAD adds, changes or renames since `merge`, by its path at HEAD: its path at `merge`,
    or None where HEAD adds it. A deleted file is none: nothing of it is read at HEAD."""
    fields = git("diff", "--name-status", "-z", "--find-renames", "--diff-filter=AMR", f"{merge}...HEAD",
                 "--", "tests", cwd=cwd).split("\0")
    found: dict[str, str | None] = {}
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        if status.startswith("R"):
            old, new = fields[index + 1], fields[index + 2]
            index += 3
        else:
            new = fields[index + 1]
            old = None if status == "A" else new
            index += 2
        if new.endswith(".py"):
            found[new] = old
    return found


def findings(tree: Path, files: dict[str, str], environment: Path) -> collections.Counter[tuple[str, str]]:
    """Pyright's attribute findings in `files` (each path in `tree` by the name it is counted under), each
    (its name, its message) by how often: `tree`'s own configuration (check_tool_types.py's
    `pyright_report`, its execution environments with it), every other rule off, its imports resolved in
    the virtual environment `environment` beneath the tree's own source (named in the configuration, which
    pyproject's own `venv` would otherwise name)."""
    tree = tree.resolve()
    present = {path: name for path, name in files.items() if (tree / path).is_file()}
    if not present:
        return collections.Counter()
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=tree, delete=False) as handle:
        written = Path(handle.name)
    try:
        diagnostics = pyright_report(tree, {"typeCheckingMode": "off", RULE: "error",
                                            "venvPath": str(environment.parent), "venv": environment.name},
                                     list(present), written)
    finally:
        written.unlink()
    return collections.Counter((present[str(Path(each["file"]).relative_to(tree))], each["message"])
                               for each in diagnostics if each.get("rule") == RULE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="the ref HEAD's change is read from (its merge-base with it)")
    options = parser.parse_args()
    top = Path(git("rev-parse", "--show-toplevel").strip())
    merge = git("merge-base", options.base, "HEAD").strip()
    files = touched(merge)
    if not files:
        return 0
    # The base's checkout beside this one, on its disk: never /tmp, memory on a planner's machine (bible §8).
    with tempfile.TemporaryDirectory(prefix="test-types-base-", dir=top.parent) as scratch:
        before = Path(scratch) / "base"
        git("worktree", "add", "-q", "--detach", str(before), merge, cwd=top)
        try:
            environment = Path(sys.prefix)
            new = findings(top, {name: name for name in files}, environment) - findings(
                before, {old: name for name, old in files.items() if old is not None}, environment)
        finally:
            git("worktree", "remove", "--force", str(before), cwd=top)
    for (name, message), count in sorted(new.items()):
        print(f"test-types: {name}: {message}" + (f" ({count} reads)" if count > 1 else ""))
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())

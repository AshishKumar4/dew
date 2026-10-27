"""tools/check.sh on a scratch checkout of the gate's own tools: its gates are tools/check_gates' lines, run
all side by side, one (`--gate`) or listed (`--list`); and --changed, the gates a head's own files decide,
where a head that touches SLOP020's inventory, or deletes a file it names, fails as the full run fails it."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tool_support import git

TOOLS = Path(__file__).resolve().parents[1] / "tools"
GATE = ("check.sh", "check_gates", "lint_inputs.sh", "base_history.sh", "lint_slop.py", "lint_scopes.py",
        "lint_tolerances.py", "lint_promotions.py", "duplicates.py", "forked.py")
# Twenty lines beside the comparison, so a copy with its comparison gone is still the same file to Git.
CLOSE = ("import numpy as np\n\n" + "".join(f"WIDTH_{index} = {index}\n" for index in range(20))
         + "\n\ndef test_close():\n    np.testing.assert_allclose(np.ones(2), np.ones(2), rtol=1e-6)\n")
HELD = {"tests/test_close.py": {"comparisons": 1, "lane": "Data", "by": "2099-01-01"}}


def inventory(files: dict) -> str:
    return json.dumps({"about": "SLOP020's held comparisons", "empty_by": "2099-01-01", "files": files})


def checkout(tmp_path: Path) -> Path:
    """A checkout whose main holds one test comparing floats within a tolerance, held by the inventory, with
    main as origin/main, so tools/base_history.sh fetches nothing, and a head branch checked out."""
    root = tmp_path / "checkout"
    for name in ("src/dew", "tools", "tests", "examples", "site"):
        (root / name).mkdir(parents=True)
    for name in GATE:
        shutil.copy(TOOLS / name, root / "tools" / name)
    (root / "tests/test_close.py").write_text(CLOSE)
    (root / "tools/tolerance_inventory.json").write_text(inventory(HELD))
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "main")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(root, "checkout", "-q", "-b", "head")
    return root


def check(root: Path, *arguments: str, **environ: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "tools/check.sh", *arguments], cwd=root, capture_output=True, text=True,
                          timeout=600, env={**os.environ, "PYTHON": sys.executable, **environ})


def changed(root: Path) -> subprocess.CompletedProcess:
    return check(root, "--changed")


COPIED = ("def {name}(values, limit):\n    total = 0\n    for index, value in enumerate(values):\n"
          "        if value > limit:\n            total += value * index\n        else:\n"
          "            total -= value\n    return total\n")


def test_the_registrys_gates_run_together_or_one_at_a_time(tmp_path):
    """check.sh runs tools/check_gates' gates: all of them together, its verdict any one's failure, or one by
    name (the lanes' lint gate runs each on a container of its own). Here the registry holds the real gates
    a scratch checkout has the tools for."""
    root = checkout(tmp_path)
    lines = (TOOLS / "check_gates").read_text().splitlines()
    (root / "tools/check_gates").write_text(
        "".join(f"{line}\n" for line in lines if line.split("\t")[0] in ("ratchet", "duplicate-structure")))
    git(root, "commit", "-qam", "the gates this checkout has the tools for")
    assert check(root, "--list").stdout.split() == ["ratchet", "duplicate-structure"]
    assert check(root).returncode == 0
    assert check(root, "--gate", "duplicate-structure").returncode == 0
    # One function written in two files: the structure gate fails, alone and in the whole run.
    for name in ("first", "second"):
        (root / "src/dew" / f"{name}.py").write_text(COPIED.format(name=name))
    git(root, "add", "-A")
    git(root, "commit", "-qm", "a head that writes one function twice")
    assert check(root, "--gate", "duplicate-structure").returncode == 1
    assert check(root, "--gate", "ratchet").returncode == 0
    assert check(root).returncode == 1
    assert check(root, "--gate", "a-gate-no-line-names").returncode != 0


def test_the_ratchet_gate_alone_fetches_the_history_it_reads_into_a_shallow_clone(tmp_path):
    """A gate run on its own container (the lanes' lint gate on armada) has a shallow checkout: the ratchet
    fetches main's history itself, and holds the inventory to the ref BASE names."""
    root = checkout(tmp_path)
    git(root, "checkout", "-q", "main")
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth=1", root.as_uri(), str(shallow))
    assert subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=shallow, capture_output=True,
                          text=True, check=True).stdout.strip() == "true"
    ratchet = check(shallow, "--gate", "ratchet")
    assert ratchet.returncode == 0, ratchet.stdout + ratchet.stderr
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=shallow, capture_output=True, text=True,
                          check=True).stdout.strip()
    assert check(shallow, "--gate", "ratchet", BASE=head).returncode == 0
    # Dropping the held comparison's entry loosens nothing; adding a held file the base lacks does.
    (shallow / "tools/tolerance_inventory.json").write_text(
        inventory({**HELD, "tests/test_other.py": {"comparisons": 1, "lane": "Data", "by": "2099-01-01"}}))
    git(shallow, "commit", "-qam", "a head that holds one more comparison")
    assert check(shallow, "--gate", "ratchet", BASE=head).returncode == 1


def test_a_head_touching_the_inventory_or_deleting_a_file_it_names_fails_its_changed_check(tmp_path):
    root = checkout(tmp_path)
    assert changed(root).returncode == 0
    # Uncommitted: the inventory no longer holds the comparison.
    (root / "tools/tolerance_inventory.json").write_text(inventory({}))
    assert changed(root).returncode == 1
    # Committed: the same.
    git(root, "commit", "-qam", "a head that drops the entry")
    assert changed(root).returncode == 1
    # A deleted file whose entry stays.
    git(root, "checkout", "-q", "-b", "deleting", "origin/main")
    (root / "tests/test_close.py").unlink()
    git(root, "commit", "-qam", "a head that deletes the held file")
    assert changed(root).returncode == 1
    # A file renamed with its comparisons gone, its old name's entry left: Git reads it as a rename.
    git(root, "checkout", "-q", "-b", "renaming", "origin/main")
    git(root, "mv", "tests/test_close.py", "tests/test_near.py")
    exact = "assert (np.ones(2) == np.ones(2)).all()"
    (root / "tests/test_near.py").write_text(
        CLOSE.replace("np.testing.assert_allclose(np.ones(2), np.ones(2), rtol=1e-6)", exact))
    git(root, "commit", "-qam", "a head that renames the held file and drops its comparison")
    renamed = subprocess.run(["git", "diff", "--name-status", "origin/main"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
    assert renamed.startswith("R")
    assert changed(root).returncode == 1

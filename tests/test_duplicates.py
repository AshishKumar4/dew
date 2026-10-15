"""tools/duplicates.py, the duplicate-code gate tools/check.sh runs, refuses a block two files copy and
passes the same files once the block has one owner, with the test environment's pylint."""

import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parents[1] / "tools" / "duplicates.py"

BLOCK = """\
    total = 0
    for index, value in enumerate(values):
        if value > limit:
            total += value * index
        else:
            total -= value
    return total
"""
"""Seven lines of logic, one past the gate's six."""


def gate(root: Path) -> int:
    """The gate's exit status over the files under `root`: pylint's message categories, as bits."""
    return subprocess.run([sys.executable, str(GATE), str(root)], capture_output=True, text=True,
                          timeout=300).returncode


def test_seven_lines_two_files_copy_fail_the_gate_and_six_or_one_owner_pass_it(tmp_path):
    copied = tmp_path / "copied"
    copied.mkdir()
    for name in ("first", "second"):
        (copied / f"{name}.py").write_text(f"def {name}(values, limit):\n{BLOCK}")
    # Refactor messages, duplicate-code among them, set bit 8.
    assert gate(copied) & 8
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "first.py").write_text(f"def first(values, limit):\n{BLOCK}")
    (owned / "second.py").write_text("from first import first\n\n\ndef second(values, limit):\n"
                                     "    return first(values, limit)\n")
    assert gate(owned) == 0
    six = tmp_path / "six"
    six.mkdir()
    (six / "first.py").write_text(f"def first(values, limit):\n{BLOCK}")
    # The last line differs, so the two share six: not past the bar.
    differing = BLOCK.replace("return total", "return -total")
    (six / "second.py").write_text(f"def second(values, limit):\n{differing}")
    assert gate(six) == 0

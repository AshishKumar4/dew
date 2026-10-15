"""tools/duplicates.py, the duplicate-code gate tools/check.sh runs, refuses a block two files copy and
passes the same files once the block has one owner. CI runs it beside the gate, in the gate's own
environment: `uv run --no-project --with pylint==4.1.3 --with pytest python -m pytest -q --noconftest
tests/test_duplicates.py`."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pylint")

GATE = Path(__file__).resolve().parents[1] / "tools" / "duplicates.py"

BLOCK = """\
    total = 0
    for index, value in enumerate(values):
        if value > limit:
            total += value * index
        elif value < -limit:
            total -= value
        else:
            total += 1
    return total
"""
"""Nine lines of logic, past the gate's eight."""


def gate(root: Path) -> int:
    """The gate's exit status over the files under `root`: pylint's message categories, as bits."""
    return subprocess.run([sys.executable, str(GATE), str(root)], capture_output=True, text=True,
                          timeout=300).returncode


def test_a_block_two_files_copy_fails_the_gate_and_one_owner_passes_it(tmp_path):
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

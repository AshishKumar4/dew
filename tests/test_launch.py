"""`dew launch`: the pool it starts, the commands it writes, and how it stops.

Each test runs the launcher as its own process against small Python
programs on this machine, so what is checked is what a shell sees: the exit
code, the output, and which processes are left alive.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def test_training_does_not_import_the_command_line():
    """The pool's variable names are shared with `prepare_process` through a
    module with no imports, so the training layer never loads the CLI."""
    probe = ("import sys, dew.training.runtime\n"
             "print(sorted(name for name in sys.modules if name.startswith('dew.cli')))\n")
    done = subprocess.run([sys.executable, "-c", probe], env=ENV, capture_output=True,
                          text=True, check=True)
    assert done.stdout.strip() == "[]"

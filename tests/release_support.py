"""A release's own code under the transformers it runs under: a check run in a fresh interpreter of its
own, that transformers layered over the suite's environment (`uv run --with`), so it runs wherever its test
does."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def under_release(requirement: str, check: Callable[..., None], *arguments: str | bool) -> None:
    """`check(*arguments)`, a module-level function of a test module, run in a fresh interpreter under
    `requirement` (the transformers a release's code runs under, `transformers==5.18.0`), layered over this
    interpreter's environment by `uv run --with`, Dew and the suite's modules importable and the Hub
    offline, as a test's own settings hold it; its failure, with its output, fails the test."""
    called = f"import {check.__module__} as module; module.{check.__qualname__}(*{arguments!r})"
    path = os.pathsep.join(filter(None, (str(ROOT / "tests"), str(ROOT / "src"),
                                         os.environ.get("PYTHONPATH"))))
    run = subprocess.run(["uv", "run", "--no-project", "--python", sys.executable, "--with", requirement,
                          "python", "-c", called], cwd=ROOT, capture_output=True, text=True, check=False,
                         env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": path})
    assert run.returncode == 0, f"{requirement}: {run.stdout[-4000:]}{run.stderr[-8000:]}"

"""The suite's Hypothesis profile (tests/conftest.py, docs/design/testing.md): a property test draws the same
examples on every run, so a red reproduces anywhere, runs the profile's count of them under its deadline, and
keeps nothing in the checkout, inside CI or out of it (Hypothesis loads a profile of its own on CI)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SITE = """\
import time

from hypothesis import given, strategies


@given(strategies.integers())
def test_drawn(value):
    with open({drawn!r}, "a") as out:
        out.write(f"{{value}}\\n")


@given(strategies.just(0))
def test_slow(value):
    with open({slow!r}, "a") as out:
        out.write("ran\\n")
    time.sleep(2.6)
"""


@pytest.mark.parametrize("ci", ["", "true"])
def test_a_property_test_runs_the_profiles_examples_alike_every_run_and_keeps_nothing_in_the_checkout(
        tmp_path, ci):
    drawn, slowed = tmp_path / "drawn.txt", tmp_path / "slow.txt"
    site = tmp_path / "test_site.py"
    site.write_text(SITE.format(drawn=str(drawn), slow=str(slowed)))
    tests = Path(__file__).resolve().parent
    env = {name: value for name, value in os.environ.items()
           if name not in ("CI", "HYPOTHESIS_STORAGE_DIRECTORY")}
    env = {**env, "PYTHONPATH": os.pathsep.join([str(tests), os.environ.get("PYTHONPATH", "")]),
           **({"CI": ci} if ci else {})}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider"]

    def ran(test: str) -> subprocess.CompletedProcess:
        return subprocess.run([*command, f"{site}::{test}"], cwd=tmp_path, env=env, capture_output=True,
                              text=True, timeout=300)

    runs = []
    for _ in range(2):
        drawn.unlink(missing_ok=True)
        done = ran("test_drawn")
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        runs.append(drawn.read_text().split())
    assert runs[0] == runs[1] and len(runs[0]) == 50, (len(runs[0]), len(runs[1]))
    # Its one example past the 2 s deadline, and the quarter more Hypothesis allows while it generates,
    # fails the test (which passes with no deadline), run again only to confirm it.
    slow = ran("test_slow")
    assert slow.returncode == 1 and 0 < len(slowed.read_text().split()) <= 3, slow.stdout[-2000:]
    assert not (tmp_path / ".hypothesis").exists()

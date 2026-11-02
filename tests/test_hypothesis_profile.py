"""The suite's Hypothesis profile (tests/conftest.py, docs/design/testing.md): a property test draws the same
examples on every run, so a red reproduces anywhere, and keeps no example database."""

import os
import subprocess
import sys
from pathlib import Path


def test_a_property_test_draws_the_same_examples_every_run_and_keeps_no_database(tmp_path):
    site = tmp_path / "test_site.py"
    drawn = tmp_path / "drawn.txt"
    site.write_text("from hypothesis import given, strategies\n\n\n"
                    "@given(strategies.integers())\n"
                    "def test_drawn(value):\n"
                    f"    with open({str(drawn)!r}, 'a') as out:\n"
                    "        out.write(f'{value}\\n')\n")
    tests = Path(__file__).resolve().parent
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tests), os.environ.get("PYTHONPATH", "")])}
    runs = []
    for _ in range(2):
        drawn.unlink(missing_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider"]
        done = subprocess.run([*command, str(site)], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=300)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        runs.append(drawn.read_text().split())
    assert runs[0] == runs[1] and len(set(runs[0])) > 1, runs
    assert not (tmp_path / ".hypothesis").exists()

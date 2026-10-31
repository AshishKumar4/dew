"""A plain pytest run's selection: CI's (tools/armada/session.py's `SELECTION`), the network-marked tests set
aside unless the run names its marks."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def collected(*arguments: str, addopts: str = "") -> list[str]:
    """The test ids pytest collects from a scratch file under tests/ (so tests/conftest.py loads; its name no
    test glob matches), run as a lane would run it, with `arguments`."""
    scratch = ROOT / "tests" / f"_selection_probe_{os.getpid()}.py"
    scratch.write_text("import pytest\n\n\ndef test_offline():\n    pass\n\n\n"
                       "@pytest.mark.network\ndef test_online():\n    pass\n")
    plain = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    if addopts:
        plain["PYTEST_ADDOPTS"] = addopts
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(scratch),
             *arguments], cwd=ROOT, capture_output=True, text=True, env=plain)
    finally:
        scratch.unlink()
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return [line.split("::")[1] for line in done.stdout.splitlines() if "::" in line]


def test_a_run_that_names_no_marks_sets_the_network_tests_aside_as_ci_does():
    """`pytest tests/test_x.py` collects what CI's tasks run: the offline test, not the network-marked one,
    which would download a real checkpoint (a container's disk filled on gpt-oss-20b's)."""
    assert collected() == ["test_offline"]


def test_a_run_that_names_its_marks_selects_them():
    """The network lane names its marks (`-m network`), and so may anyone: the default yields to them."""
    assert collected("-m", "network") == ["test_online"]
    assert collected("-mnetwork") == ["test_online"]
    assert collected("-m", "network or not network") == ["test_offline", "test_online"]
    assert collected("-m", "") == ["test_offline", "test_online"]
    assert collected(addopts="-m network") == ["test_online"]
    assert collected("-o", "addopts=-m network") == ["test_online"]

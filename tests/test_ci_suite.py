"""CI's suite gate, tools/ci/suite.sh, as it sends its armada jobs: run from a checkout with the armada CLI
stood in for, each call it makes recorded."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def suite_maps(checkout: Path, tmp_path: Path) -> list[list[str]]:
    """The words of each map tools/ci/suite.sh sends from `checkout`, its gangs run (no PROVEN, no GANGLESS),
    the armada CLI stood in for: each call's words, one call a line; its verdict green, so no suite runs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "armada").write_text(f'#!/bin/sh\necho "$*" >> {tmp_path}/calls\ncat > /dev/null\n')
    (bin_dir / "armada").chmod(0o755)
    if not (bin_dir / "python3").exists():
        (bin_dir / "python3").symlink_to(sys.executable)
    (tmp_path / "calls").unlink(missing_ok=True)
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "ARMADA": "armada",
                   "OUT": str(tmp_path / "out")}
    for name in ("PROVEN", "GANGLESS", "DEW_WHOLE_SUITE"):
        environment.pop(name, None)
    ran = subprocess.run(["bash", "tools/ci/suite.sh", "a" * 40], cwd=checkout, capture_output=True,
                         text=True, timeout=300, env=environment)
    maps = [line.split() for line in (tmp_path / "calls").read_text().splitlines() if line.startswith("map ")]
    assert maps, ran.stdout + ran.stderr
    return maps


def test_every_map_of_cis_suite_runs_on_the_cpus_its_tasks_are_pinned_to(tmp_path):
    """CI's suite gate (tools/ci/suite.sh) runs every armada job it sends on the CPUs .armada.json pins its
    tasks to (task.cpu): its multi-host gangs as its suite and release checks do, as a CPU's arithmetic is its
    own (2026-11-10: tasks of maps that named no CPU landed on AMD 25/1 x86-64-v3 hosts, where a dense sum and
    a float64 rsqrt round apart from the pinned models'); and a checkout that pins none names none."""
    cpu = json.loads((ROOT / ".armada.json").read_text())["task"]["cpu"]
    pinned = ",".join(f"{each['vendor']}/{each['family']}/{each['model']}"
                      for each in ([cpu] if isinstance(cpu, dict) else cpu))
    maps = suite_maps(ROOT, tmp_path / "pinned")
    assert all(f"--cpu={pinned}" in words for words in maps), maps
    # A checkout of the same tools whose .armada.json pins no CPU.
    unpinned = tmp_path / "unpinned"
    shutil.copytree(ROOT / "tools", unpinned / "tools")
    (unpinned / "tests").mkdir()
    shutil.copy(ROOT / "tests/test_multihost.py", unpinned / "tests/test_multihost.py")
    shutil.copy(ROOT / ".python-version", unpinned / ".python-version")
    config = json.loads((ROOT / ".armada.json").read_text())
    del config["task"]["cpu"]
    (unpinned / ".armada.json").write_text(json.dumps(config))
    maps = suite_maps(unpinned, tmp_path / "unpinned-calls")
    assert not any(word.startswith("--cpu") for words in maps for word in words), maps

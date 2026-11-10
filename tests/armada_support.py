"""The tools of the armada CI test modules (tests/test_armada_*.py): Dew's CI on armada
(tools/armada/ci.py) run on small test files written into a checkout of their own."""

import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from tool_support import git, load

ROOT = Path(__file__).resolve().parents[1]


def loaded(tmp_path: Path, monkeypatch):
    """tools/armada/ci.py, run from the checkout at `tmp_path`, with no test cache named and none of the
    armada task these tests run in: CI's own environment names the shared test cache, and the task's own
    artifacts and the progress of its recalled attempt (ARMADA_ARTIFACTS, ARMADA_RESUME), which these runs
    of the tools must neither read nor write (a recalled attempt's nested tasks set aside the tests its
    progress named, and recorded none); their own task's artifacts under `tmp_path`."""
    module = load("ci", "tools/armada")
    monkeypatch.chdir(tmp_path)
    apart(monkeypatch)
    # Each test's task keeps its artifacts (a stall's record, `stalls.kept`) under its own directory.
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    (tmp_path / "tests").mkdir()
    return module


APART = ("DEW_XLA_CACHE_URL", "DEW_XLA_CACHE_TOKEN", "DEW_XLA_CACHE_READ_TOKEN", "DEW_TEST_CACHE",
         "ARMADA_ARTIFACTS", "ARMADA_RESUME")
"""What CI's environment names that these tests' runs of its tools must not see (`apart`)."""


def apart(monkeypatch) -> None:
    """The environment these tests' runs of CI's tools see: no shared test cache, and none of the armada task
    the tests run in (`loaded`)."""
    for name in APART:
        monkeypatch.delenv(name, raising=False)


def interpreter(root: Path, version: str) -> None:
    """The checkout's `.venv-<version>` python and its py-spy, as the CI image has them: here, this
    interpreter and its environment's."""
    venv = root / f".venv-{version}/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    (venv / "py-spy").symlink_to(Path(sys.executable).with_name("py-spy"))


def written(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        (root / "tests" / name).write_text(source)


def cached(monkeypatch, url: str) -> None:
    """The shared test cache at `url`, as CI's environment names it."""
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")


def lifetime_task(ci, tmp_path, monkeypatch, cache_worker, files: dict[str, str],
                  sources: dict[str, str]) -> Callable[[], list[bool]]:
    """`files` under tests/ and `sources` under src/ written, run as one 3.12 task with the test cache
    on: whether each test file's row is reused now, in the files' order."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    written(tmp_path, files)
    (tmp_path / "src").mkdir(exist_ok=True)
    for name, source in sources.items():
        (tmp_path / "src" / name).write_text(source)
    interpreter(tmp_path, "3.12")
    tests = [f"tests/{name}" for name in files if name.startswith("test_")]
    ci.task("3.12", tests, "", tmp_path / "verdict.json", 600.0)
    rows = json.loads((tmp_path / "verdict.json").read_text())["rows"]
    assert all(row["exitCode"] == 0 for row in rows), rows
    return lambda: reusable_now(ci, tests)


def reusable_now(ci, tests: list[str]) -> list[bool]:
    """Whether each of `tests` would reuse its 3.12 row now."""
    common = ci.shared("3.12")
    return [ci.reused("3.12", name, "", common) is not None for name in tests]


ON_SRC = "import sys\nsys.path.insert(0, 'src')\n"
PASSING = {"test_a.py": "def test_a():\n    pass\n", "test_b.py": "def test_b():\n    pass\n"}


def commit(root: Path, message: str) -> str:
    """Checkout `root`'s changes committed, and the commit's sha: every file at the first, which makes the
    repository, and the tracked and staged changes after, so what a run leaves in the checkout stays out."""
    first = not (root / ".git").exists()
    if first:
        git(root, "init", "-q")
        git(root, "add", "-A")
    git(root, "commit", "-qm" if first else "-qam", message)
    return git(root, "rev-parse", "HEAD").strip()


def armada_task(ci, monkeypatch, out: Path, policy: str = "on", target: float = 300.0,
                census: bool = False) -> list[dict]:
    """The 3.12 tasks the plan makes under DEW_TEST_CACHE=`policy` for tasks of `target` seconds, the census's
    with `census` (`plan --census`), each run as armada starts it: .armada.json's command filled from its plan
    entry, in .armada.json's environment, which the plan's DEW_TEST_CACHE is not part of. Their rows, in the
    plan's order."""
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    monkeypatch.setenv("DEW_TEST_CACHE", policy)
    entries = [entry for entry in ci.plan(target, {}, census=census) if entry["python"] == "3.12"]
    monkeypatch.delenv("DEW_TEST_CACHE")
    rows = []
    for entry in entries:
        values = {**entry, "out": out}
        filled = [re.sub(r"\{(\w+)\}", lambda field, values=values: str(values[field[1]]), word)
                  for word in command]
        monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
        assert ci.main() == 0
        rows += json.loads(out.read_text())["rows"]
    return rows


def cached_checkout(ci, root: Path, monkeypatch, cache_worker) -> Path:
    """Checkout `root` run by 3.12 tasks on one CPU class against the shared test cache, and where their
    verdict is written: beside the checkout's entries, not among them (the checkout's own directory is an
    input)."""
    interpreter(root, "3.12")
    url, _, _ = cache_worker
    monkeypatch.setattr(ci, "cpu_class", lambda: "GenuineIntel-6-143")
    cached(monkeypatch, url)
    (root / ".verdict").mkdir()
    return root / ".verdict/verdict.json"


def fetcher(root: Path, *, due: bool = False) -> Path:
    """A stand-in tools/fetched_fixtures.py under `root`: its fetch records that it ran, and with `due`
    refuses as the real one does a pin whose upload is due."""
    (root / "tools").mkdir(exist_ok=True)
    (root / "tools/fetched_fixtures.py").write_text(
        "import sys, time\nfrom pathlib import Path\n"
        "assert sys.argv[1:] == ['fetch']\n"
        "Path('fetched').write_text(str(time.monotonic()))\n"
        + ("raise SystemExit('x.npz: pinned with no revision: its upload is due')\n" if due else ""))
    return root / "fetched"


def gpu_driver(root: Path, monkeypatch) -> None:
    """nvidia-smi as the driver answers it for an 80 GiB GPU, first on PATH."""
    tools = root / "bin"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 81920\n")
    (tools / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")


LANE_TEST = ("import os, subprocess, sys, time\n"
             "from pathlib import Path\n"
             "def test_runs():\n"
             "    began = time.monotonic()\n"
             "    time.sleep(1)\n"
             "    share = os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']\n"
             "    grown = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')\n"
             "    threads = os.environ.get('OMP_NUM_THREADS', 'all')\n"
             "    Path(f'threads-{Path(__file__).stem}').write_text(threads)\n"
             "    ran = Path(f'ran-{Path(__file__).stem}-{os.getpid()}')\n"
             "    ran.write_text(f'{share} {began} {time.monotonic()} {grown}')\n")
"""A GPU lane's test file standing in: it records the share it was given, when it ran and whether its pool
grows."""


def gpu_lane_at(root: Path, monkeypatch, url: str, files: int) -> list[str]:
    """A GPU lane's machine at `root`: the test cache at `url`, half of one 80 GiB GPU, its session growing
    every job's pool as a GPU node's does (dewops' settings), and `files` test files of `LANE_TEST`. Their
    names."""
    cached(monkeypatch, url)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gpu_driver(root, monkeypatch)
    names = [f"tests/test_file_{index}.py" for index in range(files)]
    for name in names:
        (root / name).write_text(LANE_TEST)
    return names

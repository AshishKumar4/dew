"""The processes a test file starts, as executed.py's records count them: a native tool's start keeps the
file's key (`TOOLS`), a Python child that records itself is followed, and any other start leaves the
file without a key, as code no event sees ran.

A child is this checkout's Python started with an environment that names DEW_EXECUTED_CHILDREN, the
parent's or one passed on with it. The `.pth` the parent installs in the environment (`installed`)
runs `follow` as the child starts: it records the checkout's files the child executed, where it read
the backend, the processes it started in turn, and the GPU pool it held, and writes them as it exits.
A child that ends without its record (killed, or leaving through os._exit) leaves its start mark
behind, and the parent then counts its file's starts as unfollowed (`gathered`), as it does a child
whose record names another file or a count of records other than the starts it saw."""

import atexit
import dis
import json
import os
import subprocess
import sys
import sysconfig
import time
import types
from pathlib import Path

TOOLS = {"git", "nvidia-smi", "uname", "file", "ptxas"}
"""The programs a test file may start and keep its key: git (a run's commit), nvidia-smi (the GPUs),
uname and file (the `platform` module's answers) and ptxas (a GPU kernel's PTX compiled to machine
code, on the host alone). Any other start, a shell's or a launcher's included, may run the
checkout's code or hold the GPU."""
SPAWNS = {"subprocess.Popen", "_posixsubprocess.fork_exec", "os.posix_spawn", "os.spawn", "os.exec",
          "os.fork", "os.forkpty"}
"""Audit events of a process start; a multiprocessing child's is executed.py's `CHILD_START`'s."""
CHILDREN = "DEW_EXECUTED_CHILDREN"
"""Where a followed child writes its record and its start mark, and what makes it one."""
FOLLOWER_TOOL = 4
"""The sys.monitoring tool a child records with: executed.py's pytest process holds 3, and a task's
pytest is itself a followed child of a test that runs one."""
PTH = "dew_followed.pth"
GPUS = ("cuda", "rocm")
OPENED = ("_init_backend", os.path.join("jax", "_src", "xla_bridge.py"))
"""Where jax opens a backend, by its platform argument: a child that opens a GPU's holds a pool there.
jaxlib 0.11.2 announces no backend's opening in public: jax.monitoring records no event for it, and
jax.extend.backend's `backends()` and `get_backend()` open every backend they report, so asking them
would itself take a pool. This private function's name and its `platform` argument are the one private
use of another library in tools/armada; the pool is then read through `jax.devices()`."""
FRACTION = "XLA_PYTHON_CLIENT_MEM_FRACTION"
decided: dict[types.CodeType, bool] = {}


def checkout(name: str, root: str) -> bool:
    """Whether `name` is a Python file of the checkout at `root`. The environment's packages, which CI
    installs in the checkout, are the locks' to key."""
    return name.startswith(root) and name.endswith(".py") and "site-packages" not in name


def reads_backend(code: types.CodeType) -> bool:
    """Whether `code` calls `default_backend` (jax's), by any name table, or reads a device's
    `platform`: an attribute of anything but the `sys` and `platform` modules."""
    if code not in decided:
        steps = list(dis.get_instructions(code))
        decided[code] = any(
            step.argval == "default_backend"
            or (step.opname == "LOAD_ATTR" and step.argval == "platform" and not (
                index and steps[index - 1].opname in ("LOAD_GLOBAL", "LOAD_NAME")
                and steps[index - 1].argval in ("sys", "platform")))
            for index, step in enumerate(steps))
    return decided[code]


def programs(event: str, args: tuple) -> list:
    """The programs a process start of `event` may run, by the paths its audit names: `subprocess`'s
    executable (its command's first word where it names none), `_posixsubprocess`'s candidates, and
    none for a fork, whose child runs this process's own code."""
    if event in ("subprocess.Popen", "os.posix_spawn", "os.exec"):
        return [args[0]]
    if event == "_posixsubprocess.fork_exec":
        return list(args[0] or ())
    if event == "os.spawn":
        return [args[1]]
    return []


def native(names: list) -> bool:
    """Whether every one of `names`, and at least one, is a program of `TOOLS`. A start that names
    its program by a file descriptor (`os.execve`'s), or by nothing, is no tool's."""
    return bool(names) and all(isinstance(name, (str, bytes, os.PathLike))
                               and os.path.basename(os.fsdecode(name)) in TOOLS for name in names)


def start(event: str, args: tuple, directory: str) -> str | None:
    """How a process start the audit names counts for its test file: None for none of its own (a
    native tool, or `subprocess`'s own start beneath the Popen its audit already named), "followed"
    for a child that records itself into `directory`, else "unfollowed"."""
    if event not in SPAWNS:
        return None
    if event != "subprocess.Popen" and _beneath_popen():
        return None
    if native(programs(event, args)):
        return None
    if event == "subprocess.Popen":
        program, environment = args[0], args[3]
        named = (os.environ if environment is None else environment).get(CHILDREN)
        if named == directory and _this_python(program):
            return "followed"
    return "unfollowed"


def _beneath_popen() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename == subprocess.__file__:
            return True
        frame = frame.f_back
    return False


def _this_python(program) -> bool:
    if not isinstance(program, (str, bytes, os.PathLike)):
        return False
    path = os.fsdecode(program)
    return os.path.realpath(path if os.sep in path else os.path.join(sys.prefix, "bin", path)) == \
        os.path.realpath(sys.executable)


def installed() -> None:
    """The `.pth` that starts `follow` in every Python of this environment launched with
    DEW_EXECUTED_CHILDREN set, and in no other: written once, the same for every checkout, as the
    environment's checkouts share it."""
    line = (f"import os; os.environ.get({CHILDREN!r}) and __import__('runpy').run_path("
            f"os.environ['DEW_EXECUTED_FOLLOWER'], run_name='dew_followed')\n")
    path = Path(sysconfig.get_paths()["purelib"]) / PTH
    if not path.is_file() or path.read_text() != line:
        written = path.with_name(f".{PTH}.{os.getpid()}")
        written.write_text(line)
        written.replace(path)


def follow() -> None:
    """Record this child for the process that started it, as it exits."""
    directory, root = Path(os.environ[CHILDREN]), os.environ["DEW_EXECUTED_ROOT"]
    mark = directory / f"{os.getpid()}.started"
    # A child given its parent's fraction holds one more of the parent's shares where it opens the GPU.
    inherited = os.environ.get(FRACTION, "") == os.environ.get("DEW_EXECUTED_SHARE", "")
    record = {"file": os.environ.get("DEW_EXECUTED_FILE", ""), "files": set(), "reads": set(), "expected": 0,
              "unfollowed": False, "began": time.time(), "pool": None, "inherited": inherited}
    mark.write_text(record["file"])

    def started(code, offset):
        if code.co_name == OPENED[0] and code.co_filename.endswith(OPENED[1]):
            # Registered after jax's own clean-up, so it reads the pool before the backend closes.
            if sys._getframe(1).f_locals.get("platform") in GPUS:
                atexit.register(held)
            return None
        name = code.co_filename
        if checkout(name, root):
            record["files"].add(name[len(root):])
            if reads_backend(code):
                record["reads"].add(f"{name[len(root):]}:{code.co_qualname}")
        return sys.monitoring.DISABLE

    def held() -> None:
        import jax

        try:
            devices = jax.local_devices(backend="gpu")
        except RuntimeError:  # the backend failed to open, and holds no pool
            return
        stats = [device.memory_stats() or {} for device in devices]
        record["pool"] = sum(stat.get("peak_pool_bytes", stat.get("bytes_limit", 0)) for stat in stats)

    def audited(event, args):
        kind = start(event, args, str(directory))
        if kind == "followed":
            record["expected"] += 1
        elif kind == "unfollowed":
            record["unfollowed"] = True

    def written() -> None:
        ended = {**record, "files": sorted(record["files"]), "reads": sorted(record["reads"]),
                 "ended": time.time()}
        (directory / f"{os.getpid()}.json").write_text(json.dumps(ended))
        mark.unlink()

    sys.monitoring.use_tool_id(FOLLOWER_TOOL, "dew-followed")
    sys.monitoring.register_callback(FOLLOWER_TOOL, sys.monitoring.events.PY_START, started)
    sys.monitoring.set_events(FOLLOWER_TOOL, sys.monitoring.events.PY_START)
    sys.addaudithook(audited)
    atexit.register(written)


def gathered(directory: Path, expected: dict[str, int]) -> dict[str, dict]:
    """What each test file's children recorded in `directory`, by the file that started them, given
    how many children each file's own process started (`expected`): their files and reads, whether
    every start was followed, and the GPU pools they held at once, the parent's share counted apart
    (`shares`, the children that took its fraction) from the bytes of those that named their own."""
    records = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    lost = {path.read_text() for path in directory.glob("*.started")}
    found: dict[str, dict] = {}
    for name in {*expected, *(record["file"] for record in records)}:
        own = [record for record in records if record["file"] == name]
        starts = expected.get(name, 0) + sum(record["expected"] for record in own)
        pooled = [record for record in own if record["pool"]]
        found[name] = {
            "files": {path for record in own for path in record["files"]},
            "reads": {site for record in own for site in record["reads"]},
            "followed": name not in lost and len(own) == starts and not any(r["unfollowed"] for r in own),
            "shares": _most_at_once([r for r in pooled if r["inherited"]], lambda record: 1),
            "pools": _most_at_once([r for r in pooled if not r["inherited"]], lambda record: record["pool"])}
    return found


def _most_at_once(records: list[dict], weight) -> int:
    """The most of `weight` the `records` held at once, over their lives."""
    events = sorted([(record["began"], weight(record)) for record in records]
                    + [(record["ended"], -weight(record)) for record in records])
    most = held = 0
    for _, change in events:
        held += change
        most = max(most, held)
    return most


if __name__ == "dew_followed":
    follow()

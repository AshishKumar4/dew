"""The processes a test file starts, as executed.py's records count them: a native tool's start keeps the
file's key (`TOOLS`), a Python child that records itself is followed, and any other start leaves the
file without a key, as code no event sees ran.

A child is this checkout's Python started with an environment that names DEW_EXECUTED_CHILDREN, the
parent's or one passed on with it. The `.pth` the parent installs in the environment (`installed`)
runs `follow` as the child starts: it records the checkout's files the child executed and read
(`read`, as the parent records its own), where it read the backend, the processes it started in turn,
and the GPU pool it held, and writes them as it exits.
A child marks its start, and one that ends without its record (killed, or leaving through os._exit)
has a mark and no record, and the parent then counts its file's starts as unfollowed (`gathered`), as
it does a child whose record names another file or a count of records other than the starts it saw.
It imports the standard library alone, since the `.pth` runs it in every child before the child's own
imports, so its two writes publish by a rename of their own rather than through dew.files."""

import atexit
import dis
import functools
import json
import os
import pathlib
import subprocess
import sys
import sysconfig
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path

TOOLS = {"git", "nvidia-smi", "uname", "file", "ptxas"}
"""The programs a test file may start and keep its key: git (a run's commit), nvidia-smi (the GPUs),
uname and file (the `platform` module's answers) and ptxas (a GPU kernel's PTX compiled to machine
code, on the host alone). Any other start, a shell's or a launcher's included, may run the
checkout's code or hold the GPU."""
SPAWNS = {"subprocess.Popen", "_posixsubprocess.fork_exec", "os.posix_spawn", "os.spawn", "os.exec",
          "os.fork", "os.forkpty"}
"""Audit events of a process start; a multiprocessing child's is executed.py's `CHILD_START`'s."""
HIDDEN_INPUTS = (".github", ".armada.json", ".python-version", ".gitattributes")
"""The checkout's hidden entries a test may read as any of its files (a workflow, the run's settings), each
the input of the files that read it alone; any other hidden entry (.git, an environment, a cache) is none."""
READS = {"open", "os.listdir", "os.scandir"}
LIMITS = {"cpu.max": "limit:cpu", "memory.max": "limit:memory"}
"""The cgroup files that set a process's execution limits, by the input a run that read one keys: host.py's
`LIMITS`, which this module cannot import (it runs inside every test process, before the checkout's tools)."""
"""Audit events of a read: a file opened, a directory listed."""
IMPORTING = {"<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>", "<frozen zipimport>"}
"""The import system's frozen modules, which list the directories it imports from."""
CHILDREN = "DEW_EXECUTED_CHILDREN"
"""Where a followed child writes its record and its start mark, and what makes it one."""
FOLLOWER_TOOL = 4
"""The sys.monitoring tool a child records with: executed.py's pytest process holds 3, and a task's
pytest is itself a followed child of a test that runs one."""
FOLLOWER = "dew-followed"
"""The recorder's name for its tool."""
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
placing = threading.local()
"""Whether the `os.open` this thread is making opens relative to a directory descriptor (`opening`)."""


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


def read(event: str, args: tuple, root: str) -> str | None:
    """The checkout's file or directory, relative to `root`, that a read event names, or None: a file
    opened only to read (`open`'s mode, or its flags for `os.open`), or a directory listed but by a
    listing tool (`listing_tool`), the code that listed it two frames up, beneath the audit hook that
    calls this. An open relative to a directory descriptor is `opening`'s to place. An open has no
    caller to ask about: the interpreter audits the main script's from no frame at all."""
    path = args[0]
    if isinstance(path, int):
        return None
    if event == "open":
        mode, flags = args[1], args[2]
        written = any(letter in mode for letter in "wax+") if mode is not None else (
            flags & os.O_ACCMODE != os.O_RDONLY)
        if written or (mode is None and getattr(placing, "relative", False)):
            return None
    elif listing_tool(sys._getframe(2).f_code.co_filename):
        return None
    name = os.path.abspath(os.fsdecode(path if path is not None else "."))
    # A cgroup's CPU quota or memory limit read: the run keys the limit that binds it (host.py's `limit`).
    cgroup = os.environ.get("DEW_CGROUP_ROOT", "/sys/fs/cgroup")
    if os.path.basename(name) in LIMITS and name.startswith(cgroup.rstrip(os.sep) + os.sep):
        return LIMITS[os.path.basename(name)]
    return placed(name, root)


def placed(name: str, root: str) -> str | None:
    """Absolute path `name` relative to the checkout at `root`, its own directory as `.`, or None
    where it is not the checkout's or is never an input (a hidden entry but `HIDDEN_INPUTS`, compiled
    bytecode, the environment's packages)."""
    if name == root.removesuffix(os.sep):
        return "."
    if not name.startswith(root):
        return None
    relative = name[len(root):]
    top = relative.split(os.sep)[0]
    hidden = top.startswith(".") and top not in HIDDEN_INPUTS
    if hidden or "__pycache__" in relative or "site-packages" in relative:
        return None
    return relative


PROBES = ((os.path, ("exists", "lexists", "isfile", "isdir", "getsize")),
          (pathlib.Path, ("exists", "is_file", "is_dir", "stat")))
"""The existence and size probes a run's record keeps as reads (`opening`), which raise no audit event:
`os.path`'s and `pathlib.Path`'s, which every test, collection, conftest, loader and followed child calls."""


def opening(recorded: Callable[[str], None], root: str) -> None:
    """`os.open` wrapped, whose audit event names a path relative to `dir_fd` without the descriptor:
    such an open, once it has opened to read, is passed to `recorded` as what the descriptor it
    opened names (`placed`). Each existence or size probe (`PROBES`) is passed to `recorded` as a read of
    the path it asks after, there or not, whose contents the record then keys (ci.py's `contents`): a
    fixture's arrival reruns its probers, and no other file (Main, 2026-10-29)."""
    for owner, names in PROBES:
        for name in names:
            setattr(owner, name, probed(getattr(owner, name), recorded, root))
    plain = os.open

    def opened(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            return plain(path, flags, mode)
        placing.relative = True
        try:
            descriptor = plain(path, flags, mode, dir_fd=dir_fd)
        finally:
            placing.relative = False
        if flags & os.O_ACCMODE == os.O_RDONLY and (name := placed(
                os.readlink(f"/proc/self/fd/{descriptor}"), root)) is not None:
            recorded(name)
        return descriptor

    os.open = opened


def listing_tool(name: str) -> bool:
    """Whether code in file `name` lists directories for the import system, for its search of the
    installed distributions (the locks' to key) or for pytest's collection, whose listings are no
    test's input: what they find is recorded as it runs. The rest of the standard library's
    frozen modules (`os.walk`'s) list for whoever calls them."""
    return (name in IMPORTING or f"{os.sep}_pytest{os.sep}" in name
            or f"{os.sep}importlib{os.sep}metadata{os.sep}" in name)


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


def probed(probe: Callable[..., object], recorded: Callable[[str], None], root: str) -> Callable[..., object]:
    """`probe` (`PROBES`), which passes the path it is asked about to `recorded` before it answers."""

    @functools.wraps(probe)
    def asked(path, *arguments, **keywords):
        if not isinstance(path, int) and (name := placed(os.path.abspath(os.fsdecode(os.fspath(path))),
                                                          root)) is not None:
            recorded(name)
        return probe(path, *arguments, **keywords)

    return asked


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
    """Record this child for the process that started it, as it exits: once, though the environment
    processes its site-packages' `.pth` files twice (a uv environment's `site` adds the directory
    again), so a second start finds the recorder holding its tool and leaves it. Where another tool
    holds that id the child records no files, and says so (`unfollowed`): its file then has no key,
    and the child starts as it would without the recorder."""
    holder = sys.monitoring.get_tool(FOLLOWER_TOOL)
    if holder == FOLLOWER:
        return
    directory, root = Path(os.environ[CHILDREN]), os.environ["DEW_EXECUTED_ROOT"]
    mark = directory / f"{os.getpid()}.started"
    # A child given its parent's fraction holds one more of the parent's shares where it opens the GPU.
    inherited = os.environ.get(FRACTION, "") == os.environ.get("DEW_EXECUTED_SHARE", "")
    record = {"file": os.environ.get("DEW_EXECUTED_FILE", ""), "files": set(), "reads": set(), "expected": 0,
              "unfollowed": False, "began": time.time(), "pool": None, "peak": None, "inherited": inherited}
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
        except RuntimeError as error:  # the backend failed to open, and holds no pool
            record["unopened"] = str(error)
            return
        stats = [device.memory_stats() or {} for device in devices]
        record["pool"] = sum(stat.get("peak_pool_bytes", stat.get("bytes_limit", 0)) for stat in stats)
        record["peak"] = max(stat.get("peak_bytes_in_use", 0) for stat in stats)

    def audited(event, args):
        if event in READS:
            name = read(event, args, root)
            if name is not None:
                record["files"].add(name)
            return
        kind = start(event, args, str(directory))
        if kind == "followed":
            record["expected"] += 1
        elif kind == "unfollowed":
            record["unfollowed"] = True

    def written() -> None:
        ended = {**record, "files": sorted(record["files"]), "reads": sorted(record["reads"]),
                 "ended": time.time()}
        # Whole or not at all, beside the mark, which stays: the parent may read either as this child ends.
        partial = directory / f"{os.getpid()}.partial"
        partial.write_text(json.dumps(ended))
        partial.replace(directory / f"{os.getpid()}.json")

    atexit.register(written)
    if holder is not None:
        record["unfollowed"] = True
        return
    opening(record["files"].add, root)
    sys.monitoring.use_tool_id(FOLLOWER_TOOL, FOLLOWER)
    sys.monitoring.register_callback(FOLLOWER_TOOL, sys.monitoring.events.PY_START, started)
    sys.monitoring.set_events(FOLLOWER_TOOL, sys.monitoring.events.PY_START)
    sys.addaudithook(audited)


def gathered(directory: Path, expected: dict[str, int]) -> dict[str, dict]:
    """What each test file's children recorded in `directory`, by the file that started them, given
    how many children each file's own process started (`expected`): their files and reads, whether
    every start was followed, the most GPU memory a child that took the parent's fraction held
    (`peak`), and the GPU pools they held at once, the parent's share counted apart (`shares`, the
    children that took its fraction) from the bytes of those that named their own. Records are read
    before marks, so a child that ends between the two counts as one without its record."""
    published = list(directory.glob("*.json"))
    records = [json.loads(path.read_text()) for path in published]
    ended = {path.stem for path in published}
    lost = {path.read_text() for path in directory.glob("*.started") if path.stem not in ended}
    found: dict[str, dict] = {}
    for name in {*expected, *(record["file"] for record in records)}:
        own = [record for record in records if record["file"] == name]
        starts = expected.get(name, 0) + sum(record["expected"] for record in own)
        pooled = [record for record in own if record["pool"]]
        inherited = [record for record in pooled if record["inherited"]]
        found[name] = {
            "files": {path for record in own for path in record["files"]},
            "reads": {site for record in own for site in record["reads"]},
            "followed": name not in lost and len(own) == starts and not any(r["unfollowed"] for r in own),
            "peak": max((record["peak"] for record in inherited if record["peak"] is not None), default=None),
            "shares": _most_at_once(inherited, lambda record: 1),
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

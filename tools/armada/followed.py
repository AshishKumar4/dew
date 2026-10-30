"""The processes a test file starts, as executed.py's records count them: a native tool's start keeps the
file's key (`TOOLS`), a Python child that records itself is followed, and any other start leaves the
file without a key, as code no event sees ran.

A child is this checkout's Python started with an environment that names DEW_EXECUTED_CHILDREN, the
parent's or one passed on with it. The `.pth` the parent installs in the environment (`installed`)
runs `follow` as the child starts: it logs the checkout's files the child executed and read (`read`, as
the parent logs its own, lifetime.py), the modules it imported, where it read the backend, the processes it
started in turn and the GPU pool it held, each as it happens, and a closing line as its last atexit runs.
A child whose log never closed (killed, or leaving through os._exit) is lost, and the parent then counts
its file's starts as unfollowed (`gathered`), as it does a count of logs other than the starts it saw.
It imports the standard library alone, since the `.pth` runs it in every child before the child's own
imports."""

import atexit
import dis
import functools
import importlib.machinery
import os
import pathlib
import subprocess
import sys
import sysconfig
import threading
import time
import types
from collections import Counter
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
SHARED = ""
"""The owner of what runs for every file: before the first test, around them, and after the last."""
CLOSED = "closed"
"""The last line of a log the process wrote to its end: its atexit, registered before any other."""
READ = "read:"
"""What a read of the backend, or a test's mark of session.py's `MARKS`, is logged as: the GPU tier's
evidence (ci.py's `gpu_tier`), no input."""
STARTED = "start:"
"""What a process start is logged as: `start:followed` for a child that records itself (followed.py),
`start:unfollowed` for one no event sees."""
PEAK = "peak:"
"""What the most GPU memory the process held is logged as, read before jax's backend closes."""
VISIT = "visit"
"""A switch of the owner of what runs (`switch`): a file's tests begin, or the session's end does."""
EVIDENCE = (READ, STARTED, PEAK, CLOSED, VISIT)
"""What a log holds beside the record's inputs."""
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


PROBES = ((os, ("stat", "lstat", "access")),
          (os.path, ("exists", "lexists", "isfile", "isdir", "getsize")),
          (pathlib.Path, ("exists", "is_file", "is_dir", "stat")))
"""The existence and size probes a run's record keeps as reads (`opening`), which raise no audit event:
`os`'s stat calls, `os.path`'s (getmtime and islink among them, through os.stat) and `pathlib.Path`'s,
which every test, collection, conftest, loader and followed child calls. A glob's wildcard parts list the
directories they walk (`read`), so a file arriving where a glob looks changes a listing it keyed."""
PROBED = "probe:"
"""What a probed path's name is recorded under: a probe's answer is whether the path is there and of
what kind, and a file's size, so ci.py's `contents` keys a probed directory by its kind, not by the
entries a listing of it reads."""
IMPORTED = "import:"
"""What a top-level module a run imported is recorded under (executed.py): a helper of that name under
tests/, which leads sys.path, would shadow it, so ci.py's `contents` keys whether tests/ holds one."""


def opening(recorded: Callable[[str], None], root: str) -> None:
    """`os.open` wrapped, whose audit event names a path relative to `dir_fd` without the descriptor:
    such an open, once it has opened to read, is passed to `recorded` as what the descriptor it
    opened names (`placed`). Each existence or size probe (`PROBES`) is passed to `recorded` as the path it
    asks after (`probed`), there or not, which the record then keys (ci.py's `contents`): a fixture's
    arrival reruns its probers, and no other file (Main, 2026-10-29)."""
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
    """`probe` (`PROBES`), which passes the path it is asked about to `recorded`, as `PROBED` and that path,
    before it answers: but a listing tool's (`listing_tool`), whose probes, as its listings, look for what
    it then records as it runs, and one a probe makes in answering (pathlib's `is_dir` asks `stat`), whose
    caller's probe is already recorded."""

    @functools.wraps(probe)
    def asked(path, *arguments, **keywords):
        if getattr(placing, "probing", False):
            return probe(path, *arguments, **keywords)
        # A path relative to a directory descriptor (os.stat's dir_fd) is placed in the directory it names.
        at = keywords.get("dir_fd")
        base = os.readlink(f"/proc/self/fd/{at}") if at is not None else ""
        if not isinstance(path, int) and not listing_tool(sys._getframe(1).f_code.co_filename) and (
                name := placed(os.path.abspath(os.path.join(base, os.fsdecode(os.fspath(path)))),
                               root)) is not None:
            recorded(PROBED + name)
        placing.probing = True
        try:
            return probe(path, *arguments, **keywords)
        finally:
            placing.probing = False

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


def closed_imports(seen: set[str], root: str):
    """The import system's first finder once a process's log closed: a module of the checkout at `root` that
    `seen` does not name is not imported, so no code the record misses changes what the process did (Main,
    2026-10-30: a finalizer that execs a checkout file by hand past the audit hooks' clearing is out of
    contract)."""

    class Closed:
        @staticmethod
        def find_spec(name, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            origin = getattr(spec, "origin", None) or ""
            if checkout(origin, root) and origin[len(root):] not in seen:
                raise ImportError(f"{origin[len(root):]} was not imported before the run's record closed")

    return Closed


def imports(write: Callable[[str], None]):
    """The import system's first finder that passes each module imported by name, a statement's or
    `importlib.import_module`'s, to `write` as the top-level name a helper under tests/ would shadow
    (`IMPORTED`); it finds nothing itself."""

    class Imports:
        @staticmethod
        def find_spec(name, path=None, target=None):
            write(f"{IMPORTED}{name.partition('.')[0]}")

    return Imports


def follow() -> None:
    """Log this child for the process that started it, as it runs, once: though the environment processes
    its site-packages' `.pth` files twice (a uv environment's `site` adds the directory again), a second
    start finds the recorder holding its tool and leaves it. Where another tool holds that id the child
    records no files, and says so (`unfollowed`): its file then has no key, and the child starts as it would
    without the recorder."""
    holder = sys.monitoring.get_tool(FOLLOWER_TOOL)
    if holder == FOLLOWER:
        return
    directory, root = Path(os.environ[CHILDREN]), os.environ["DEW_EXECUTED_ROOT"]
    log = (directory / f"{os.getpid()}.log").open("a", buffering=1)
    logged: set[str] = set()
    seen: set[str] = set()

    def write(kind: str, value: object) -> None:
        line = f"{kind}\t{value}"
        if line not in logged:
            logged.add(line)
            log.write(line + "\n")

    def recorded(name: str) -> None:
        if checkout(os.path.join(root, name), root):
            seen.add(name)
        write("file", name)

    # A child given its parent's fraction holds one more of the parent's shares where it opens the GPU.
    write("owner", os.environ.get("DEW_EXECUTED_FILE", ""))
    write("inherited", int(os.environ.get(FRACTION, "") == os.environ.get("DEW_EXECUTED_SHARE", "")))
    write("began", time.time())
    for name in list(sys.modules):
        write("file", f"{IMPORTED}{name.partition('.')[0]}")

    def started(code, offset):
        if code.co_name == OPENED[0] and code.co_filename.endswith(OPENED[1]):
            # Registered after jax's own clean-up, so it reads the pool before the backend closes.
            if sys._getframe(1).f_locals.get("platform") in GPUS:
                atexit.register(held)
            return None
        name = code.co_filename
        if checkout(name, root):
            recorded(name[len(root):])
            if reads_backend(code):
                write("read", f"{name[len(root):]}:{code.co_qualname}")
        return sys.monitoring.DISABLE

    def held() -> None:
        import jax

        try:
            devices = jax.local_devices(backend="gpu")
        except RuntimeError:  # the backend failed to open, and holds no pool
            return
        stats = [device.memory_stats() or {} for device in devices]
        write("pool", sum(stat.get("peak_pool_bytes", stat.get("bytes_limit", 0)) for stat in stats))
        write("peak", max(stat.get("peak_bytes_in_use", 0) for stat in stats))

    def audited(event, args):
        if event in READS:
            name = read(event, args, root)
            if name is not None:
                write("file", name)
            return
        kind = start(event, args, str(directory))
        if kind == "followed":
            log.write("start\tfollowed\n")
        elif kind == "unfollowed":
            write("unfollowed", 1)

    def closing() -> None:
        write("ended", time.time())
        write("closed", 1)
        sys.meta_path.insert(0, closed_imports(seen, root))

    atexit.register(closing)
    if holder is not None:
        write("unfollowed", 1)
        return
    sys.meta_path.insert(0, imports(lambda item: write("file", item)))
    opening(lambda item: write("file", item), root)
    sys.monitoring.use_tool_id(FOLLOWER_TOOL, FOLLOWER)
    sys.monitoring.register_callback(FOLLOWER_TOOL, sys.monitoring.events.PY_START, started)
    sys.monitoring.set_events(FOLLOWER_TOOL, sys.monitoring.events.PY_START)
    sys.addaudithook(audited)


def logged(path: Path) -> dict:
    """A child's log read back: whose it is, the files it executed and read and the modules it imported, its
    reads of the backend, how many children it followed in turn, whether it started one no event sees or
    was not followed itself, its life, its GPU pool, and whether it closed."""
    record = {"file": "", "files": set(), "reads": set(), "expected": 0, "unfollowed": False, "began": 0.0,
              "ended": None, "pool": None, "peak": None, "inherited": False, "closed": False}
    for line in path.read_text().splitlines():
        kind, _, value = line.partition("\t")
        if kind == "owner":
            record["file"] = value
        elif kind == "file":
            record["files"].add(value)
        elif kind == "read":
            record["reads"].add(value)
        elif kind == "start":
            record["expected"] += 1
        elif kind == "unfollowed":
            record["unfollowed"] = True
        elif kind in ("began", "ended"):
            record[kind] = float(value)
        elif kind in ("pool", "peak"):
            record[kind] = int(value)
        elif kind == "inherited":
            record["inherited"] = value == "1"
        elif kind == "closed":
            record["closed"] = True
    return record


def gathered(directory: Path, expected: dict[str, int]) -> dict[str, dict]:
    """What each test file's children logged in `directory`, by the file that started them (`""`, every
    file's, for those started outside a file's tests), given how many children each file's own process
    started (`expected`): their files and reads, whether every start was followed and every child's log
    closed, the most GPU memory a child that took the parent's fraction held (`peak`), and the GPU pools they
    held at once, the parent's share counted apart (`shares`, the children that took its fraction) from the
    bytes of those that named their own. Read once the process group has exited, so every log is whole."""
    records = [logged(path) for path in directory.glob("*.log")] if directory.is_dir() else []
    found: dict[str, dict] = {}
    for name in {*expected, *(record["file"] for record in records)}:
        own = [record for record in records if record["file"] == name]
        starts = expected.get(name, 0) + sum(record["expected"] for record in own)
        whole = [record for record in own if record["closed"]]
        pooled = [record for record in whole if record["pool"]]
        inherited = [record for record in pooled if record["inherited"]]
        found[name] = {
            "files": {path for record in own for path in record["files"]},
            "reads": {site for record in own for site in record["reads"]},
            "followed": len(whole) == len(own) == starts and not any(r["unfollowed"] for r in own),
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


def recorded(path: Path) -> dict[str, dict]:
    """Each test file's record in the log at `path` and its children's (followed.py's `gathered`): the files
    it executed and read, every earlier file's up to the end of its last visit, `SHARED`'s whenever, and its
    own and `SHARED`'s children's (`depends`, None for a file without a key); its reads of the backend and its
    marks; and the GPU memory it held, its children's pools and shares. A file the log never names has no
    record."""
    text = path.read_text() if path.is_file() else ""
    # A line cut short by a kill names no event.
    lines = [line.split("\t", 1) for line in text.splitlines() if "\t" in line]
    closed = [SHARED, CLOSED] in lines
    starts = Counter(name for name, item in lines if item == f"{STARTED}followed")
    children = gathered(path.with_name(f"{path.name}.children"), dict(starts))
    blank = {"files": set(), "reads": set(), "followed": True, "peak": None, "shares": 0, "pools": 0}
    shared_child = children.get(SHARED, blank)
    visits = [(index, name) for index, (name, item) in enumerate(lines) if item == VISIT]
    files = list(dict.fromkeys(name for _, name in visits if name != SHARED))
    shared = [item for name, item in lines if name == SHARED]
    peaks = [int(item[len(PEAK):]) for item in shared if item.startswith(PEAK)]
    out = {}
    for file in files:
        # Up to the end of its last visit: the next owner's visit, or the log's end.
        last = max(index for index, name in visits if name == file)
        end = next((index for index, _ in visits if index > last), len(lines))
        before = lines[:end]
        child = children.get(file, blank)
        unfollowed = any(item == f"{STARTED}unfollowed" for _, item in before) or \
            f"{STARTED}unfollowed" in shared
        whole = closed and not unfollowed and child["followed"] and shared_child["followed"]
        inputs = {item for _, item in before if not item.startswith(EVIDENCE)} | {
            item for item in shared if not item.startswith(EVIDENCE)}
        held = [each for each in (*peaks, child["peak"]) if each is not None]
        out[file] = {
            "depends": sorted(inputs | child["files"] | shared_child["files"]) if whole else None,
            "reads": sorted({item[len(READ):] for name, item in lines if name in (file, SHARED)
                             and item.startswith(READ)} | child["reads"] | shared_child["reads"]),
            "peak": max(held) if len(files) == 1 and whole and held else None,
            "shares": child["shares"], "pools": child["pools"]}
    return out


if __name__ == "dew_followed":
    follow()

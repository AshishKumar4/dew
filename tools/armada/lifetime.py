"""The test cache's recorder of what a task's pytest process executed, read, imported and started, for its
whole life (row 2560): it starts before the interpreter's site, the environment's `.pth` files and pytest
(`boot`, which session.py's `pytest_run` launches with `-S`), and writes each event to an append-only log as
it happens, under the test file it is owed to (`owners`). The supervisor (ci.py's `tested`) reads the log
once the process group has exited (`recorded`), so what ran last, a late sessionfinish hook, an atexit, a
finalizer, is in it however late it ran, with no report frozen at a hook to outlive.

An event's owner is the test file whose tests run (executed.py's hooks switch it), or `SHARED`, the empty
name, for what runs for every file: collection, conftests, session- and package-scoped fixtures, and all
from the session's end on. A file's record (`recorded`) is everything the process did up to the end of the
file's last visit, its owner's and any earlier file's alike, as what ran before it may serve it from a
cache or a module it left, everything `SHARED` did, whenever, and its own children's and `SHARED`'s
children's records (followed.py). A process whose log never closed, or one with a start no event follows,
leaves the files it reached without a key. The standard library alone: it runs before site-packages are
on the path."""

import atexit
import os
import runpy
import sys
from multiprocessing.process import BaseProcess
from pathlib import Path

import followed
from followed import CLOSED, EVIDENCE, PEAK, READ, SHARED, STARTED, UNOPENED, VISIT
from session import DURATIONS

LOG = "DEW_EXECUTED_LOG"
"""Where the recorder writes its log, named by the supervisor (ci.py's `tested`)."""
TOOL = 3
"""sys.monitoring's tool ids 0-2 and 5 are the debugger's, coverage's, the profiler's and the optimizer's;
followed.py's children hold 4."""
CHILD_START = BaseProcess.start.__code__
"""Where every multiprocessing context starts a child: Python 3.12 audits no spawn or forkserver start."""

root = os.getcwd() + os.sep
owners: list[str] = [SHARED]
"""The owner of what runs now: a test file while its tests run (`switch`), with `SHARED` pushed over it while
a session-scoped fixture is made or ended (`push`)."""
readers: list[str] = []
"""Whose evidence a read of the backend is: a test file's, a test module's while it is collected, or every
file's (`SHARED`) where none is set and while a session fixture runs."""
seen: set[str] = set()
"""The checkout's files the log names: after it closes, no other may be imported (`closing`)."""
logged: set[tuple[str, str]] = set()
log = None
children = ""
"""Where this process's followed children log (`gathered`), beside the log; named in their environment once
the site is made, so the `.pth` that starts `follow` in a child (followed.py) never starts it here."""


def owner() -> str:
    """Whose an event is now."""
    return owners[-1]


def write(name: str, item: str) -> None:
    """Event `item` owed to `name`, once each, as it happens: a line of its own, so a process killed mid-run
    leaves every line it wrote before."""
    if (name, item) in logged:
        return
    logged.add((name, item))
    if not item.startswith(EVIDENCE) and checkout_file(item):
        seen.add(item)
    log.write(f"{name}\t{item}\n")


def checkout_file(item: str) -> bool:
    return item.endswith(".py") and ":" not in item


def switch(name: str) -> None:
    """The test file `name`'s tests run now (`SHARED` once the session ends), each code object it runs
    recorded again under it, and the visit logged: a file a session parameter revisits holds what ran
    between its visits (`recorded`)."""
    owners[0] = name
    log.write(f"{name}\t{VISIT}\n")
    if len(owners) == 1:
        os.environ["DEW_EXECUTED_FILE"] = name
    sys.monitoring.restart_events()


def push(name: str) -> None:
    """`name` owns what runs until the matching `pop`: `SHARED`, over a test file, for a session fixture."""
    owners.append(name)
    os.environ["DEW_EXECUTED_FILE"] = name
    sys.monitoring.restart_events()


def pop() -> None:
    owners.pop()
    os.environ["DEW_EXECUTED_FILE"] = owners[-1]
    sys.monitoring.restart_events()


def kept(read: str | None) -> None:
    # The durations record is split.py's, which cuts a split file from it, read or probed: the file's own
    # entries are its groups' input (ci.py's `cache_key`), and no other file's.
    if read is not None and read.removeprefix(followed.PROBED) != str(DURATIONS):
        write(owner(), read)


def started(code, offset):
    """sys.monitoring's PY_START: the checkout's code a run executes, once an owner, and where it reads the
    backend; a multiprocessing child's start, whose child no event follows."""
    if code is CHILD_START:
        write(owner(), f"{STARTED}unfollowed")
        return None
    if code.co_name == followed.OPENED[0] and code.co_filename.endswith(followed.OPENED[1]):
        # Registered after jax's own clean-up, so it reads the pool before the backend closes.
        if sys._getframe(1).f_locals.get("platform") in followed.GPUS:
            atexit.register(peaked)
        return None
    name = code.co_filename
    if not followed.checkout(name, root):
        return sys.monitoring.DISABLE
    write(owner(), name[len(root):])
    if Path(name).name == "conftest.py":
        return sys.monitoring.DISABLE
    if code.co_name == "<module>":
        # Each module body sees every reader it calls, however many bodies called it before.
        sys.monitoring.restart_events()
    if followed.reads_backend(code):
        site = f"{READ}{name[len(root):]}:{code.co_qualname}"
        if readers:
            write(readers[-1], site)
        # The callback runs in the reader's frame, under the bodies of the modules being imported.
        frame = sys._getframe(1)
        while frame is not None:
            body = frame.f_code.co_filename
            if frame.f_code.co_name == "<module>" and followed.checkout(body, root) and \
                    Path(body).name != "conftest.py":
                if Path(body).name.startswith("test_"):
                    write(body[len(root):], site)
                elif not readers:
                    write(SHARED, site)
            frame = frame.f_back
    return sys.monitoring.DISABLE


def audited(event, args):
    """A read the audit names, kept as the run's input; a process start, kept as how its owner's child ran."""
    if event in followed.READS:
        kept(followed.read(event, args, root))
        return
    kind = followed.start(event, args, children)
    if kind is not None:
        write(owner(), f"{STARTED}{kind}")


def peaked() -> None:
    memory = followed.gpu_memory()
    write(SHARED, f"{UNOPENED}{memory}" if isinstance(memory, str) else f"{PEAK}{memory[1]}")


def closing() -> None:
    """The log's last line, as the last atexit runs; past it, no checkout module it does not name imports."""
    write(SHARED, CLOSED)
    sys.meta_path.insert(0, followed.closed_imports(seen, root))


def boot() -> None:
    """Start recording, then run the module `python -m` would (sys.argv's first argument) with the rest as its
    arguments: the log opened and its closing registered before anything else registers an atexit, monitoring
    and the audit hook in place before the interpreter's site, the environment's `.pth` files and pytest run,
    and the modules already loaded the shared run's."""
    global log, children
    if LOG not in os.environ:
        # A targeted run (ci.py's `targeted`) records nothing: the site, then the module, as `-m` would.
        run()
        return
    path = Path(os.environ[LOG])
    log = path.open("a", buffering=1)
    atexit.register(closing)
    directory = path.with_name(f"{path.name}.children")
    directory.mkdir(exist_ok=True)
    children = str(directory)
    for name, module in list(sys.modules.items()):
        write(SHARED, f"{followed.IMPORTED}{name.partition('.')[0]}")
        if followed.checkout(file := getattr(module, "__file__", None) or "", root):
            write(SHARED, file[len(root):])
    sys.meta_path.insert(0, followed.imports(lambda item: write(owner(), item)))
    followed.opening(kept, root)
    sys.monitoring.use_tool_id(TOOL, "dew-test-cache")
    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, started)
    sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)
    sys.addaudithook(audited)
    run(followed_children)


def followed_children() -> None:
    """The children this process starts record themselves (followed.py), given the GPU share it was given."""
    os.environ |= {followed.CHILDREN: children, "DEW_EXECUTED_ROOT": root,
                   "DEW_EXECUTED_FOLLOWER": str(Path(followed.__file__).resolve()),
                   "DEW_EXECUTED_SHARE": os.environ.get(followed.FRACTION, ""), "DEW_EXECUTED_FILE": SHARED}
    followed.installed()


def run(installed=lambda: None) -> None:
    """The interpreter's site made, as `python` without -S makes it, `installed` called, and the module
    sys.argv names first run as `python -m` would, with the rest as its arguments."""
    import site

    site.main()
    installed()
    module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)

"""Every data file of a checkout opened, and every file of it written, while a task's tests run, by any
process, as inotify reports them: what a native reader opens (Rust's tokenizers and safetensors,
tensorstore, a child's C code) raises no Python audit event, so the test cache's records (`ci.py`'s `record`)
take their data inputs from here; and a run that writes an input it reads would leave a key over the bytes
it wrote, not the ones it read, so a row whose inputs the task wrote keeps no key (row 2559).

The task's supervisor holds the watch from before its pytest starts until after pytest and its process
group stopped (`ci.py`'s `tested`), so an open or a write at any time of the run counts: a later
session-finish hook's, one at exit, a thread's (train 239's review, 2026-10-28). What is opened is the input
of every test file the task ran, not of the one running at the time: a reader a session fixture or a
process-lived cache shares is opened under one file and serves the next without reopening its data.

Opens are watched under tests/ and src/ (`READS`): Python files are code, which a record takes from what
runs (`executed.py`), and a directory's own open is a listing, which the audit hook places by who listed it
(pytest's collection and the import system are no file's input). Writes are watched in every directory of
the checkout that can hold an input: the files the audit hook records as read are anywhere in it (a tool's
data, a page, a notebook, a workflow), and a file it saw opened to write is no read, so no record tells a
file it read and then rewrote. A written file is each file closed after being opened for writing, created,
deleted or moved into or out of a directory, and that directory too where its listing changed (a name that
is not hidden, as `ci.py`'s `contents` lists a directory). A directory that arrives during the run, made
or moved in, is watched at once and everything already in it is written, as it came with the run. Hidden
directories at the checkout's top but `hidden` (the ones a record places, followed.py's `HIDDEN_INPUTS`),
`__pycache__` and a build's `*.egg-info` hold no input and are not watched.

Where a directory could not be watched (fs.inotify.max_user_watches), the kernel's queue overflowed, or a
watched directory was deleted, moved or unmounted, taking its watch with it, the watch is blind: what was
opened or written then is not known, and no record keeps a key (owner, 2026-10-27)."""

import ctypes
import os
import select
import struct
import threading

OPEN, CLOSE_WRITE, MOVED_FROM, MOVED_TO, CREATE, DELETE = 0x20, 0x8, 0x40, 0x80, 0x100, 0x200
DELETE_SELF, MOVE_SELF, UNMOUNT, OVERFLOW, IGNORED = 0x400, 0x800, 0x2000, 0x4000, 0x8000
DIRECTORY = 0x40000000
ENTRIES = MOVED_FROM | MOVED_TO | CREATE | DELETE
"""The events of a name coming or going in a directory, which change its listing."""
MUTATED = CLOSE_WRITE | ENTRIES
"""The events that write a file or the directory it is in: not a write call's own (IN_MODIFY, one event
each syscall, which a large write would send past the queue's bound), but the close of a file opened for
writing."""
LOST = DELETE_SELF | MOVE_SELF | UNMOUNT | OVERFLOW | IGNORED
"""The events after which a watch no longer sees what it watched: its directory gone, moved or unmounted
(the kernel then drops the watch, IN_IGNORED), or events lost to the queue's overflow."""
EVENT = struct.Struct("iIII")
READS = ("tests", "src")
"""The checkout's directories whose opens of data files are watched."""
DRAIN_SECONDS = 0.2
"""How often the watch's events are read while the task runs: the kernel holds fs.inotify.max_queued_events
(16384 by default) before it overflows, which a task's opens could pass between two reads seconds apart."""
SKIPPED = ("tests/test_durations.json",)
"""The durations record is split.py's, which cuts a split file from it: the file's own entries are its
groups' input (`ci.py`'s `cache_key`), and no other file's."""


def built(name: str) -> bool:
    """Whether the entry `name` is a build's output, never an input: compiled bytecode or its directory, or an
    editable install's metadata (`ci.py`'s `built`)."""
    return name == "__pycache__" or name.endswith((".pyc", ".egg-info"))


class Opens:
    """The watch of every directory of `root`'s checkout that can hold an input (the module's text), read by a
    thread of its own until it is `closed`. `hidden` names the hidden entries at the top of the checkout that
    do (followed.py's `HIDDEN_INPUTS`)."""

    def __init__(self, root: str, hidden: tuple[str, ...] = ()) -> None:
        self.root, self.hidden = root, hidden
        self.blind = False
        self.opened: set[str] = set()
        self.written: set[str] = set()
        self.watched: dict[int, str] = {}
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            self.blind = True
            return
        self.watch(".", arrived=False)
        if self.blind:
            # No more watches (or a directory gone as it was looked at): the descriptor closed, so `closed`
            # neither waits on a reader nor reads one.
            os.close(self.fd)
            self.fd = -1
            return
        self.stopped = threading.Event()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def pruned(self, relative: str) -> bool:
        """Whether the directory `relative` to the checkout holds no input (the module's text)."""
        parts = relative.split(os.sep)
        top = len(parts) == 1 and parts[0].startswith(".") and parts[0] not in self.hidden
        return built(parts[-1]) or top

    def watch(self, relative: str, arrived: bool) -> None:
        """Directory `relative` to the checkout watched, and every directory under it that holds an input;
        with `arrived`, each entry in them written, as a directory that came with the run holds what it
        made. A directory that cannot be watched (the kernel holds no more watches, or it is gone) leaves the
        watch blind."""
        top = relative.split(os.sep)[0]
        reading = top in READS and not any(part.startswith(".") for part in relative.split(os.sep))
        directory = os.path.join(self.root, relative)
        mask = MUTATED | DELETE_SELF | MOVE_SELF | (OPEN if reading else 0)
        found = self.libc.inotify_add_watch(self.fd, directory.encode(), mask)
        if found < 0:
            self.blind = True
            return
        self.watched[found] = os.path.normpath(relative)
        try:
            entries = list(os.scandir(directory))
        except OSError:
            # Gone between its arrival and this look.
            self.blind = True
            return
        for entry in entries:
            inside = os.path.normpath(os.path.join(relative, entry.name))
            if arrived and not built(entry.name):
                self.written.add(inside)
            if entry.is_dir(follow_symlinks=False) and not self.pruned(inside):
                self.watch(inside, arrived)
                if self.blind:
                    return

    def read(self) -> None:
        while not self.stopped.wait(DRAIN_SECONDS):
            self.drain()

    def drain(self) -> None:
        """The events the kernel holds, read: each data file's open kept (`opened`), each file written and
        each directory whose listing changed (`written`), a lost watch blind."""
        # Read while the kernel holds events (select's readiness), so no read waits or finds none.
        while self.fd >= 0 and select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 1 << 16)
            at = 0
            while at < len(data) and not self.blind:
                watch, mask, _, length = EVENT.unpack_from(data, at)
                start = at + EVENT.size
                name = data[start:start + length].split(b"\0", 1)[0].decode()
                at += EVENT.size + length
                if mask & LOST:
                    self.blind = True
                elif watch in self.watched and name:
                    self.heard(self.watched[watch], name, mask)

    def heard(self, directory: str, name: str, mask: int) -> None:
        """What an event `mask` of entry `name` in watched `directory` (relative to the checkout) is: an open
        of a data file, or a write."""
        path = os.path.normpath(os.path.join(directory, name))
        opening = mask & OPEN and not mask & DIRECTORY and not name.endswith((".py", ".pyc"))
        if opening and path not in SKIPPED:
            self.opened.add(path)
        if mask & MUTATED and not built(name):
            self.written.add(path)
            if mask & ENTRIES and not name.startswith("."):
                self.written.add(directory)
            if mask & (CREATE | MOVED_TO) and mask & DIRECTORY and not self.pruned(path):
                self.watch(path, arrived=True)

    def closed(self) -> set[str] | None:
        """Every data file opened since the watch was set, relative to the checkout, read once the task's
        processes stopped, and every file written (`written`); None where the watch was blind at any time.
        The watch is closed."""
        if self.fd >= 0:
            self.stopped.set()
            self.reader.join()
        self.drain()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        return None if self.blind else self.opened

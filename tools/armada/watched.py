"""Every open of a data file under tests/ and src/ of a checkout while a task's tests run, by any process, as
inotify reports it: what a native reader opens (Rust's tokenizers and safetensors, tensorstore, a child's C
code) raises no Python audit event, so the test cache's records (`ci.py`'s `record`) take their data inputs
from here.

The task's supervisor holds the watch from before its pytest starts until after pytest and its process group
stopped (`ci.py`'s `tested`), so an open at any time of the run counts: a later session-finish hook's, one
at exit, a thread's (train 239's review, 2026-10-28). What is opened is the input of every test file the
task ran, not of the one running at the time: a reader a session fixture or a process-lived cache shares is
opened under one file and serves the next without reopening its data.

Python files are code, which a record takes from what runs (`executed.py`), and a directory's own open is a
listing, which the audit hook places by who listed it (pytest's collection and the import system are no
file's input). Where a directory could not be watched (fs.inotify.max_user_watches), the kernel's queue
overflowed, or a watched directory was deleted, moved or unmounted, taking its watch with it, the watch is
blind: what was opened then is not known, and no record keeps a key (owner, 2026-10-27)."""

import ctypes
import os
import select
import struct
import threading

MODIFY, OPEN, MOVED_TO, CREATE, DELETE_SELF, MOVE_SELF, UNMOUNT, OVERFLOW, IGNORED, ONLYDIR, DIRECTORY = (
    0x2, 0x20, 0x80, 0x100, 0x400, 0x800, 0x2000, 0x4000, 0x8000, 0x1000000, 0x40000000)
LOST = DELETE_SELF | MOVE_SELF | UNMOUNT | OVERFLOW | IGNORED
"""The events after which a watch no longer sees what it watched: its directory gone, moved or unmounted
(the kernel then drops the watch, IN_IGNORED), or events lost to the queue's overflow."""
EVENT = struct.Struct("iIII")
DRAIN_SECONDS = 0.2
"""How often the watch's events are read while the task runs: the kernel holds fs.inotify.max_queued_events
(16384 by default) before it overflows, which a task's opens could pass between two reads seconds apart."""
SKIPPED = ("tests/test_durations.json",)
"""The durations record is split.py's, which cuts a split file from it: the file's own entries are its
groups' input (`ci.py`'s `cache_key`), and no other file's."""


class Opens:
    """The watch of every directory under `root`'s tests/ and src/, but hidden ones and __pycache__, read by a
    thread of its own until it is `closed`."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.blind = False
        self.opened: set[str] = set()
        self.changed: set[str] = set()
        """The files the task's processes wrote or moved in over another, of any kind (`closed`)."""
        self.made: set[str] = set()
        """The files the task's processes made, which no run read before them."""
        self.watched: dict[int, str] = {}
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            self.blind = True
            return
        for top in ("tests", "src"):
            for directory, names, _ in os.walk(os.path.join(root, top)):
                names[:] = [name for name in names if name != "__pycache__" and not name.startswith(".")]
                watch = libc.inotify_add_watch(self.fd, directory.encode(),
                                               OPEN | MODIFY | CREATE | MOVED_TO | DELETE_SELF | MOVE_SELF)
                if watch < 0:
                    # The kernel holds no more watches (fs.inotify.max_user_watches): blind, its descriptor
                    # closed, so `closed` neither waits on a reader nor reads one.
                    self.blind = True
                    os.close(self.fd)
                    self.fd = -1
                    return
                self.watched[watch] = os.path.relpath(directory, root)
        self.stopped = threading.Event()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self) -> None:
        while not self.stopped.wait(DRAIN_SECONDS):
            self.drain()

    def drain(self) -> None:
        """The events the kernel holds, read: each data file's open kept (`opened`), a lost watch blind."""
        for watch, mask, name in events(self.fd):
            if mask & LOST:
                self.blind = True
                continue
            if watch not in self.watched or not name or mask & DIRECTORY:
                continue
            path = os.path.join(self.watched[watch], name)
            if mask & CREATE:
                self.made.add(path)
            if mask & (MODIFY | MOVED_TO):
                self.changed.add(path)
            if mask & OPEN and not name.endswith((".py", ".pyc")) and path not in SKIPPED:
                self.opened.add(path)

    def closed(self) -> set[str] | None:
        """Every data file opened since the watch was set, relative to the checkout, read once the task's
        processes stopped; None where the watch was blind at any time, or where the processes changed a file
        that was there before them and that they opened, or a module (row 2559: a key hashes the bytes a run
        left, not those it read), by any writer, a native one's among them. The watch is closed."""
        if self.fd >= 0:
            self.stopped.set()
            self.reader.join()
        self.drain()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        changed = self.changed - self.made
        read = changed & self.opened or {path for path in changed if path.endswith(".py")}
        return None if self.blind or read else self.opened


def events(fd: int) -> list[tuple[int, int, str]]:
    """The events inotify descriptor `fd` holds, read while the kernel holds any (select's readiness, so no
    read waits or finds none): each one's watch, mask and name; none where `fd` is closed."""
    found = []
    while fd >= 0 and select.select([fd], [], [], 0)[0]:
        data = os.read(fd, 1 << 16)
        at = 0
        while at < len(data):
            watch, mask, _, length = EVENT.unpack_from(data, at)
            start = at + EVENT.size
            found.append((watch, mask, data[start:start + length].split(b"\0", 1)[0].decode()))
            at += EVENT.size + length
    return found


class Writes:
    """The watch of one directory for the files that arrive in it, made or moved or linked in, read on demand
    (`gained`): JAX's compilation cache, whose programs a test file's run compiled (tests/remote_cache.py),
    known without listing a directory of hundreds of thousands (row 3496: 281 listings took 727 s of a
    1,074 s task). `watching` is false where the watch could not be made (no inotify, no such directory)."""

    def __init__(self, directory: str) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        start = getattr(libc, "inotify_init1", None)
        self.fd = start(os.O_NONBLOCK | os.O_CLOEXEC) if start is not None else -1
        watched = CREATE | MOVED_TO | ONLYDIR
        if self.fd >= 0 and libc.inotify_add_watch(self.fd, directory.encode(), watched) < 0:
            os.close(self.fd)
            self.fd = -1
        self.watching = self.fd >= 0

    def gained(self) -> set[str] | None:
        """The names that arrived in the directory since the last call; None where not every one is known:
        the kernel's queue overflowed meanwhile (fs.inotify.max_queued_events), or the watch was lost (its
        directory gone, moved or unmounted) or closed."""
        found = events(self.fd)
        if self.fd < 0 or any(mask & LOST for _, mask, _ in found):
            return None
        return {name for _, _, name in found if name}

    def close(self) -> None:
        """The watch ended, its descriptor closed: `gained` knows nothing after."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

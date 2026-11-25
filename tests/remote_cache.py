"""A shared layer under JAX's persistent compilation cache, for CI on armada and the GPU queue.

Each armada task and GPU pass starts on a cold machine, so the on-disk cache conftest turns on
(`dew._cache.enable_compilation_cache`) is always empty there, and the suite spends most of its time in XLA:
on the Colab A100 tier, compiling took 55% of every file's wall time. With DEW_XLA_CACHE_URL and
DEW_XLA_CACHE_TOKEN set (.armada.json), `install` keeps each test file's programs in that file's bundle on the
object store those name, through JAX's own cache directory: one GET as the file is collected writes its last
run's programs into the directory, and one PUT after it ran, where it compiled any program the bundle lacked
or did not use, sends the programs compiled under it and those of the fetched ones it opened. Bundles are kept
under jaxlib's version, the backend, the Python minor and the device's kind, beside JAX's own key of each
program and its flags.

JAX keeps a program as one file, `<key>-cache`, in `jax_compilation_cache_dir`, and with no
`jax_compilation_cache_max_size` (Dew sets none) it neither evicts nor stamps them
(jax/_src/lru_cache.py in jax 0.11.2), so the files a directory gains while a file's tests run are
the programs they compiled, which a watch of the directory reports as they arrive, and the programs JAX
loads are the files opened in it (tools/armada/watched.py's `Writes`), so no switch of bundle lists it. A
bundle keeps what its run compiled or loaded and nothing else, its chunks past those dropped from the Worker:
a bundle that kept every program it ever fetched only grew, recut groups borrowing their siblings' programs
and writing them under their own cut, to its cap (train 796, 2026-11-13: a four-test group of
test_attention_mosaic_compile.py fetched 6,283 programs, 3.1 GB, and two such groups on one container filled
its 20 GB disk). A bundle past the Worker's bound
(`MAX_BYTES`) is kept in chunks of it (`chunk`), each a body of its own, so a file keeps every program
it compiled: one bundle at the bound dropped the rest, which every run compiled afresh, and a bundle at
it was never sent again (2026-11-08: the A100 tier's parts compiled 1,767 programs afresh, 922 s,
their bundles read at 100.0 MB). A program another file's bundle brought in joins this file's bundle only
where it is compiled under it, since its open is seen where the directory gained it, not for whom. Each pytest
worker that keeps a bundle has a directory of its own (conftest), so the files a worker's bundle
gains are its own whole writes, never another worker's half-written one.

A bundle is a file's in one execution partition (conftest's `bundle`): a split file's group, or an
xdist worker, has its own, so partitions that compile apart never replace each other's programs. A
group with no bundle yet, its file's cut changed, seeds the programs of up to `SIBLINGS` groups of its
file's other cuts whose share of the file overlaps its own (`_siblings`), and writes its own bundle
under its own cut (2026-10-28: 565 of 1,182 recut groups restored nothing after CI's target moved). A
bundle and not a program at a time, as a test that runs a model eagerly compiles thousands of
programs of a few milliseconds each (14,068 of the A100 tier's 16,537, under 50 ms), quicker to
compile than to ask a Worker for: asking about each one took a DeepSeek drafting test from 103 s to
962 s. A file's bundle is its programs' few megabytes (the CPU lane's same-bits test, 6546
programs, 46 MB).

The remote can only make a task slower: a fetch or an upload that fails, or a local read or write
of the directory that fails, counts against it, and after `FAILURES` in a row it is left alone for
the rest of the process, so an outage costs a few timeouts. Being left alone is final and is
checked as each request is made: an answer that arrives after it, a request already under way,
does not take the remote back, and the uploads still queued are dropped unsent. Uploads run on a
background thread, and the process waits up to `DRAIN_SECONDS` for them at exit.
"""

import contextlib
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import queue
import re
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

FAILURES = 3
TIMEOUT_SECONDS = 30.0
DRAIN_SECONDS = 120.0


def _armada(name: str):
    """tools/armada's module `name`, loaded by its path (the suite imports no `tools` package)."""
    spec = importlib.util.spec_from_file_location(
        f"armada_{name}", Path(__file__).resolve().parents[1] / "tools" / "armada" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r2 = _armada("r2")  # the Worker's one client and its contract (tools/armada/r2.py)
HOST = _armada("host")  # the host's CPU class, which keys a bundle (tools/armada/host.py)
WATCHED = _armada("watched")  # inotify's one reader (tools/armada/watched.py's `Writes`)
GRADING = _armada("grading")  # which test file a node is of (tools/armada/grading.py's `owning`)

MAX_BYTES = r2.MAX_BYTES
"""The largest bundle sent or read: the Worker's bound (tools/armada/r2.json)."""
CHUNKS = 32
"""The most chunks of `MAX_BYTES` one file's bundle spans (`chunk`): a file's programs past them go
without."""
SIBLINGS = 3
"""The most other-cut groups a recut group seeds from, nearest overlap first, within one `MAX_BYTES`."""
SIBLING_KEYS = 64
"""The most bundle keys a recut group lists for its file's other cuts."""
GROUP = re.compile(r"^(?P<path>.+?)/(?P<index>\d+)-of-(?P<count>\d+)(?P<rest>(?:/.*)?)$")
"""A split group's bundle: its file, its group G of N (`bundle`) and any worker after it."""
SUFFIX = "-cache"
"""What JAX's cache names a program's file after its key (jax/_src/lru_cache.py)."""
STAGING = ".staging"
"""The directory within the cache where a restored program is written before it is linked in whole."""
UNREACHABLE = (OSError, r2.R2Error)
"""A request the Worker did not answer as asked (`r2.R2Error`), or a local read or write of the cache
directory that fails (an OSError)."""

logger = logging.getLogger(__name__)


class RemoteCache:
    """JAX's cache `directory`, each test file's programs also read from and written to `url`."""

    def __init__(self, directory: Path, url: str, token: str, prefix: str):
        self.directory, self.prefix = Path(directory), prefix.strip("/")
        self.store = r2.Store(url, token, "dew-ci-xla-cache", TIMEOUT_SECONDS)
        self.failures = 0
        self.given_up = False
        self.lock = threading.Lock()
        """Guards the failure count and the counters."""
        self.state = threading.Lock()
        """Guards the bundles' bookkeeping and the directory's view: a bundle's switch, its seeding, its
        sealing; held across no request."""
        self.file: str | None = None
        self.seen: set[str] | None = None
        """The programs in the directory when the current bundle's run began or last moved on: listed once,
        then kept by what arrives in it (`writes`)."""
        # The directory the watch watches, there before it (conftest's `enable_compilation_cache` makes it; a
        # caller that names a new one has it made here). One that cannot be made (a file in its place) cannot
        # be watched: the remote is left alone.
        with contextlib.suppress(OSError):
            self.directory.mkdir(parents=True, exist_ok=True)
        self.writes = WATCHED.Writes(str(self.directory))
        """What arrives in the directory, JAX's programs and the bundles' own (watched.py's `Writes`): a
        bundle's switch lists no directory of hundreds of thousands of programs (row 3496)."""
        self.written: dict[str, list[str]] = {}
        """The programs compiled while each bundle's tests ran, in the order they appeared."""
        self.used: dict[str, set[str]] = {}
        """The programs opened while each bundle's tests ran: JAX's loads of the fetched ones it took."""
        self.blind: set[str] = set()
        """The bundles while whose tests ran the watch lost events: what they opened is not known, so each
        keeps every program it fetched."""
        self.sealed: set[str] = set()
        """The bundles written back (`closed`): opens after it, its own reads among them, are no use of
        theirs."""
        self.fetched: dict[str, list[str]] = {}
        """The programs each bundle's GET brought, in the bundle's order."""
        self.held: dict[str, set[str]] = {}
        """The programs each bundle holds on the Worker, as read or last written."""
        self.queued: dict[str, set[str]] = {}
        """The compiled programs each bundle's last upload was made for: a bundle the cap cut short is not
        sent again until its tests compile another."""
        self.chunked: dict[str, list[list[str]]] = {}
        """The programs of each chunk of each bundle on the Worker, as read or last written: a chunk that
        holds the same is not sent again."""
        self.unrestored: set[str] = set()
        """The bundles whose fetched programs did not all reach the directory: never written back, so a
        partial restore does not replace a whole bundle."""
        # Daemon threads, which the interpreter does not wait for at exit; `drain` bounds the wait.
        self.uploads: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        """Each body to send, by its key on the Worker (a chunk's, `chunk`); None, a chunk to delete."""
        self.uploader = threading.Thread(target=self._uploading, daemon=True, name="xla-cache")
        self.uploader.start()
        self.programs_fetched = self.uploaded = 0
        """The programs the remote's bundles put in the directory, and the bundles it took."""
        self.bytes_fetched = self.bytes_uploaded = 0
        """The bundles' bytes read and sent, which the summary reports."""
        self.programs_dropped = 0
        """The programs a bundle held that were not the bytes written (`unpacked`), compiled again."""
        if not self.writes.watching:
            # No program a run compiles could be told from the rest without listing the directory at every
            # switch: the task compiles without the remote.
            self.given_up = True
            logger.warning("the shared XLA cache cannot watch %s; compiling without it", self.directory)

    @property
    def available(self) -> bool:
        return not self.given_up

    def _failed(self, error: Exception) -> None:
        with self.lock:
            self.failures += 1
            if self.failures >= FAILURES and not self.given_up:
                self.given_up = True
                logger.warning("the shared XLA cache is unreachable (%s); compiling without it", error)

    def _answered(self) -> None:
        """A request answered: the failures in a row end, unless the remote was given up meanwhile."""
        with self.lock:
            if not self.given_up:
                self.failures = 0

    def _key(self, file: str) -> str:
        return f"{self.prefix}/bundles/{file}"

    def _chunk(self, file: str, index: int) -> str:
        return chunk(self._key(file), index)

    def _programs(self) -> set[str]:
        return {path.name.removesuffix(SUFFIX) for path in self.directory.glob(f"*{SUFFIX}")}

    def _moved_on(self) -> None:
        """The programs the directory gained since the current bundle's run began or last moved on,
        written to its account (under `state`): those that arrived in it (`writes`), the directory listed
        again only where the watch's queue overflowed meanwhile."""
        gained = self.writes.gained()
        if self.file is None or self.seen is None:
            return
        if gained is None:
            programs = self._programs()
            fresh, self.seen = programs - self.seen, programs
            self.blind.add(self.file)
        else:
            arrived, opened = gained
            fresh = {name.removesuffix(SUFFIX) for name in arrived if name.endswith(SUFFIX)} - self.seen
            self.seen |= fresh
            if self.file not in self.sealed:
                self.used.setdefault(self.file, set()).update(
                    name.removesuffix(SUFFIX) for name in opened if name.endswith(SUFFIX))
        written = self.written.setdefault(self.file, [])
        written += sorted(fresh - set(written))

    def mark(self) -> tuple[str | None, int]:
        """Where the current bundle's compiled programs stand now (`compiled_since`), read through the watch
        its account already keeps (`_moved_on`): the local watch's, which a remote given up keeps, so a
        test that compiled while the Worker was unreachable is still timed cold."""
        if not self.writes.watching:
            return None, 0
        with self.state:
            self._moved_on()
            return self.file, len(self.written.get(self.file, [])) if self.file else 0

    def compiled_since(self, mark: tuple[str | None, int]) -> int | None:
        """The programs compiled since `mark` (`mark`) under the same bundle: a program the shared cache did
        not hold, so the test that ran meanwhile ran cold; None where it is not known (no shared cache,
        the bundle moved on, the watch lost events)."""
        if not self.writes.watching or mark[0] is None:
            return None
        with self.state:
            self._moved_on()
            if self.file != mark[0] or mark[0] in self.blind:
                return None
            return len(self.written.get(mark[0], [])) - mark[1]

    def opened(self, file: str) -> None:
        """Bundle `file` in the local cache, and the programs compiled from here on its: its account kept
        while the directory is watched, its programs fetched while the remote is available."""
        if not self.writes.watching:
            return
        try:
            with self.state:
                if file != self.file:
                    self._moved_on()
                    self.file = file
                    if self.seen is None:
                        self.seen = self._programs()
                elif file in self.sealed:
                    self._moved_on()
                # A bundle written back and now running again (A, B, A) counts what it opens from here on: the
                # opens since its close, its own reads among them, went to it while sealed (review).
                self.sealed.discard(file)
                if file in self.held or not self.available:
                    return
                self.held[file] = set()
            # What a prepare fetched into the directory ahead for it (`prefetch`), its values there already,
            # with no request; else the Worker's bundle, or another cut's for a recut group.
            if (record := taken_prefetched(self.directory, self._key(file))) is not None:
                with self.state:
                    self.chunked[file] = record["chunks"]
                entries = [(key, None) for keys in record["chunks"] for key in keys]
                borrowed = [(key, None) for key in record["borrowed"]]
            else:
                entries = self._fetched(file)
                borrowed = [] if entries else self._siblings(file)
        except UNREACHABLE as error:
            self._failed(error)
            return
        except ValueError as error:  # an answer that is no whole bundle is a miss
            self._failed(error)
            return
        keys, new = [], 0
        with self.state:
            try:
                for key, value in entries:
                    keys.append(key)
                    # One the directory had already (another file's tests compiled it meanwhile, or a
                    # prepare fetched it, its value None) stays among the programs the current file's
                    # run gained.
                    if value is not None and self._seeded(key, value):
                        new += 1
                        self.seen.add(key)
                # Another cut's programs: fetched for this group, which holds none of them yet, so
                # `closed` writes them into its own bundle under its own cut, once.
                for key, value in borrowed:
                    keys.append(key)
                    if value is not None and self._seeded(key, value):
                        new += 1
                        self.seen.add(key)
            except UNREACHABLE as error:
                self.unrestored.add(file)
                self._failed(error)
            self.fetched[file] = keys
            self.held[file] = {key for key, _ in entries}
        if keys and file not in self.unrestored:
            self._answered()
        with self.lock:
            self.programs_fetched += new

    def _fetched(self, file: str) -> list[tuple[str, bytes]]:
        """The entries of `file`'s bundle on the Worker, each of its chunks in turn up to the first it
        lacks; none where it has none or the remote was given up. An answer counts as one only once its
        programs are in the directory (`opened`)."""
        entries: list[tuple[str, bytes]] = []
        chunks: list[list[str]] = []
        for index in range(CHUNKS):
            if self.given_up:  # checked as each request is made
                break
            bundle = self.store.get(self._chunk(file, index), limit=MAX_BYTES)
            if bundle is None:
                break
            with self.lock:
                self.bytes_fetched += len(bundle)
            found, dropped = unpacked(bundle)
            with self.lock:
                self.programs_dropped += dropped
            chunks.append([key for key, _ in found])
            entries += found
        with self.state:
            self.chunked[file] = chunks
        return entries

    def _siblings(self, file: str) -> list[tuple[str, bytes]]:
        """`siblings` of `file` on the Worker, unless the remote was given up, each bundle read counted."""
        entries, read, dropped = siblings(self.store, self._key, file, lambda: self.given_up)
        with self.lock:
            self.bytes_fetched += read
            self.programs_dropped += dropped
        return entries

    def _seeded(self, key: str, value: bytes) -> bool:
        return seeded(self.directory, key, value)

    def closed(self, file: str) -> None:
        """Bundle `file` written back, where its tests compiled a program it lacked or left one it holds
        unused: the programs fetched for it that its run opened (every one, where the watch lost what it
        opened), in its bundle's order, then those compiled under it, in chunks of at most `MAX_BYTES`
        (`CHUNKS` of them), a program past the bound going without; each chunk sent whose programs differ
        from what the Worker holds there, so a bundle that grew sends its last chunks alone, and each chunk it
        no longer fills deleted; nothing sent where its run used none, or where what it keeps is what the
        Worker holds."""
        if not self.available or file in self.unrestored:
            return
        try:
            with self.state:
                if file == self.file:
                    self._moved_on()
                written = list(self.written.get(file, []))
                used = self.used.get(file, set())
                every = self.fetched.get(file, [])
                loaded = [key for key in every if key in used]
                # A run that loaded under half of what it fetched ran a part of its file's tests (a test
                # node's row, a `-k`, a recycled row's later process, a site's map): it keeps what it did not
                # load, which the rest of its file's tests load. Pruned so, each part rewrote the bundle the
                # next part's run had to compile again and rewrite (2026-11-24: one file's bundle written
                # 7,459 times in a day, as often as it was read). A whole run prunes what it did not load.
                # One that loaded none of it and compiled as many as it held found it stale (its file's code
                # changed): it prunes.
                keeps = file in self.blind or (2 * len(loaded) < len(every)
                                               and (bool(loaded) or len(written) < len(every)))
                fetched = every if keeps else loaded
                self.sealed.add(file)
                # A run that loaded none of it and compiled none writes nothing, whatever it fetched (a recut
                # group's borrowed programs, `_siblings`, among them).
                if not loaded and not written and file not in self.blind:
                    return
                # What it keeps: the fetched programs its run loaded (a recut group's, borrowed from its
                # file's other cuts, `_siblings`, written under its own cut once), and those it compiled.
                wanted = {*written, *fetched}
                if not wanted or wanted == self.queued.get(file) or wanted == self.held.get(file):
                    return
            # Laid out by the programs' sizes alone, no payload read: one bundle's payloads read once, for the
            # chunks it sends.
            layout, displaced = self._layout([*fetched, *written])
            # A kept bundle that leaves no room for a program its run compiled drops what the run did not
            # load, where that makes room for it.
            if keeps and displaced & set(written):
                pruned, still = self._layout([*loaded, *written])
                if displaced & set(written) - still:
                    fetched, layout = loaded, pruned
                    wanted = {*written, *fetched}
        except UNREACHABLE as error:
            self._failed(error)
            return
        kept = {key for keys in layout for key in keys}
        with self.state:
            if not kept or kept == self.held.get(file, set()):
                self.queued[file] = wanted
                return  # nothing new fits: the Worker's bundle stays as it is
            held = self.chunked.get(file, [])
        changed = [index for index, keys in enumerate(layout) if index >= len(held) or keys != held[index]]
        # Its changed chunks' payloads read before anything is held sent: a close that cannot read one sends
        # nothing, and the next sends it.
        try:
            bodies = [(index, packed(self._payloads(layout[index]))) for index in changed]
        except UNREACHABLE as error:
            self._failed(error)
            return
        with self.state:
            self.queued[file] = wanted
            self.held[file] = kept
            self.chunked[file] = [list(keys) for keys in layout]
        for index, body in bodies:
            self.uploads.put((self._chunk(file, index), body))
        # The chunks it held past those it now fills, deleted, so the next fetch ends where this bundle does.
        for index in range(len(layout), len(held)):
            self.uploads.put((self._chunk(file, index), None))

    def _payloads(self, keys: list[str]) -> list[tuple[str, bytes]]:
        """`keys`' programs as the directory holds them. One gone since its layout (JAX's cache evicts) or
        that cannot be read fails the whole (OSError): the close sends nothing and holds nothing, so the
        next lays its bundle out again."""
        return [(key, (self.directory / f"{key}{SUFFIX}").read_bytes()) for key in keys]

    def _layout(self, keys: list[str]) -> tuple[list[list[str]], set[str]]:
        """`keys`' programs of the directory laid out by their sizes in chunks of at most `MAX_BYTES`
        (`CHUNKS` of them), in order, no payload read; and those a full layout displaced. A program past the
        bound, which no chunk holds, goes without and is none displaced."""
        layout: list[list[str]] = [[]]
        size, displaced = 0, set()
        for key in dict.fromkeys(keys):
            path = self.directory / f"{key}{SUFFIX}"
            if not path.exists():
                continue
            cost = HEADER.size + len(key.encode()) + path.stat().st_size
            if cost > MAX_BYTES:  # a program past the bound goes without; the rest may fit
                continue
            if size + cost > MAX_BYTES:
                if len(layout) == CHUNKS:
                    displaced.add(key)
                    continue
                layout.append([])
                size = 0
            size += cost
            layout[-1].append(key)
        return layout, displaced

    def _uploading(self) -> None:
        while True:
            key, bundle = self.uploads.get()
            try:
                if not self.given_up and bundle is None:  # a chunk the bundle no longer fills
                    self.store.delete(key)
                    self._answered()
                elif not self.given_up:  # checked as each request is made: a queued bundle goes unsent
                    self.store.put(key, bundle)
                    self._answered()
                    with self.lock:
                        self.uploaded += 1
                        self.bytes_uploaded += len(bundle)
            except (*UNREACHABLE, ValueError) as error:  # a bundle that does not reach it is compiled again
                self._failed(error)
            finally:
                self.uploads.task_done()

    def summary(self) -> str:
        """What the remote did for this process, which a task's log ends with: a slow task whose
        compilations missed it says so."""
        used = sum(len(set(self.fetched.get(file, [])) & self.used.get(file, set())) for file in self.fetched)
        return (f"shared XLA cache: {self.programs_fetched} programs fetched "
                f"({self.bytes_fetched / 1e6:.1f} MB read), {used} of those listed loaded, "
                f"{self.uploaded} bundles uploaded "
                f"({self.bytes_uploaded / 1e6:.1f} MB)"
                + (f", {self.programs_dropped} fetched programs dropped as not the bytes written"
                   if self.programs_dropped else "")
                + ("" if self.available else ", given up as unreachable"))

    def drain(self, seconds: float = DRAIN_SECONDS) -> None:
        """Every bundle written back, waiting up to `seconds` for the uploads, at the end of pytest's
        session; what is left is abandoned with the process, as it is where the upload thread
        stopped on an error it does not expect, which its own traceback reports."""
        if self.file is not None:
            self.closed(self.file)
        for file in list(self.written):
            self.closed(file)
        began = time.monotonic()
        while (self.uploads.unfinished_tasks and self.uploader.is_alive()
               and time.monotonic() - began < seconds):
            time.sleep(0.1)
        self.writes.close()


def bundle(node: str, group: str | None, worker: str | None) -> str:
    """The bundle of test `node`'s file in its execution partition: a split file's group
    (tools/armada/split.py's `G/N`) and an xdist worker compile apart, so each keeps its own."""
    parts = (group and group.replace("/", "-of-"), worker)
    return "/".join([GRADING.owning(node), *(part for part in parts if part)])


def seeded(directory: Path, key: str, value: bytes) -> bool:
    """`value` as program `key`'s file, written to a name of its own beside the directory's programs (its
    STAGING, which the watch does not see, so a restored program is one event, not two) and linked in
    whole, so a reader sees all of it or none; False, writing nothing, where the directory has it
    already."""
    staging = directory / STAGING
    staging.mkdir(exist_ok=True)
    handle, staged = tempfile.mkstemp(prefix=".", suffix=".partial", dir=staging)
    try:
        with os.fdopen(handle, "wb") as written:
            written.write(value)
        os.link(staged, directory / f"{key}{SUFFIX}")
    except FileExistsError:
        return False
    finally:
        os.unlink(staged)
    return True


def siblings(store: r2.Store, key: Callable[[str], str], file: str,
             stopped: Callable[[], bool]) -> tuple[list[tuple[str, bytes]], int, int]:
    """For split group `file` with no bundle of its own, the entries of up to SIBLINGS groups of its file's
    other cuts whose share of the file overlaps its own, the nearest overlap first, together within
    MAX_BYTES, the bytes read and the programs dropped (`unpacked`); none for a file that is no split group,
    nor once `stopped`. `key` is a bundle's key on the Worker."""
    group = GROUP.match(file)
    if group is None or stopped():
        return [], 0, 0
    index, count = int(group["index"]), int(group["count"])
    start, end = (index - 1) / count, index / count
    overlaps = []
    for name in itertools.islice(store.keys(f"{key(group['path'])}/"), SIBLING_KEYS):
        other = GROUP.match(f"{group['path']}/{name}")
        if other is None or other["rest"] != group["rest"] or int(other["count"]) == count:
            continue
        at, of = int(other["index"]), int(other["count"])
        low, high = (at - 1) / of, at / of
        if low < end and high > start:
            overlaps.append((min(end, high) - max(start, low), name))
    entries, left, dropped = [], MAX_BYTES, 0
    for _, name in sorted(overlaps, reverse=True)[:SIBLINGS]:
        if stopped() or left <= 0:
            break
        try:
            bundle = store.get(key(f"{group['path']}/{name}"), limit=left)
        except ValueError:  # past what is left, or no whole bundle: the next may fit
            continue
        if bundle is not None:
            left -= len(bundle)
            found, lost = unpacked(bundle)
            entries += found
            dropped += lost
    return entries, MAX_BYTES - left, dropped


PREFETCHED = ".prefetched"
"""In a cache directory, a record of each bundle a prepare fetched into it (`prefetch`), by its key's
digest: the programs of each of its chunks, or, for a recut group with none of its own, those it borrowed
from its file's other cuts (`siblings`). A run's `opened` restores the bundle from the directory with no
request and removes its record, so a later run of it asks the Worker again (`taken_prefetched`)."""


def prefetched_path(directory: Path, key: str) -> Path:
    return directory / PREFETCHED / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def taken_prefetched(directory: Path, key: str) -> dict | None:
    """The record of bundle `key` a prepare fetched into `directory` (`prefetch`), removed as it is read:
    `chunks` and `borrowed`; None where none is."""
    path = prefetched_path(directory, key)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    path.unlink()
    return record


def prefetch(directory: Path, url: str, token: str, prefix: str, bundles: list[str]) -> tuple[int, int]:
    """`bundles` (each a test file's in its partition, `bundle`) fetched from the Worker into `directory`,
    each chunk of each, or, for a recut group with no bundle of its own, its file's other cuts' it overlaps
    (`siblings`), each recorded there (`PREFETCHED`), so a run on a GPU that opens them finds them there and
    asks nothing (Main, 2026-11-08: a cold on-demand A100 started each part's files fetching them as they
    ran, and a tier's groups are recut as its GPU seconds move): the programs written. A remote that cannot
    be reached stops it where it is, the bundles fetched so far recorded, and the run fetches the rest
    itself. How many programs it wrote, and how many it dropped as not the bytes written (`unpacked`)."""
    store = r2.Store(url, token, "dew-ci-xla-cache", TIMEOUT_SECONDS)
    directory.mkdir(parents=True, exist_ok=True)

    def key(name: str) -> str:
        return f"{prefix.strip('/')}/bundles/{name}"

    written = dropped = 0
    try:
        for name in bundles:
            chunks, borrowed = [], []
            for index in range(CHUNKS):
                body = store.get(chunk(key(name), index), limit=MAX_BYTES)
                if body is None:
                    break
                entries, lost = unpacked(body)
                dropped += lost
                written += sum(seeded(directory, program, value) for program, value in entries)
                chunks.append([program for program, _ in entries])
            if not chunks:
                entries, _, lost = siblings(store, key, name, lambda: False)
                dropped += lost
                written += sum(seeded(directory, program, value) for program, value in entries)
                borrowed = [program for program, _ in entries]
            if chunks or borrowed:
                path = prefetched_path(directory, key(name))
                path.parent.mkdir(exist_ok=True)
                staged = path.with_suffix(".partial")
                staged.write_text(json.dumps({"chunks": chunks, "borrowed": borrowed}))
                staged.replace(path)
    except UNREACHABLE as error:
        logger.warning("the shared XLA cache is unreachable (%s); the run fetches the rest", error)
    return written, dropped


def chunk(key: str, index: int) -> str:
    """The Worker's key of chunk `index` of the bundle at `key`: the first is the bundle's own, so a
    bundle within one chunk is where it always was; each next is `<key>/=<index>`, which no split
    group's name (`GROUP`) takes for one of its file's cuts."""
    return key if index == 0 else f"{key}/={index}"


HEADER = struct.Struct(">IQ32s")
"""A bundle entry's key's and value's lengths and its value's sha256."""
FORMAT = "digested"
"""The bundles' format, the first part of their keys (`prefix`): each entry's value digested in its header
(row 9797), so no bundle written without one is read."""


def packed(entries: list[tuple[str, bytes]]) -> bytes:
    """`entries` as one bundle: each key's and value's length and the value's sha256, then the key and the
    value."""
    return b"".join(HEADER.pack(len(key.encode()), len(value), hashlib.sha256(value).digest()) + key.encode()
                    + value for key, value in entries)


def unpacked(bundle: bytes) -> tuple[list[tuple[str, bytes]], int]:
    """The `(key, value)` entries of a `packed` bundle and how many it dropped, or ValueError where it is
    not one whole: more
    than `MAX_BYTES`, an entry its lengths run past, or a key that is no file name in one directory
    (empty, `.` or `..`, a separator or NUL in it, or too long for its `-cache` file's name). An entry whose
    value is not the bytes its digest was taken of is dropped and said, its program compiled again: JAX's
    values carry no checksum, and XLA loads wrong bytes within an intact frame as they are (row 9797: a
    train's base run aborted in jaxlib reading back a cached program's result)."""
    if len(bundle) > MAX_BYTES:
        raise ValueError(f"a bundle of more than {MAX_BYTES} bytes")
    entries, at, dropped = [], 0, 0
    while at < len(bundle):
        if at + HEADER.size > len(bundle):
            raise ValueError(f"a bundle's entry header at {at} runs past its {len(bundle)} bytes")
        keyed, sized, digest = HEADER.unpack_from(bundle, at)
        at += HEADER.size
        if at + keyed + sized > len(bundle):
            raise ValueError(f"a bundle's entry at {at} runs past its {len(bundle)} bytes")
        key = bundle[at:at + keyed].decode()
        if (key in ("", ".", "..") or any(character in key for character in "/\\\0")
                or len(f"{key}{SUFFIX}".encode()) > 255):
            raise ValueError(f"a bundle's key {key[:80]!r} is no file name")
        value = bundle[at + keyed:at + keyed + sized]
        at += keyed + sized
        if hashlib.sha256(value).digest() != digest:
            logger.warning("the shared XLA cache's program %s is not the bytes it was written with; it is "
                           "dropped and compiled again", key)
            dropped += 1
            continue
        entries.append((key, value))
    return entries, dropped


def prefix(kind: str | None = None, backend: str | None = None) -> str:
    """The bundles' leading parts (their `FORMAT`; the device's `kind` and the `backend` this process's first
    device's, or those a prepare names for its node's GPU, which it cannot see): what a cached executable is
    only valid for beside JAX's own key, and the device's kind, whose programs are another file's on another
    GPU. jax's own version is not among them: JAX's key holds the program and jaxlib's version
    (jax/_src/cache_key.py), so a release of jax alone that lowers a program differently keys it anew, and
    one that does not (a post-release of Dew's patch stack) keeps every file's programs. The Python minor is
    among them because jax 0.11.2 names 3.14's stdlib zstd codec "zlib" in its key
    (`dew._cache.default_compilation_cache_dir`); and the host's CPU class (`HOST.cpu_class`), on every
    backend, as XLA:CPU compiles for the features of the host that compiles and a GPU lane's bundles hold
    CPU executables too (its host-side computations): armada's hosts span classes, and a program from
    another class need not link here, after which the process compiles nothing more."""
    import jax
    import jaxlib

    kind = "".join(character if character.isalnum() else "-"
                   for character in (kind or jax.devices()[0].device_kind))
    backend = backend or jax.default_backend()
    minor = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"{FORMAT}/jaxlib{jaxlib.__version__}/{backend}/{minor}/{kind}/{HOST.cpu_class()}"


def install(url: str, token: str) -> RemoteCache | None:
    """A `RemoteCache` at `url` over the cache directory JAX is configured with, or None, logged,
    where JAX keeps no local directory (the cache off, none set, or one on a remote store) or `url`
    is no http(s) address."""
    import jax

    directory = jax.config.jax_compilation_cache_dir
    if not jax.config.jax_enable_compilation_cache or not directory or "://" in directory:
        logger.warning("the shared XLA cache needs JAX's cache on in a local directory, not %r", directory)
        return None
    try:
        address = urllib.parse.urlparse(url)
        # Reading the port raises here, not at the first request, where it is no number.
        valid = address.scheme in ("http", "https") and bool(address.hostname) and address.port != 0
    except ValueError:
        valid = False
    if not valid:
        logger.warning("the shared XLA cache's URL %r is no http(s) address", url)
        return None
    return RemoteCache(Path(directory), url, token, prefix())


def requested() -> bool:
    """Whether this process's environment asks for the shared cache: DEW_XLA_CACHE_URL and
    DEW_XLA_CACHE_TOKEN both set (`install_from_environment`)."""
    return bool(os.environ.get("DEW_XLA_CACHE_URL") and os.environ.get("DEW_XLA_CACHE_TOKEN"))


def install_from_environment() -> RemoteCache | None:
    """`install` where the environment asks for the shared cache (`requested`). The local cache keeps
    no LRU bound: a task's container is short-lived, and the bound makes JAX lock and stamp its index
    on every compilation. The remote's bound is the bucket's 14-day expiry."""
    if not requested():
        return None
    return install(os.environ["DEW_XLA_CACHE_URL"], os.environ["DEW_XLA_CACHE_TOKEN"])


def main(arguments: list[str]) -> int:
    """`python tests/remote_cache.py prefetch TOPOLOGY ITEM...`: a GPU job's prepare, on its node's CPU,
    fetching the bundles of each of its run's items (a test file, or a group of one as CI names it,
    `<file>#G/N`) into the cache directory its run reads (`dew._cache.default_compilation_cache_dir`), keyed
    for the node's GPU as its serialized TOPOLOGY names it (a GPU node's session hands a prepare its node's,
    $DEW_NODE_TOPOLOGY)."""
    command, topology, *items = arguments
    if command != "prefetch" or not requested():
        raise SystemExit("usage: remote_cache.py prefetch TOPOLOGY ITEM..., with DEW_XLA_CACHE_URL and "
                         "DEW_XLA_CACHE_TOKEN set")
    from jax.experimental import topologies

    from dew._cache import default_compilation_cache_dir
    from dew.nn.kernels.generation import compile_only_gpus

    # The CUDA plugin, which a serialized topology deserializes through, initialized with no GPU and no
    # client, as a compile-only target initializes it (`dew.training.ahead.host_gpus`).
    compile_only_gpus("a100", 1)
    kind = topologies.TopologyDescription.deserialize(Path(topology).read_bytes()).devices[0].device_kind
    bundles = [bundle(file, group or None, None)
               for file, _, group in (item.partition("#") for item in items)]
    written, dropped = prefetch(Path(default_compilation_cache_dir()), os.environ["DEW_XLA_CACHE_URL"],
                                os.environ["DEW_XLA_CACHE_TOKEN"], prefix(kind, "gpu"), bundles)
    print(f"shared XLA cache: {written} programs fetched ahead for {len(bundles)} bundles"
          + (f", {dropped} dropped as not the bytes written" if dropped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
